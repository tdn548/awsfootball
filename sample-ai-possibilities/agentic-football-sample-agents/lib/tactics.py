"""Pre-computed tactical analysis for AI soccer agents.

Two entry points, both opt-in via create_invoke_handler():

  - fast_path(game_state, team_id, my_player_id, position_label)
      Returns commands for situations with an obvious answer (clear shot,
      keeper distribution, defender under pressure), or None to defer to the
      LLM. Skipping the model on these ticks removes its latency exactly when
      a late decision costs the most.

  - tactical_insights(game_state, team_id, my_player_id, position_label)
      Returns extra text appended to the state summary: shot quality, a
      recommended aim, the safest progressive pass, and a match plan based on
      the score. Small models choose much better from numbers than from raw
      coordinates, and computing them here costs no extra round trip.
"""

from __future__ import annotations

import math

from state import (
    _is_my_team,
    _player_idx,
    dist,
    get_goal_positions,
    get_possession_info,
    holder_is_me,
)

GOAL_HALF_WIDTH = 5.0
LANE_RADIUS = 4.0
SHOOT_QUALITY_THRESHOLD = 0.45
CLOSE_RANGE = 14.0
UNDER_PRESSURE = 5.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _xy(p: dict) -> dict:
    pos = p.get("position", {}) if "position" in p else p
    return {"x": pos.get("x", 0), "y": pos.get("y", 0)}


def _point_segment_distance(p: dict, a: dict, b: dict) -> float:
    ax, ay, bx, by = a["x"], a["y"], b["x"], b["y"]
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return dist(p, a)
    t = max(0.0, min(1.0, ((p["x"] - ax) * dx + (p["y"] - ay) * dy) / length_sq))
    return dist(p, {"x": ax + t * dx, "y": ay + t * dy})


def lane_blockers(a: dict, b: dict, opponents: list, radius: float = LANE_RADIUS) -> int:
    """Number of opponents standing within `radius` of the straight line a→b."""
    return sum(1 for o in opponents if _point_segment_distance(_xy(o), a, b) < radius)


def _split(players: list, team_id: int) -> tuple[list, list]:
    mine = [p for p in players if _is_my_team(p, team_id)]
    opps = [p for p in players if not _is_my_team(p, team_id)]
    return mine, opps


def _opponent_keeper(opps: list, opp_goal_x: float):
    keeper = next((p for p in opps if _player_idx(p) == 0), None)
    if keeper is None and opps:
        keeper = min(opps, key=lambda p: abs(_xy(p)["x"] - opp_goal_x))
    return keeper


def _nearest_distance(pos: dict, players: list) -> float:
    return min((dist(pos, _xy(p)) for p in players), default=99.0)


# ---------------------------------------------------------------------------
# Shooting
# ---------------------------------------------------------------------------

def shot_quality(me_pos: dict, opp_goal_x: float, keeper, opps: list) -> float:
    """0.0-1.0 estimate of a shot's chance, from distance, angle, keeper and blockers."""
    goal = {"x": opp_goal_x, "y": 0.0}
    d = dist(me_pos, goal)
    distance_factor = max(0.0, 1.0 - d / 45.0)
    angle_factor = min(1.0, math.atan2(GOAL_HALF_WIDTH, max(d, 1.0)) / 0.25)
    # A wide angle (|y| large) narrows the visible goal.
    angle_factor *= max(0.3, 1.0 - abs(me_pos["y"]) / 30.0)

    keeper_factor = 0.0
    if keeper is not None:
        k = _xy(keeper)
        keeper_factor = min(1.0, abs(k["y"]) / 6.0) * 0.15
        keeper_factor += min(1.0, abs(k["x"] - opp_goal_x) / 12.0) * 0.1

    outfield = [o for o in opps if o is not keeper]
    blockers = lane_blockers(me_pos, goal, outfield, radius=3.0)

    q = distance_factor * 0.55 + angle_factor * 0.25 + keeper_factor - 0.2 * blockers
    return round(max(0.0, min(1.0, q)), 2)


def choose_shot(me_pos: dict, keeper, team_id: int, opp_goal_x: float) -> tuple[str, float]:
    """Pick (aim_location, power): the corner on the side the keeper is not covering.

    Left/right is taken from the shooter's view facing the opponent goal, so
    HOME (attacking +x) has left at +y and AWAY has left at -y. Bottom corners
    for close range (harder to save low), top corners from distance.
    """
    keeper_y = _xy(keeper)["y"] if keeper is not None else 0.0
    if keeper_y > 1.0:
        target_y = -1.0
    elif keeper_y < -1.0:
        target_y = 1.0
    else:
        # Keeper central: aim for the far post.
        target_y = -1.0 if me_pos["y"] > 0 else 1.0

    left_is_positive_y = team_id == 0
    side = "L" if (target_y > 0) == left_is_positive_y else "R"

    d = dist(me_pos, {"x": opp_goal_x, "y": 0.0})
    height = "B" if d < 18 else "T"
    power = round(min(1.0, 0.65 + d / 80.0), 2)
    return f"{height}{side}", power


# ---------------------------------------------------------------------------
# Passing
# ---------------------------------------------------------------------------

def best_pass(me: dict, mine: list, opps: list, opp_goal_x: float,
              exclude_ids: set[int] | None = None):
    """Return (target_id, pass_type, score) for the safest progressive pass, or None.

    Score rewards forward progress and a free receiver, and penalises
    opponents standing in the passing lane.
    """
    exclude = set(exclude_ids or ()) | {_player_idx(me)}
    me_pos = _xy(me)
    my_goal_dist = abs(me_pos["x"] - opp_goal_x)

    best = None
    for p in mine:
        pid = _player_idx(p)
        if pid in exclude:
            continue
        pos = _xy(p)
        length = dist(me_pos, pos)
        if length < 3:
            continue
        progress = (my_goal_dist - abs(pos["x"] - opp_goal_x)) / 30.0
        blockers = lane_blockers(me_pos, pos, opps)
        marked = _nearest_distance(pos, opps) < 4.0
        score = progress - 0.5 * blockers - (0.4 if marked else 0.0) - length / 100.0

        if blockers and length > 20:
            pass_type = "AERIAL"
        elif progress > 0.25 and not blockers:
            pass_type = "THROUGH"
        else:
            pass_type = "GROUND"

        if best is None or score > best[2]:
            best = (pid, pass_type, round(score, 2))
    return best


# ---------------------------------------------------------------------------
# Off-ball positioning
# ---------------------------------------------------------------------------

def open_space(me_pos: dict, opps: list, opp_goal_x: float, attacking: bool) -> dict:
    """The candidate spot near me that is furthest from any opponent."""
    direction = 1.0 if opp_goal_x > 0 else -1.0
    step = 10.0 if attacking else 0.0
    candidates = [
        {"x": max(-50.0, min(50.0, me_pos["x"] + direction * step + dx)),
         "y": max(-30.0, min(30.0, me_pos["y"] + dy))}
        for dx in (-5.0, 0.0, 5.0)
        for dy in (-10.0, 0.0, 10.0)
    ]
    return max(candidates, key=lambda c: _nearest_distance(c, opps))


# ---------------------------------------------------------------------------
# Match plan
# ---------------------------------------------------------------------------

def match_plan(score: dict, team_id: int) -> tuple[int, str]:
    """(stance, explanation) based on goal difference from our side."""
    ours = score.get("home", 0) if team_id == 0 else score.get("away", 0)
    theirs = score.get("away", 0) if team_id == 0 else score.get("home", 0)
    diff = ours - theirs
    if diff >= 2:
        return 2, f"LEADING by {diff}: protect the lead, keep shape, no risky passes."
    if diff == 1:
        return 0, "LEADING by 1: stay balanced, counter-attack, avoid losing the ball near our goal."
    if diff == 0:
        return 0, "LEVEL: balanced play, take good chances when they appear."
    return 1, f"TRAILING by {-diff}: push forward, press high, shoot on any decent chance."


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _cmd(cmd_type: str, pid: int, tid: int, params: dict, duration: int = 0) -> dict:
    return {"commandType": cmd_type, "playerId": pid, "teamId": tid,
            "parameters": params, "duration": duration}


def _context(game_state: dict, team_id: int, my_player_id: int):
    players = game_state.get("players", [])
    mine, opps = _split(players, team_id)
    me = next((p for p in mine if _player_idx(p) == my_player_id), None)
    my_goal_x, opp_goal_x = get_goal_positions(team_id)
    keeper = _opponent_keeper(opps, opp_goal_x)
    return players, mine, opps, me, my_goal_x, opp_goal_x, keeper


def fast_path(game_state: dict, team_id: int, my_player_id: int, position_label: str):
    """Commands for clear-cut on-ball situations, or None to let the LLM decide."""
    players, mine, opps, me, _, opp_goal_x, keeper = _context(game_state, team_id, my_player_id)
    ball = game_state.get("ball", {})
    if me is None or not holder_is_me(ball, players, team_id, my_player_id):
        return None

    me_pos = _xy(me)

    if position_label == "GK":
        target = best_pass(me, mine, opps, opp_goal_x)
        if target is None:
            return None
        receiver = next(p for p in mine if _player_idx(p) == target[0])
        method = "THROW" if dist(me_pos, _xy(receiver)) < 25 else "KICK"
        return [_cmd("GK_DISTRIBUTE", my_player_id, team_id,
                     {"target_player_id": target[0], "method": method})]

    quality = shot_quality(me_pos, opp_goal_x, keeper, opps)
    close = abs(me_pos["x"] - opp_goal_x) < CLOSE_RANGE and abs(me_pos["y"]) < 12
    if quality >= SHOOT_QUALITY_THRESHOLD or close:
        aim, power = choose_shot(me_pos, keeper, team_id, opp_goal_x)
        return [_cmd("SHOOT", my_player_id, team_id,
                     {"aim_location": aim, "power": power})]

    if position_label == "DEF" and _nearest_distance(me_pos, opps) < UNDER_PRESSURE:
        target = best_pass(me, mine, opps, opp_goal_x, exclude_ids={0})
        if target is not None:
            return [_cmd("PASS", my_player_id, team_id,
                         {"target_player_id": target[0], "type": target[1]})]

    return None


def tactical_insights(game_state: dict, team_id: int, my_player_id: int, position_label: str) -> str:
    """Text block of pre-computed numbers, appended to the state summary."""
    players, mine, opps, me, my_goal_x, opp_goal_x, keeper = _context(game_state, team_id, my_player_id)
    ball = game_state.get("ball", {})
    ball_pos = _xy(ball)
    stance, plan = match_plan(game_state.get("score", {}), team_id)

    lines = ["Tactical analysis (pre-computed, trust these numbers):",
             f"  Match plan: {plan} Suggested SET_STANCE={stance}."]
    if me is None:
        return "\n".join(lines)

    me_pos = _xy(me)
    _, _, we_have_ball = get_possession_info(ball, players, team_id)

    if holder_is_me(ball, players, team_id, my_player_id):
        quality = shot_quality(me_pos, opp_goal_x, keeper, opps)
        aim, power = choose_shot(me_pos, keeper, team_id, opp_goal_x)
        lines.append(f"  YOU HAVE THE BALL. Shot quality={quality:.2f} "
                     f"(shoot if >= {SHOOT_QUALITY_THRESHOLD}); best aim={aim} power={power}.")
        lines.append(f"  Nearest opponent: {_nearest_distance(me_pos, opps):.1f} units away.")
        target = best_pass(me, mine, opps, opp_goal_x)
        if target is not None:
            lines.append(f"  Best pass: player {target[0]} type={target[1]} (score {target[2]:+.2f}).")
    elif we_have_ball:
        spot = open_space(me_pos, opps, opp_goal_x, attacking=position_label in ("MID", "FWD1", "FWD2"))
        lines.append(f"  Teammate has the ball. Open space to receive: "
                     f"({spot['x']:.1f}, {spot['y']:.1f}).")
    else:
        closest = min(mine, key=lambda p: dist(_xy(p), ball_pos))
        i_am_closest = _player_idx(closest) == my_player_id
        if opps:
            danger = min(opps, key=lambda p: abs(_xy(p)["x"] - my_goal_x))
            lines.append(f"  Most dangerous opponent: P{_player_idx(danger)} "
                         f"({abs(_xy(danger)['x'] - my_goal_x):.1f} from our goal).")
        if i_am_closest:
            lines.append("  You are OUR CLOSEST player to the ball: PRESS_BALL or INTERCEPT.")
        else:
            lines.append("  A teammate is closer to the ball: hold shape or MARK, do not chase.")

    return "\n".join(lines)
