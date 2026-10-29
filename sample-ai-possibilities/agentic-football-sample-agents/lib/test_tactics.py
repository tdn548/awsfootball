"""Tests for tactics.py and the opt-in options of create_invoke_handler.

Run directly:  python test_tactics.py
"""

import asyncio
import copy
import json

from test_helpers import GAME_STATE, mock_agentcore, _FakeApp

mock_agentcore()

from agent_base import create_invoke_handler
from fallback import FWD1_CONFIG, build_fallback
from tactics import choose_shot, fast_path, tactical_insights

HOME, AWAY = 0, 1


def _with_holder(team_code: str, idx: int, pos: dict) -> dict:
    gs = copy.deepcopy(GAME_STATE)
    gs["ball"]["possessionAgentId"] = f"agentId_{idx}"
    gs["ball"]["position"] = dict(pos)
    for p in gs["players"]:
        if p["teamCode"] == team_code and p["agentId"] == f"agentId_{idx}":
            p["position"] = dict(pos)
    # Keep only the holder's team claiming that agentId, so possession is unambiguous.
    if team_code == "away":
        gs["players"] = [p for p in gs["players"]
                         if not (p["teamCode"] == "home" and p["agentId"] == f"agentId_{idx}")]
    return gs


def test_fast_path_shoots_close_to_goal():
    gs = _with_holder("home", 3, {"x": 45, "y": 2})
    cmds = fast_path(gs, HOME, 3, "FWD1")
    assert cmds and cmds[0]["commandType"] == "SHOOT", cmds
    assert cmds[0]["playerId"] == 3 and cmds[0]["teamId"] == HOME
    print(f"=== FAST PATH: close-range shot {cmds[0]['parameters']} ===")


def test_fast_path_defers_to_llm_when_far():
    """The fixture forward holds the ball 40 units out — a judgement call for the LLM."""
    assert fast_path(GAME_STATE, HOME, 3, "FWD1") is None
    print("=== FAST PATH: far from goal, LLM decides ===")


def test_fast_path_ignores_players_without_ball():
    assert fast_path(GAME_STATE, HOME, 2, "MID") is None
    print("=== FAST PATH: no ball, LLM decides ===")


def test_keeper_distributes():
    gs = _with_holder("home", 0, {"x": -50, "y": 0})
    cmds = fast_path(gs, HOME, 0, "GK")
    assert cmds and cmds[0]["commandType"] == "GK_DISTRIBUTE", cmds
    assert cmds[0]["parameters"]["target_player_id"] != 0
    print(f"=== FAST PATH: keeper distributes {cmds[0]['parameters']} ===")


def test_aim_avoids_keeper_side_for_both_teams():
    """Keeper shading +y: HOME shoots to -y (its right), AWAY to -y (its left)."""
    keeper = {"position": {"x": 50, "y": 3}}
    aim_home, _ = choose_shot({"x": 42, "y": 0}, keeper, HOME, 55.0)
    keeper_away = {"position": {"x": -50, "y": 3}}
    aim_away, _ = choose_shot({"x": -42, "y": 0}, keeper_away, AWAY, -55.0)
    assert aim_home == "BR", aim_home
    assert aim_away == "BL", aim_away
    print(f"=== AIM: home {aim_home}, away {aim_away} ===")


def test_insights_mention_plan_and_shot():
    out = tactical_insights(GAME_STATE, HOME, 3, "FWD1")
    assert "LEADING by 1" in out
    assert "Shot quality=" in out and "Best pass: player" in out
    out_away = tactical_insights(GAME_STATE, AWAY, 2, "MID")
    assert "TRAILING by 1" in out_away
    print("=== INSIGHTS: plan follows the score from each side ===")


class _FakeAgent:
    def __init__(self, reply: str):
        self.reply = reply
        self.messages = []
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        self.messages.extend([{"role": "user"}, {"role": "assistant"}])
        return self.reply


def _invoke(handler, game_state, team_id=HOME):
    payload = {"prompt": json.dumps({"teamId": team_id, "gameState": game_state})}

    async def collect():
        return [json.loads(chunk) async for chunk in handler(payload, None)]

    return asyncio.run(collect())[0]


def _handler(agent, **options):
    return create_invoke_handler(_FakeApp(), agent, 3, "FWD1", build_fallback(FWD1_CONFIG),
                                 fallback_cfg=FWD1_CONFIG, **options)


def test_handler_fast_path_skips_llm():
    agent = _FakeAgent("[]")
    handler = _handler(agent, fast_path_fn=fast_path)
    cmds = _invoke(handler, _with_holder("home", 3, {"x": 45, "y": 2}))
    assert cmds[0]["commandType"] == "SHOOT"
    assert agent.prompts == [], "LLM should not be called on a fast-path tick"
    print("=== HANDLER: fast path answered without calling the LLM ===")


def test_handler_stateless_clears_history_and_adds_context():
    reply = '[{"commandType":"PASS","parameters":{"target_player_id":4,"type":"GROUND"}}]'
    agent = _FakeAgent(reply)
    handler = _handler(agent, context_fn=tactical_insights, stateless=True)
    for _ in range(3):
        cmds = _invoke(handler, GAME_STATE)
    assert cmds[0]["commandType"] == "PASS"
    assert len(agent.messages) == 2, f"history grew to {len(agent.messages)}"
    assert "Tactical analysis" in agent.prompts[-1]
    print("=== HANDLER: history reset each tick, insights in prompt ===")


def test_handler_defaults_unchanged():
    """Teams that do not opt in keep their history (the memory team relies on it)."""
    agent = _FakeAgent('[{"commandType":"SET_STANCE","parameters":{"stance":1}}]')
    handler = _handler(agent)
    for _ in range(3):
        _invoke(handler, GAME_STATE)
    assert len(agent.messages) == 6
    assert "Tactical analysis" not in agent.prompts[-1]
    print("=== HANDLER: default behaviour unchanged ===")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"{len(fns)} tests passed")
