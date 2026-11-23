"""The scorecard runner: scenarios replayed through the HTTP endpoint, scored per turn."""

import json

from reel_circle_agent.evaluation.cli import main
from reel_circle_agent.evaluation.runner import Scenario, load_scenarios, run_scenarios

ONLY_HORROR = [
    {
        "id": "hereditary",
        "title": "Hereditary",
        "year": 2018,
        "runtime": 127,
        "genres": ["Horror"],
        "platforms": ["Filmin"],
        "rating": 4.0,
        "language": "English",
    }
]


def test_every_bundled_scenario_passes():
    scenarios = load_scenarios()
    scorecard = run_scenarios(scenarios)
    assert len(scenarios) >= 6
    assert {scenario.lang for scenario in scenarios} == {"en", "es"}
    assert any(len(scenario.turns) > 1 for scenario in scenarios)
    assert scorecard.failures == [], scorecard.render()
    assert scorecard.scenario_pass_rate == 1.0


def test_a_wrong_expectation_is_reported_with_its_reason():
    wrong = Scenario(
        id="wrong",
        lang="en",
        tags=[],
        turns=[
            {"message": "No horror please", "expect": {"results": {"min": 1}}},
        ],
        candidate_pool=ONLY_HORROR,
    )
    scorecard = run_scenarios([wrong])
    assert scorecard.scenario_pass_rate == 0.0
    [failure] = scorecard.failures
    assert (failure.scenario, failure.turn, failure.evaluator) == ("wrong", 1, "final_answer")
    assert "at least 1" in failure.reason
    assert "wrong" in scorecard.render()


def test_a_decision_turn_answers_the_pause_of_the_turn_before():
    korean_thriller = [
        {
            "id": "memories",
            "title": "Memories of Murder",
            "year": 2003,
            "runtime": 131,
            "genres": ["Thriller"],
            "platforms": ["Netflix"],
            "rating": 4.3,
            "language": "Korean",
        }
    ]
    scenario = Scenario(
        id="relax",
        lang="en",
        tags=["relaxation"],
        candidate_pool=korean_thriller,
        turns=[
            {"message": "A Korean thriller from the 1990s", "expect": {"pause": "relaxation"}},
            {"decision": "accept", "message": "Yes", "expect": {"results": {"top": ["memories"]}}},
        ],
    )
    assert run_scenarios([scenario]).failures == []


def test_the_command_fails_below_the_threshold(tmp_path, capsys):
    (tmp_path / "wrong.json").write_text(
        json.dumps(
            {
                "id": "wrong",
                "lang": "en",
                "tags": [],
                "candidatePool": ONLY_HORROR,
                "turns": [{"message": "No horror please", "expect": {"results": {"min": 1}}}],
            }
        ),
        encoding="utf-8",
    )
    assert main(["--scenarios", str(tmp_path)]) == 1
    assert "0/1" in capsys.readouterr().out
    assert main([]) == 0
