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


def test_a_decision_without_a_pause_to_answer_is_a_failure():
    scenario = Scenario(
        id="no-pause",
        lang="en",
        tags=[],
        turns=[
            {"message": "Something funny", "expect": {}},
            {"decision": "accept", "expect": {}},
        ],
    )
    [failure] = run_scenarios([scenario]).failures
    assert (failure.turn, failure.evaluator) == (2, "runner")
    assert "no pause" in failure.reason


def test_exclusions_from_earlier_turns_are_checked_without_repeating_them():
    scenario = Scenario(
        id="carry",
        lang="en",
        tags=[],
        turns=[
            {"message": "No horror tonight", "expect": {}},
            {"message": "Start over. Something funny", "expect": {}},
        ],
    )
    [failure] = run_scenarios([scenario]).failures
    assert (failure.turn, failure.evaluator) == (2, "final_answer")
    assert "Horror" in failure.reason


REQUIRED_TAGS = {
    "runtime",
    "year-range",
    "languages",
    "genres",
    "platforms",
    "exclusions",
    "moods",
    "relaxation",
    "clarifying-question",
    "nothing-fits",
    "multi-turn",
    "start-over",
    "vibe",
}


def test_the_bundled_set_covers_the_agent_broadly():
    scenarios = load_scenarios()
    assert len(scenarios) >= 40
    assert sum(scenario.lang == "es" for scenario in scenarios) >= len(scenarios) / 3
    assert sum(len(scenario.turns) > 1 for scenario in scenarios) >= len(scenarios) / 2
    assert REQUIRED_TAGS <= {tag for scenario in scenarios for tag in scenario.tags}
    assert len({scenario.id for scenario in scenarios}) == len(scenarios)


def test_the_scorecard_breaks_results_down_by_tag():
    scorecard = run_scenarios(load_scenarios())
    rendered = scorecard.render()
    assert "By tag:" in rendered
    assert scorecard.by_tag["relaxation"][1] >= 3  # (passed, total) per tag


def test_a_scripted_model_answers_each_turn_in_order():
    nothing = {
        "genres": [],
        "excludedGenres": [],
        "platforms": [],
        "moods": [],
        "maxRuntime": None,
        "yearRange": None,
        "languages": [],
        "peopleCount": None,
        "viewingContext": None,
        "clarificationQuestion": None,
    }
    scenario = Scenario(
        id="ask",
        lang="en",
        tags=[],
        model=[
            {**nothing, "clarificationQuestion": "Light or serious?"},
            {**nothing, "genres": ["Comedy"]},
        ],
        turns=[
            {"message": "hmm", "expect": {"pause": "clarifying_question"}},
            {"message": "Something light", "expect": {"preferences": {"genres": ["Comedy"]}, "results": {"min": 1}}},
        ],
    )
    scorecard = run_scenarios([scenario])
    assert scorecard.failures == [], scorecard.render()


def test_a_known_issue_is_reported_but_does_not_fail_the_gate():
    known = Scenario(
        id="known",
        lang="en",
        tags=[],
        known_issue="#33",
        turns=[
            {"message": "No horror please", "expect": {"results": {"min": 1}}},
        ],
        candidate_pool=ONLY_HORROR,
    )
    fine = Scenario(id="fine", lang="en", tags=[], turns=[{"message": "Something funny", "expect": {}}])
    scorecard = run_scenarios([known, fine])
    assert scorecard.scenario_pass_rate == 1.0  # the gate counts only scenarios without a known issue
    assert [(item.scenario, item.issue) for item in scorecard.known_failures] == [("known", "#33")]
    assert "Known issues" in scorecard.render() and "#33" in scorecard.render()


def test_a_known_issue_that_now_passes_is_flagged_for_cleanup():
    fixed = Scenario(
        id="fixed", lang="en", tags=[], known_issue="#33", turns=[{"message": "Something funny", "expect": {}}]
    )
    scorecard = run_scenarios([fixed])
    assert scorecard.fixed_known_issues == ["fixed"]
    assert "now passes" in scorecard.render()
