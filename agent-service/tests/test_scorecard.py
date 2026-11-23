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
    turns = [turn for scenario in scenarios for turn in scenario.turns]
    expectations = [turn.get("expect", {}) for turn in turns]
    assert any(expect.get("pause") == "relaxation" for expect in expectations)
    assert any(expect.get("pause") == "clarifying_question" for expect in expectations)
    assert any("nothingFits" in expect for expect in expectations)
    assert {"accept", "reject"} <= {turn.get("decision") for turn in turns}


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
        tags=["exclusions"],
        known_issue="#33",
        known_failure=["at least 1 results"],
        turns=[
            {"message": "No horror please", "expect": {"results": {"min": 1}}},
        ],
        candidate_pool=ONLY_HORROR,
    )
    fine = Scenario(id="fine", lang="en", tags=["exclusions"], turns=[{"message": "Something funny", "expect": {}}])
    scorecard = run_scenarios([known, fine])
    assert scorecard.scenario_pass_rate == 1.0  # the gate counts only scenarios without a known issue
    assert (scorecard.turns, scorecard.successful_turns) == (1, 1)  # known-issue turns are counted apart
    assert scorecard.by_tag["exclusions"] == [1, 1]
    assert [(item.scenario, item.issue) for item in scorecard.known_failures] == [("known", "#33")]
    assert "Known issues" in scorecard.render() and "#33" in scorecard.render()


def test_a_known_issue_that_now_passes_is_flagged_for_cleanup():
    fixed = Scenario(
        id="fixed", lang="en", tags=[], known_issue="#33", turns=[{"message": "Something funny", "expect": {}}]
    )
    scorecard = run_scenarios([fixed])
    assert scorecard.fixed_known_issues == ["fixed"]
    assert "now passes" in scorecard.render()


def test_a_known_issue_failing_for_a_new_reason_fails_the_gate():
    regressed = Scenario(
        id="regressed",
        lang="en",
        tags=[],
        known_issue="#33",
        known_failure=["unexpected preferences: languages"],
        candidate_pool=ONLY_HORROR,
        turns=[{"message": "No horror please", "expect": {"results": {"min": 1}}}],
    )
    scorecard = run_scenarios([regressed])
    assert scorecard.scenario_pass_rate == 0.0
    assert "new reason" in scorecard.failures[0].reason


def test_a_scripted_model_that_runs_out_or_has_answers_left_is_a_scenario_mistake():
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
    too_short = Scenario(
        id="short",
        lang="en",
        tags=[],
        model=[nothing],
        turns=[{"message": "a", "expect": {}}, {"message": "Something funny", "expect": {}}],
    )
    too_long = Scenario(
        id="long",
        lang="en",
        tags=[],
        model=[{**nothing, "genres": ["Comedy"]}] * 2,
        turns=[{"message": "Something funny", "expect": {}}],
    )
    failures = run_scenarios([too_short, too_long]).failures
    assert [(f.scenario, f.evaluator) for f in failures if f.evaluator == "runner"] == [
        ("short", "runner"),
        ("long", "runner"),
    ]


INJECTIONS = [
    ({"embedding": "http-503"}, {"semantic": "embedding-failed"}),
    ({"embedding": "timeout"}, {"semantic": "embedding-failed"}),
    ({"embedding": "invalid-json"}, {"semantic": "embedding-failed"}),
    ({"qdrant": "http-500"}, {"semantic": "qdrant-failed"}),
    ({"qdrant": "timeout"}, {"semantic": "qdrant-failed"}),
    ({"qdrant": "invalid-json"}, {"semantic": "qdrant-failed"}),
    ({"model": "timeout"}, {"model": "timeout"}),
    ({"model": "http-429"}, {"model": "http-429"}),
    ({"model": "invalid-json"}, {"model": "invalid-json"}),
]


def test_every_injected_failure_falls_back_gracefully():
    scenarios = [
        Scenario(
            id=f"inject-{index}",
            lang="en",
            tags=["faults"],
            inject=inject,
            turns=[
                {
                    "message": "No horror, something funny",
                    "expect": {"fallback": fallback, "excludes": ["Horror"], "results": {"min": 1}},
                },
            ],
        )
        for index, (inject, fallback) in enumerate(INJECTIONS)
    ]
    scorecard = run_scenarios(scenarios)
    assert scorecard.failures == [], scorecard.render()
    assert scorecard.passes["fallback"] == [len(INJECTIONS), len(INJECTIONS)]
    assert scorecard.graceful_fallbacks == [len(INJECTIONS), len(INJECTIONS)]
    assert "fallback" in scorecard.render()


def test_a_fallback_that_did_not_happen_is_a_failure():
    wrong = Scenario(
        id="no-fault",
        lang="en",
        tags=[],
        turns=[
            {"message": "Something funny", "expect": {"fallback": {"semantic": "embedding-failed"}}},
        ],
    )
    [failure] = run_scenarios([wrong]).failures
    assert failure.evaluator == "fallback"


def test_the_bundled_set_injects_every_failure_kind():
    injected = [scenario.inject for scenario in load_scenarios() if scenario.inject]
    kinds = {(service, failure) for inject in injected for service, failure in inject.items()}
    assert {inject_pair for inject, _ in INJECTIONS for inject_pair in inject.items()} <= kinds


def test_a_fallback_reported_on_a_failed_turn_is_not_graceful():
    scenario = Scenario(
        id="reported-but-wrong",
        lang="en",
        tags=[],
        inject={"qdrant": "http-500"},
        turns=[
            {
                "message": "Something funny",
                "expect": {"fallback": {"semantic": "qdrant-failed"}, "results": {"min": 99}},
            },
        ],
    )
    scorecard = run_scenarios([scenario])
    assert scorecard.passes["fallback"] == [1, 1]  # the right reason was reported...
    assert scorecard.graceful_fallbacks == [0, 1]  # ...but the turn failed, so it was not graceful
    [graceful_row] = [line for line in scorecard.render().splitlines() if "graceful fallback" in line]
    assert graceful_row.split()[2] == "0/1"


SECRET_PERSON = {
    "id": "pat",
    "name": "Pat",
    "color": "#000",
    "preferredGenres": ["Comedy"],
    "ratings": [{"title": "Secret Rating Film", "rating": 5}],
    "watched": [{"title": "Secret Watched Film", "year": 2001}],
}


def test_the_privacy_check_finds_no_secrets_in_storage_or_responses():
    scenario = Scenario(
        id="private",
        lang="en",
        tags=["privacy"],
        persons=[SECRET_PERSON],
        privacy=["Secret Rating Film", "Secret Watched Film", "zebra-umbrella"],
        turns=[
            {"message": "Something funny, zebra-umbrella", "expect": {}},
            {"message": "Under two hours", "expect": {}},
        ],
    )
    scorecard = run_scenarios([scenario])
    assert scorecard.failures == [], scorecard.render()
    assert scorecard.passes["privacy"] == [1, 1]


def test_the_privacy_check_catches_a_secret_that_is_stored():
    scenario = Scenario(
        id="leaky",
        lang="en",
        tags=[],
        privacy=["Horror"],
        turns=[
            {"message": "No horror tonight", "expect": {}},  # the exclusion itself is (rightly) stored
        ],
    )
    [failure] = run_scenarios([scenario]).failures
    assert failure.evaluator == "privacy" and "Horror (in storage)" in failure.reason


def test_a_server_error_is_recorded_not_raised():
    from reel_circle_agent.api import app, clock

    def broken_clock():
        raise RuntimeError("boom")

    app.dependency_overrides[clock] = broken_clock
    try:
        scenario = Scenario(id="crash", lang="en", tags=[], turns=[{"message": "Something funny", "expect": {}}])
        [failure] = run_scenarios([scenario]).failures
    finally:
        app.dependency_overrides.pop(clock, None)
    assert failure.evaluator == "runner" and "HTTP 500" in failure.reason


def test_the_bundled_set_includes_attacks_privacy_and_adversarial_input():
    scenarios = load_scenarios()
    assert sum(any("attack" in turn.get("expect", {}) for turn in s.turns) for s in scenarios) >= 3
    assert any(s.privacy for s in scenarios)
    assert sum("adversarial" in s.tags for s in scenarios) >= 3
