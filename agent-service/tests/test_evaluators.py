"""The evaluators, on hand-written responses whose correct verdict is known."""

from reel_circle_agent.evaluation.evaluators import evaluate_final_answer, evaluate_path, evaluate_turn

FULL_PATH = ["extract_preferences", "filter_candidates", "semantic_retrieval", "rank_candidates", "validate_results"]


def response(**overrides) -> dict:
    base = {
        "preferences": {"genres": [], "excludedGenres": [], "platforms": [], "moods": []},
        "recommendations": [],
        "meta": {"trace": [{"tool": tool, "status": "completed"} for tool in FULL_PATH]},
    }
    return {**base, **overrides}


def movie(movie_id: str, **fields) -> dict:
    return {
        "movie": {
            "id": movie_id,
            "runtime": 90,
            "genres": ["Comedy"],
            "platforms": ["Filmin"],
            "year": 2010,
            "language": "English",
            **fields,
        }
    }


def test_the_path_passes_when_the_trace_matches_and_no_step_limit_was_hit():
    assert evaluate_path({"path": FULL_PATH}, response()).passed


def test_the_path_fails_on_a_different_route_and_says_where():
    score = evaluate_path({"path": FULL_PATH}, response(meta={"trace": [{"tool": "extract_preferences"}]}))
    assert not score.passed
    assert "filter_candidates" in score.reason


def test_the_path_fails_when_the_step_limit_was_hit():
    trace = [{"tool": "extract_preferences"}, {"tool": "step_limit"}]
    assert not evaluate_path({}, response(meta={"trace": trace})).passed


def test_the_path_checks_the_expected_pause_and_its_absence():
    paused = response(pause={"type": "relaxation", "constraint": "yearRange", "from": [1990, 1999], "to": None})
    assert evaluate_path({"pause": "relaxation"}, paused).passed
    assert not evaluate_path({"pause": "clarifying_question"}, paused).passed
    assert not evaluate_path({}, paused).passed  # an unexpected pause is a wrong route
    assert not evaluate_path({"pause": "relaxation"}, response()).passed


def test_a_recommendation_breaking_a_hard_constraint_fails_the_final_answer():
    preferences = {"genres": [], "excludedGenres": ["Horror"], "platforms": [], "moods": [], "maxRuntime": 100}
    too_long = response(preferences=preferences, recommendations=[movie("long", runtime=130)])
    scary = response(preferences=preferences, recommendations=[movie("scary", genres=["Horror", "Comedy"])])
    fine = response(preferences=preferences, recommendations=[movie("fine")])
    assert "runtime" in evaluate_final_answer({}, too_long).reason
    assert not evaluate_final_answer({}, scary).passed
    assert evaluate_final_answer({}, fine).passed


def test_language_codes_and_names_both_count_as_the_same_language():
    preferences = {"genres": [], "excludedGenres": [], "platforms": [], "moods": [], "languages": ["ko"]}
    korean = response(preferences=preferences, recommendations=[movie("k", language="Korean")])
    english = response(preferences=preferences, recommendations=[movie("e", language="English")])
    assert evaluate_final_answer({}, korean).passed
    assert not evaluate_final_answer({}, english).passed


def test_kept_exclusions_and_result_counts_are_checked():
    preferences = {"genres": [], "excludedGenres": ["Horror"], "platforms": [], "moods": []}
    two = response(preferences=preferences, recommendations=[movie("a"), movie("b")])
    assert evaluate_final_answer({"excludes": ["Horror"], "results": {"min": 2}}, two).passed
    assert not evaluate_final_answer({"excludes": ["Drama"]}, two).passed
    assert not evaluate_final_answer({"results": {"min": 3}}, two).passed
    assert not evaluate_final_answer({"results": {"count": 0}}, two).passed
    assert evaluate_final_answer({"results": {"top": ["a"]}}, two).passed
    assert not evaluate_final_answer({"results": {"top": ["b"]}}, two).passed


def test_nothing_fits_must_be_announced_when_expected():
    honest = response(nothingFits={"tightestConstraint": "excludedGenres"})
    assert evaluate_final_answer({"nothingFits": "excludedGenres"}, honest).passed
    assert not evaluate_final_answer({"nothingFits": "excludedGenres"}, response()).passed
    assert not evaluate_final_answer({"nothingFits": "maxRuntime"}, honest).passed


def test_a_successful_turn_needs_every_evaluator_to_pass():
    preferences = {"genres": [], "excludedGenres": ["Horror"], "platforms": [], "moods": []}
    good = response(preferences=preferences, recommendations=[movie("a")])
    bad_route = response(preferences=preferences, recommendations=[movie("a")], meta={"trace": []})
    assert all(score.passed for score in evaluate_turn({"path": FULL_PATH, "excludes": ["Horror"]}, good))
    assert not all(score.passed for score in evaluate_turn({"path": FULL_PATH}, bad_route))


def test_expected_preferences_must_match_exactly():
    preferences = {"genres": ["Drama"], "excludedGenres": [], "platforms": [], "moods": [], "languages": ["ko"]}
    turn = response(preferences=preferences)
    assert evaluate_final_answer({"preferences": {"genres": ["Drama"], "languages": ["ko"]}}, turn).passed
    assert not evaluate_final_answer({"preferences": {"genres": ["Comedy"]}}, turn).passed
    assert not evaluate_final_answer({"preferences": {"platforms": ["Netflix"]}}, turn).passed
