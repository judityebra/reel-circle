"""`uv run reel-eval`: replays the scenarios, prints the agent scorecard, fails below a threshold."""

import argparse
from pathlib import Path

from .runner import SCENARIOS_DIR, load_scenarios, run_scenarios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reel-eval", description="Score the recommendation agent on scenarios.")
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS_DIR, help="folder of scenario JSON files")
    parser.add_argument(
        "--min-scenario-pass", type=float, default=1.0, help="exit with an error below this share of passed scenarios"
    )
    args = parser.parse_args(argv)
    scorecard = run_scenarios(load_scenarios(args.scenarios))
    print(scorecard.render())
    return 0 if scorecard.scenario_pass_rate >= args.min_scenario_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
