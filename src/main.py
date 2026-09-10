import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation.evaluation import run_evaluation
from .agent import SupportAgent
from .prepare import prepare


def ensure_prepared(data_path, processed_dir, brand, max_cases, golden_size):
    processed_dir = Path(processed_dir)
    if not (processed_dir / "cases.csv").exists() or not (processed_dir / "golden_set.csv").exists():
        prepare(data_path, processed_dir, brand, max_cases=max_cases, golden_size=golden_size)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "run"]:
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--data", default="data/raw/twcs.csv")
        subparser.add_argument("--processed", default="data/processed")
        subparser.add_argument("--brand", default="AmazonHelp")
        subparser.add_argument("--max-cases", type=int, default=24_000)
        subparser.add_argument("--golden-size", type=int, default=200)
    predict = subparsers.add_parser("predict")
    predict.add_argument("message")
    predict.add_argument("--processed", default="data/processed")
    predict.add_argument("--openai", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.data, args.processed, args.brand, max_cases=args.max_cases, golden_size=args.golden_size), indent=2))
        return
    if args.command == "run":
        ensure_prepared(args.data, args.processed, args.brand, args.max_cases, args.golden_size)
        print(json.dumps(run_evaluation(args.processed), indent=2))
        return
    cases = pd.read_csv(Path(args.processed) / "cases.csv")
    golden = pd.read_csv(Path(args.processed) / "golden_set.csv")
    train = cases[~cases["case_id"].isin(golden["case_id"])].reset_index(drop=True)
    output = SupportAgent(use_openai=args.openai).fit(train).predict(args.message)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
