"""Summarise final fitness across independent runs."""
import argparse
import csv
from pathlib import Path
from constants import RESULTS_DIR
from evaluation.common import RunLog, discover_logs, group_by_variant
from evaluation.common import mean, sample_std, variant_sort_key

def build_parser() -> argparse.ArgumentParser:
    parser= argparse.ArgumentParser(
        description="Summarise assignment results",
    )
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "summary.csv",
    )
    return parser

def summarise_variant(variant: str, runs: list[RunLog]) -> dict[str, str]:
    finals= [run.final_best for run in runs]
    return {
        "variant": variant,
        "runs": str(len(runs)),
        "mean_final_best": f"{mean(finals):.4f}",
        "std_final_best": f"{sample_std(finals):.4f}",
    }
def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns=["variant", "runs", "mean_final_best", "std_final_best"]
    with path.open("w", newline="") as handle:
        writer =csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

def print_summary(grouped: dict[str, list[RunLog]]) -> None:
    for variant in sorted(grouped, key=variant_sort_key):
        row = summarise_variant(variant, grouped[variant])
        print(
            f"{row['variant']:>8}: n={row['runs']} "
            f"mean={row['mean_final_best']} std={row['std_final_best']}",
        )




def run(args: argparse.Namespace) -> Path:
    logs= discover_logs(args.results)
    if not logs:
        raise SystemExit(f"no logs found under {args.results}")
    grouped= group_by_variant(logs)
    rows =[
        summarise_variant(variant, grouped[variant])
        for variant in sorted(grouped, key=variant_sort_key)
    ]
    write_summary(args.out, rows)
    print_summary(grouped)
    print(f"wrote {args.out}")
    return args.out
def main() -> None:
    run(build_parser().parse_args())





if __name__ == "__main__":
    main()
