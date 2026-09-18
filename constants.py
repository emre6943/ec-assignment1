from pathlib import Path

HERE= Path(__file__).parent
TARGET_DIR= HERE / "target_bodies"
RESULTS_DIR =HERE / "results"

DEFAULT_SEED =1
DEFAULT_SEEDS= (1, 2, 3, 4, 5)
DEFAULT_VARIANTS= (2, 4, 8)
DEFAULT_POP =50
DEFAULT_GENS=100
DEFAULT_MAX_MODULES= 20

MUTATION_PROBABILITIES: tuple[tuple[str, float], ...]=(
    ("point", 0.4),
    ("subtree", 0.4),
    ("shrink", 0.1),
    ("hoist", 0.1),
)
MAX_SHRINK_ATTEMPTS=20
DIVERSITY_SAMPLE_SIZE=20
DIVERSITY_SEED_OFFSET=1_000_003
LOG_COLUMNS=(
    "gen", "evals", "best", "mean", "worst", "diversity",
    "fallbacks", "cap_fallbacks", "forced_mutations",
)
def ea_result_dir(results_dir: Path, parents: int, seed: int) -> Path:
    return results_dir / f"k{parents}" / f"seed{seed}"




def baseline_result_dir(results_dir: Path, seed: int) -> Path:
    return results_dir / "random" / f"seed{seed}"
