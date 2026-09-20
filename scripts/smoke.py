from importlib.util import find_spec
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FALLBACK = ROOT / "data" / "fallback"

SAMPLES = {
    "houses.csv": "median_house_value",
    "machines.csv": "machine_failed",
    "adult_income.csv": "income",
    "credit_g.csv": "class",
    "wine.csv": "quality",
}


def main() -> None:
    modules = [
        "streamlit",
        "pandas",
        "numpy",
        "pyarrow",
        "plotly",
        "matplotlib",
        "seaborn",
        "sklearn",
        "datasets",
        "huggingface_hub",
        "autogluon.tabular",
        "mitra_finetune",
    ]
    missing = [module for module in modules if find_spec(module) is None]
    if missing:
        raise RuntimeError(f"Missing imports: {', '.join(missing)}")
    for filename, target in SAMPLES.items():
        frame = pd.read_csv(FALLBACK / filename)
        assert len(frame) >= 20, filename
        assert target in frame, f"{target} missing from {filename}"
    print("smoke ok: dependencies and five local datasets")


if __name__ == "__main__":
    main()
