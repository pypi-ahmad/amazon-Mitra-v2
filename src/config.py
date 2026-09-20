from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
PAPER_URL = "https://arxiv.org/abs/2609.04540"
CLASSIFIER_ID = "autogluon/mitra-classifier-2"
REGRESSOR_ID = "autogluon/mitra-regressor-2"

SOURCES = {
    "Fine-tuning recipe": "https://huggingface.co/autogluon/mitra-finetune",
    "Classifier": "https://huggingface.co/autogluon/mitra-classifier-2",
    "Regressor": "https://huggingface.co/autogluon/mitra-regressor-2",
    "Technical report": PAPER_URL,
    "AutoGluon": "https://github.com/autogluon/autogluon",
}
