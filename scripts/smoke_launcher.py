"""Statically verify the Windows launcher contract without starting Streamlit."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "cache" / "launcher_ok.json"


def main() -> None:
    """Assert launcher setup, token guard, port cleanup, and app command details."""
    launcher = (ROOT / "run.cmd").read_text(encoding="utf-8")
    required = [
        "py -3 -m venv .venv",
        ".venv\\Scripts\\pip install -r requirements.txt",
        "git clone https://huggingface.co/autogluon/mitra-finetune",
        ".venv\\Scripts\\pip install .\\vendor\\mitra-finetune",
        "Get-NetTCPConnection -LocalPort 8541",
        ".venv\\Scripts\\streamlit run app.py --server.port 8541",
    ]
    missing = [snippet for snippet in required if snippet not in launcher]
    assert not missing, f"Missing launcher commands: {missing}"
    env_block = launcher.index("if not exist .env")
    notepad = launcher.index("start /wait notepad.exe .env", env_block)
    early_exit = launcher.index("exit /b 1", notepad)
    token_check = launcher.index("if not defined HF_TOKEN", early_exit)
    assert env_block < notepad < early_exit < token_check
    assert "hf_" not in launcher

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"launcher": "ok", "port": 8541}, indent=2), encoding="utf-8")
    print(f"launcher smoke passed: {OUTPUT}")


if __name__ == "__main__":
    main()
