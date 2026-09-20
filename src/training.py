from __future__ import annotations

import pandas as pd

from src.load import split_from_metadata
from src.mitra_run import device_info


def split_data(frame: pd.DataFrame, metadata: dict):
    return split_from_metadata(frame, metadata)


__all__ = ["device_info", "split_data"]
