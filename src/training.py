"""Small compatibility exports used by the tutorial's training-oriented pages."""

from __future__ import annotations

import pandas as pd

from src.load import split_from_metadata
from src.mitra_run import device_info


def split_data(frame: pd.DataFrame, metadata: dict):
    """Return the deterministic known/hidden split for a table.

    Args:
        frame: Full table associated with ``metadata``.
        metadata: Split metadata produced by :func:`src.load.build_metadata`.

    Returns:
        A ``(known, hidden)`` pandas DataFrame pair.
    """
    return split_from_metadata(frame, metadata)


__all__ = ["device_info", "split_data"]
