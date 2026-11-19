"""Leakage-safe splitting: deduplicate first, then split, optionally by group."""

from __future__ import annotations

import hashlib

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def content_hash(text: str) -> str:
    """Stable hash of whitespace- and case-normalised text, for exact-duplicate detection."""
    normalised = " ".join(text.lower().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def dedupe(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Drop exact duplicates (after normalisation) so none can straddle train and test."""
    if text_col not in df.columns:
        raise KeyError(f"column {text_col!r} not in frame")
    hashes = df[text_col].astype(str).map(content_hash)
    return df.loc[~hashes.duplicated()].reset_index(drop=True)


def safe_split(
    df: pd.DataFrame,
    label_col: str,
    test_size: float = 0.2,
    seed: int = 42,
    group_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split, or group split when rows share a source (patient, video, sender).

    Never oversample before calling this; resample the training split only.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    if group_col is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=seed
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
