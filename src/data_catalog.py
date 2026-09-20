"""Definitions for the five bundled tutorial datasets and their fallbacks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Describe one downloadable or local dataset source.

    Attributes:
        kind: Loader kind accepted by :func:`src.load.load_sample`.
        source_id: Dataset identifier, URL, or local fallback filename.
        splits: Ordered dataset splits to combine when the source is on Hugging Face.
    """

    kind: str
    source_id: str
    splits: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SampleSpec:
    """Describe a normalized tutorial table and its deterministic split policy.

    Attributes:
        id: Stable cache-directory identifier.
        task: AutoGluon problem type used for the sample.
        target: Normalized target-column name shown in the app.
        source_target: Expected target-column name before normalization.
        story: Short instructional description displayed in the UI.
        source: Primary source attempted before fallbacks.
        fallbacks: Ordered sources attempted after a primary-source failure.
        test_size: Fraction of labeled rows reserved as hidden test data.
        random_state: Seed used for deterministic sampling and splitting.
        row_cap: Optional maximum number of rows retained after loading.
        column_aliases: Source-to-display column-name replacements.
        drop_columns: Columns removed before the table is exposed to the app.
        featured: Whether the Home page promotes this sample.
    """

    id: str
    task: str
    target: str
    source_target: str
    story: str
    source: SourceSpec
    fallbacks: tuple[SourceSpec, ...]
    test_size: float = 0.1
    random_state: int = 42
    row_cap: int | None = None
    column_aliases: tuple[tuple[str, str], ...] = ()
    drop_columns: tuple[str, ...] = ()
    featured: bool = False


SAMPLE_CATALOG: dict[str, SampleSpec] = {
    "Houses": SampleSpec(
        id="houses",
        task="regression",
        target="MedHouseVal",
        source_target="median_house_value",
        story="Estimate median home value from neighborhood and housing signals.",
        source=SourceSpec(
            "huggingface",
            "gvlassis/california_housing",
            ("train", "validation", "test"),
        ),
        fallbacks=(
            SourceSpec("sklearn", "fetch_california_housing"),
            SourceSpec("local", "houses.csv"),
        ),
        column_aliases=(("median_house_value", "MedHouseVal"),),
        featured=True,
    ),
    "Machines": SampleSpec(
        id="machines",
        task="binary",
        target="Machine failure",
        source_target="Target",
        story="Predict machine failure from operating conditions before a breakdown.",
        source=SourceSpec(
            "huggingface",
            "EddyGiusepe/Modified_dataset_for_predictive_maintenance",
            ("train", "test"),
        ),
        fallbacks=(
            SourceSpec(
                "url",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv",
            ),
            SourceSpec("local", "machines.csv"),
        ),
        column_aliases=(("Target", "Machine failure"), ("machine_failed", "Machine failure")),
        drop_columns=("UDI", "Product ID", "Failure Type"),
        featured=True,
    ),
    "Adult income": SampleSpec(
        id="adult_income",
        task="binary",
        target="income",
        source_target="income",
        story="Predict whether annual income exceeds $50K from census attributes.",
        source=SourceSpec("huggingface", "scikit-learn/adult-census-income", ("train",)),
        fallbacks=(
            SourceSpec("openml", "1590"),
            SourceSpec("local", "adult_income.csv"),
        ),
        row_cap=20_000,
    ),
    "Credit-g": SampleSpec(
        id="credit_g",
        task="binary",
        target="Risk",
        source_target="Risk",
        story="Classify credit applications as good or bad risk.",
        source=SourceSpec("huggingface", "AiresPucrs/german-credit-data", ("train",)),
        fallbacks=(
            SourceSpec("openml", "31"),
            SourceSpec("local", "credit_g.csv"),
        ),
        column_aliases=(("class", "Risk"),),
    ),
    "Wine quality": SampleSpec(
        id="wine_quality",
        task="regression",
        target="quality",
        source_target="quality",
        story="Estimate red wine quality from physicochemical measurements.",
        source=SourceSpec("huggingface", "codesignal/wine-quality", ("red",)),
        fallbacks=(
            SourceSpec(
                "url_semicolon",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            ),
            SourceSpec("local", "wine.csv"),
        ),
    ),
}
