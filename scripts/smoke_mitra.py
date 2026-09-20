"""Run the zero-shot regression and classification MITRA smoke pair."""

from smoke_classifier import main as smoke_classifier
from smoke_regressor import main as smoke_regressor


def main() -> None:
    """Run both released-head smokes in their required order."""
    smoke_regressor()
    smoke_classifier()


if __name__ == "__main__":
    main()
