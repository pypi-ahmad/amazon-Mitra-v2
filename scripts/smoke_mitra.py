from smoke_classifier import main as smoke_classifier
from smoke_regressor import main as smoke_regressor


def main() -> None:
    smoke_regressor()
    smoke_classifier()


if __name__ == "__main__":
    main()
