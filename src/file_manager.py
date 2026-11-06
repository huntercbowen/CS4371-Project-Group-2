from pathlib import Path
from contextlib import contextmanager


# Directory configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT_DIR / "data" / "train"
TEMP_DIR = ROOT_DIR / "data" / ".temp"


def restore_file(file):
    """Move a withheld CSV file back into the training directory."""
    destination = TRAIN_DIR / file.name

    # Prevent accidentally overwriting an existing file
    if destination.exists():
        raise FileExistsError(
            f"Cannot restore {file.name}: destination already exists."
        )

    file.rename(destination)


def recover_temp():
    """Recover training files left behind by an interrupted experiment."""
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    for file in sorted(TEMP_DIR.glob("*.csv")):
        restore_file(file)
        print(f"Recovered: {file.name}")


@contextmanager
def temporarily_withhold(files):
    """Temporarily remove selected attack files from training."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    files = list(files)

    if not files:
        raise ValueError("No training files selected.")

    # Refuse to start if files from a previous run remain
    if any(TEMP_DIR.glob("*.csv")):
        raise RuntimeError(
            "Temporary directory contains CSV files. "
            "Recover them before starting another experiment."
        )

    # Validate files before moving them
    for file in files:
        if file.parent.resolve() != TRAIN_DIR.resolve():
            raise ValueError(f"File is not in data/train: {file}")

        if not file.is_file():
            raise FileNotFoundError(f"Missing: {file.name}")

        if file.suffix.lower() != ".csv":
            raise ValueError(f"Not a CSV file: {file.name}")

    if len({file.name for file in files}) != len(files):
        raise ValueError("Duplicate training files selected.")

    moved_files = []

    try:
        # Move selected attack files into temporary storage
        for file in files:
            destination = TEMP_DIR / file.name
            file.rename(destination)
            moved_files.append(destination)

        print(f"\nWithheld {len(moved_files)} training files.")

        # Run experiment while files are hidden
        yield

    finally:
        # Restore files, even if an error occurs
        for file in reversed(moved_files):
            restore_file(file)

        print(f"Restored {len(moved_files)} training files.")
