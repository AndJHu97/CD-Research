import csv
from pathlib import Path
import sys


def extract_unique_warheads(input_csv: Path, output_csv: Path) -> Path:
    unique_warheads = set()

    with input_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"No header found in {input_csv}")

        warhead_col = None
        for col in reader.fieldnames:
            if col.strip().lower() == "warhead":
                warhead_col = col
                break

        if warhead_col is None:
            raise ValueError("No 'Warhead' column found in CSV")

        for row in reader:
            value = row.get(warhead_col)
            if value:
                value = value.strip()
                if value:
                    unique_warheads.add(value)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Warhead"])
        for w in sorted(unique_warheads):
            writer.writerow([w])

    return output_csv


def main():
    # Option 1: hardcode file here
    input_file = Path("Covalent_Complex_Records.csv")
    output_file = Path("unique_warheads.csv")

    # Option 2 (recommended): allow command-line input
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])

    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])

    result = extract_unique_warheads(input_file, output_file)
    print(f"Saved unique warheads to: {result}")


if __name__ == "__main__":
    main()