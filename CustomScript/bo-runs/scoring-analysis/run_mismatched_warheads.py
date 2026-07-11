import pandas as pd
import argparse


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def norm(x: str) -> str:
    return str(x).strip().upper()


def parse_warheads(raw: str) -> set[str]:
    """
    Labels column: "Frankenstein_Warhead"
    Example: "NITRILE, ALDEHYDE"
    """
    if pd.isna(raw):
        return set()
    return {w.strip().upper() for w in str(raw).split(",") if w.strip()}


def build_training_lookup(train: pd.DataFrame) -> dict:
    """
    Map: Name -> set of Warheads observed in training
    """
    lookup = {}

    for _, row in train.iterrows():
        name = norm(row["Name"])
        warhead = norm(row["Warhead"])

        if name not in lookup:
            lookup[name] = set()
        lookup[name].add(warhead)

    return lookup


def find_mismatches(labels: pd.DataFrame, train_lookup: dict) -> pd.DataFrame:
    mismatched_rows = []

    for _, row in labels.iterrows():
        name = norm(row["Name"])
        allowed_warheads = parse_warheads(row.get("Frankenstein_Warhead", ""))

        # If no training entries exist for this name → mismatch
        if name not in train_lookup:
            mismatched_rows.append(row)
            continue

        training_warheads = train_lookup[name]

        # Check if ANY allowed warhead exists in training for this name
        if len(allowed_warheads) == 0:
            mismatched_rows.append(row)
            continue

        if len(training_warheads.intersection(allowed_warheads)) == 0:
            mismatched_rows.append(row)

    return pd.DataFrame(mismatched_rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--labels", required=True, help="labels CSV (has Frankenstein_Warhead)")
    parser.add_argument("--training", required=True, help="training CSV (has Warhead)")
    parser.add_argument("--output", default="mismatched_warheads.csv")

    args = parser.parse_args()

    labels = load_csv(args.labels)
    train = load_csv(args.training)

    required_train_cols = {"Name", "Warhead"}
    required_label_cols = {"Name", "Frankenstein_Warhead"}

    missing_train = required_train_cols - set(train.columns)
    missing_label = required_label_cols - set(labels.columns)

    if missing_train:
        raise ValueError(f"Training missing columns: {missing_train}")
    if missing_label:
        raise ValueError(f"Labels missing columns: {missing_label}")

    print(f"[INFO] Training rows: {len(train):,}")
    print(f"[INFO] Label rows:    {len(labels):,}")

    train_lookup = build_training_lookup(train)

    mismatched = find_mismatches(labels, train_lookup)

    print(f"[INFO] Mismatched label rows: {len(mismatched):,}")

    mismatched.to_csv(args.output, index=False)
    print(f"[INFO] Exported: {args.output}")


if __name__ == "__main__":
    main()