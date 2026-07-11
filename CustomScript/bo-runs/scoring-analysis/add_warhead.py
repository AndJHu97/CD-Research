import pandas as pd

# --- file paths ---
csv1_path = "batch_pdbs_bo.csv"   # main file (to be updated)
csv2_path = "Covalent_Complex_Records_with_name.csv"   # lookup file (has Warhead column)
output_path = "batch_pdbs_bo_warheads.csv"

# --- load data ---
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

# --- sanity check ---
required_cols_1 = {"name"}
required_cols_2 = {"Name", "Warhead"}

if not required_cols_1.issubset(df1.columns):
    raise ValueError("CSV1 must contain 'Name' column")

if not required_cols_2.issubset(df2.columns):
    raise ValueError("CSV2 must contain 'Name' and 'Warhead' columns")

# --- create lookup table ---
df2 = df2.drop_duplicates(subset="Name", keep="first")

lookup = df2.set_index("Name")["Warhead"]
df1["Warhead"] = df1["name"].map(lookup)

# --- map Warhead into first dataframe ---
df1["Warhead"] = df1["name"].map(lookup)

# --- save result ---
df1.to_csv(output_path, index=False)

print(f"Saved updated file to: {output_path}")