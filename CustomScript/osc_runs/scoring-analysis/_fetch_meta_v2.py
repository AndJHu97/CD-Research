import json, urllib.request, time
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
a = pd.read_csv(r"c:\Users\Andre\Downloads\test_label_results_all.csv")
a["PDB"] = a["Name"].astype(str).str[:4].str.upper()
pdbs = sorted(a["PDB"].unique())
print("n pdbs", len(pdbs))

outp = HERE / "_rcsb_meta_v2.json"

query_tmpl = """
{
  entries(entry_ids: %s) {
    rcsb_id
    struct { title }
    rcsb_accession_info { initial_release_date }
    polymer_entities {
      rcsb_polymer_entity { pdbx_description }
      rcsb_entity_source_organism { ncbi_scientific_name }
    }
    nonpolymer_entities {
      pdbx_entity_nonpoly { name comp_id }
    }
    citation { title year journal_abbrev }
  }
}
"""

all_entries = []
for i in range(0, len(pdbs), 25):
    batch = pdbs[i:i + 25]
    q = query_tmpl % json.dumps(batch)
    req = urllib.request.Request(
        "https://data.rcsb.org/graphql",
        data=json.dumps({"query": q}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "research-annotation/1.0"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read().decode())
            if "errors" in data and not data.get("data"):
                raise RuntimeError(str(data["errors"])[:400])
            all_entries.extend(data["data"]["entries"])
            print(f"batch {i} ok total {len(all_entries)}", flush=True)
            break
        except Exception as e:
            print(f"batch {i} attempt {attempt} fail {e}", flush=True)
            time.sleep(2 + attempt)
    time.sleep(0.2)

outp.write_text(json.dumps(all_entries, indent=1), encoding="utf-8")
got = {e["rcsb_id"] for e in all_entries if e}
missing = [p for p in pdbs if p not in got]
print("WROTE", outp, "n", len(all_entries), "missing", missing)
