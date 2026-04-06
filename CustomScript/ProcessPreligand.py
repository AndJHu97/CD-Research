import argparse
import csv
import json
import os
import re
import shutil
import socket
import threading
import time
import urllib.error
import urllib.request

import anthropic
from dotenv import load_dotenv
from rdkit import Chem


DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_MAX_TOKENS = 350
DEFAULT_DOWNLOAD_TIMEOUT_SEC = 12
TRANSFORMED_COL = "transformed_ligand_smiles"
IS_TRANSFORMED_COL = "IsTransformed"
CLAUDE_TRANSFORMED_COL = "claude_is_transformed"
VALIDATION_TRANSFORMED_COL = "validation_is_transformed"
API_CALLED_COL = "api_called"
WARHEAD_COL = "warhead"
REASON_COL = "reason"
REACTION_COMPATIBLE_COL = "is_reaction_compatible"


def safe_get(row: dict, key: str) -> str:
    value = row.get(key, "")
    return str(value).strip() if value is not None else ""


def safe_get_any(row: dict, keys: list[str]) -> str:
    for key in keys:
        value = safe_get(row, key)
        if value:
            return value
    return ""


def canonicalize_smiles(smiles: str) -> str:
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol)


def find_local_pdb(lig_id: str, pdb_dir: str) -> str:
    """Find a PDB file by LigID (basename match, case-insensitive) in a directory tree."""
    if not lig_id or not pdb_dir or not os.path.isdir(pdb_dir):
        return ""

    lig_id_lower = lig_id.lower()
    direct = os.path.join(pdb_dir, f"{lig_id}.pdb")
    if os.path.isfile(direct):
        return direct

    for root, _, files in os.walk(pdb_dir):
        for fname in files:
            if not fname.lower().endswith(".pdb"):
                continue
            stem = os.path.splitext(fname)[0].lower()
            if stem == lig_id_lower:
                return os.path.join(root, fname)
    return ""


def download_pdb(lig_id: str, download_dir: str) -> str:
    """Download PDB from RCSB using LigID as PDB code."""
    if not lig_id:
        return ""

    os.makedirs(download_dir, exist_ok=True)
    lig_id_upper = lig_id.upper()
    destination = os.path.join(download_dir, f"{lig_id_upper}.pdb")
    if os.path.isfile(destination):
        return destination

    url = f"https://files.rcsb.org/download/{lig_id_upper}.pdb"
    try:
        with urllib.request.urlopen(url, timeout=DEFAULT_DOWNLOAD_TIMEOUT_SEC) as response:
            with open(destination, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        return destination
    except (urllib.error.URLError, socket.timeout, Exception):
        return ""


def smiles_from_pdb(pdb_path: str) -> str:
    """Convert a ligand PDB file to SMILES."""
    if not pdb_path or not os.path.isfile(pdb_path):
        return ""

    mol = Chem.MolFromPDBFile(pdb_path, sanitize=True, removeHs=False)
    if mol is None:
        mol = Chem.MolFromPDBFile(pdb_path, sanitize=False, removeHs=False)
    if mol is None:
        return ""

    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""


def download_ligand_component_sdf(lig_id: str, download_dir: str) -> str:
    """Download RCSB ligand-component ideal SDF by 3-letter/3-char LigID."""
    if not lig_id:
        return ""

    os.makedirs(download_dir, exist_ok=True)
    lig_id_upper = lig_id.upper()
    destination = os.path.join(download_dir, f"{lig_id_upper}_ideal.sdf")
    if os.path.isfile(destination):
        return destination

    url = f"https://files.rcsb.org/ligands/download/{lig_id_upper}_ideal.sdf"
    try:
        with urllib.request.urlopen(url, timeout=DEFAULT_DOWNLOAD_TIMEOUT_SEC) as response:
            with open(destination, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        return destination
    except (urllib.error.URLError, socket.timeout, Exception):
        return ""


def smiles_from_sdf(sdf_path: str) -> str:
    """Convert first molecule in an SDF file to SMILES."""
    if not sdf_path or not os.path.isfile(sdf_path):
        return ""

    try:
        supplier = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False)
        for mol in supplier:
            if mol is not None:
                return Chem.MolToSmiles(mol)
    except Exception:
        return ""
    return ""


def smiles_from_sdf_with_timeout(sdf_path: str, timeout_sec: int = 10) -> str:
    """Extract SMILES from SDF with timeout to prevent hangs on malformed files."""
    result = [""]
    
    def extract():
        try:
            result[0] = smiles_from_sdf(sdf_path)
        except Exception:
            result[0] = ""
    
    thread = threading.Thread(target=extract, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)
    
    if thread.is_alive():
        # Thread still running after timeout, return empty string
        return ""
    return result[0]


def extract_json(text: str) -> dict:
    """Extract the first JSON object from model output."""
    text = (text or "").strip()
    if not text:
        return {}

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}

    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def looks_like_fragment_output(input_smiles: str, output_smiles: str) -> bool:
    """Return True when output looks like a warhead-only fragment, not full electrophile."""
    if not input_smiles or not output_smiles:
        return False

    in_mol = Chem.MolFromSmiles(input_smiles)
    out_mol = Chem.MolFromSmiles(output_smiles)
    if in_mol is None or out_mol is None:
        return False

    in_heavy = in_mol.GetNumHeavyAtoms()
    out_heavy = out_mol.GetNumHeavyAtoms()
    if in_heavy <= 0 or out_heavy <= 0:
        return False

    # Very small output relative to input is likely just a warhead fragment.
    if out_heavy < max(10, int(in_heavy * 0.55)):
        return True

    in_rings = in_mol.GetRingInfo().NumRings()
    out_rings = out_mol.GetRingInfo().NumRings()
    if in_rings > 0 and out_rings == 0 and (in_heavy - out_heavy) >= 8:
        return True

    return False


def ask_claude_for_prereaction_smiles(
    client: anthropic.Anthropic,
    input_smiles: str,
    warhead_type: str,
    notes: str,
    lig_id: str,
    model: str,
    max_tokens: int,
    api_timeout_sec: int,
) -> dict:
    """Ask Claude to infer pre-reaction parent electrophile and warhead state."""
    json_schema = (
        "{\n"
        '  "pre_reaction_smiles": "<SMILES string of the whole input electrophile smile that has warhead reformed>",\n'
        '  "is_reaction_compatible": true or false,\n'
        '  "warhead": "<short warhead label>",\n'
        '  "is_transformed": true or false,\n'
        '  "reason": "one short sentence"\n'
        "}"
    )

    system_prompt = (
        "Return only one valid JSON object and no other text. "
        "Do not include explanations, reasoning, markdown, or code fences. "
        "The JSON must strictly match the requested keys and value types."
    )

    prompt = (
        "You are an expert medicinal-chemistry assistant for covalent inhibitor design.\n"
        "Goal: SDF for electrophilic ligands often have the post-reaction covalently bounded state of electrophile. Can you convert this input electrophile into the correct PRE-reaction electrophile if applicable? The warhead to reform will be given along with additional notes if applicable.\n"
        "Preserve the scaffold and make the smallest plausible edit to reconstruct the pre-reaction electrophilic warhead on the input smiles.\n\n"
        f"Input (potential post-reaction) SMILES: {input_smiles}\n"
        f"Warhead type to reform for the electrophile: {warhead_type or 'unknown'}\n"
        f"Notes: {notes or 'none'}\n"
        f"LigID: {lig_id or 'unknown'}\n\n"
        "Output strict JSON only with keys:\n"
        f"{json_schema}\n"
        "Rules:\n"
        "- No markdown, no code fences, no extra keys.\n"
        "- If already pre-reaction and reaction-compatible, return the same SMILES and is_transformed=false.\n"
        "- If not compatible or post-reaction, return a corrected pre-reaction electrophile with the warhead reformed and is_transformed=true.\n"
        "- pre_reaction_smiles must be a single valid SMILES string."
    )

    def run_model(user_prompt: str) -> str:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            timeout=api_timeout_sec,
        )
        combined = ""
        for block in message.content:
            if getattr(block, "type", "") == "text":
                combined += block.text
        return combined

    try:
        text = run_model(prompt)
    except Exception as exc:
        return {
            "raw": "",
            "pre_reaction_smiles": "",
            "is_reaction_compatible": None,
            "warhead": "",
            "is_transformed": False,
            "reason": "",
            "api_error": str(exc),
        }

    parsed = extract_json(text)

    # If the model returned prose instead of JSON, do one repair pass.
    if not parsed:
        repair_prompt = (
            "Convert the following text into exactly one valid JSON object using this schema:\n"
            f"{json_schema}\n\n"
            "Input text to convert:\n"
            f"{text}\n\n"
            "Return only JSON."
        )
        try:
            repair_text = run_model(repair_prompt)
            if repair_text.strip():
                text = repair_text
            parsed = extract_json(text)
        except Exception:
            pass

    # Guardrail: if model returned only a small fragment, force full scaffold retry.
    current_pre = str(parsed.get("pre_reaction_smiles", "")).strip()
    if current_pre and looks_like_fragment_output(input_smiles, current_pre):
        full_scaffold_prompt = (
            "Your previous pre_reaction_smiles appears to be a fragment.\n"
            "Return the full electrophile molecule, preserving the input scaffold and changing only the warhead region if needed.\n"
            "Do not return only the warhead.\n\n"
            f"Input full molecule SMILES: {input_smiles}\n"
            f"Previous fragment-like output: {current_pre}\n"
            f"Warhead type: {warhead_type or 'unknown'}\n"
            f"Notes: {notes or 'none'}\n"
            f"LigID: {lig_id or 'unknown'}\n\n"
            "Return strict JSON only with this schema:\n"
            f"{json_schema}\n"
            "No markdown, no extra text."
        )
        try:
            retry_text = run_model(full_scaffold_prompt)
            retry_parsed = extract_json(retry_text)
            retry_pre = str(retry_parsed.get("pre_reaction_smiles", "")).strip()
            if retry_parsed and retry_pre and not looks_like_fragment_output(input_smiles, retry_pre):
                text = retry_text
                parsed = retry_parsed
        except Exception:
            pass

    return {
        "raw": text.strip(),
        "pre_reaction_smiles": str(parsed.get("pre_reaction_smiles", "")).strip(),
        "is_reaction_compatible": parsed.get("is_reaction_compatible", None),
        "warhead": str(parsed.get("warhead", "")).strip(),
        "is_transformed": parsed.get("is_transformed", False),
        "reason": str(parsed.get("reason", "")).strip(),
        "api_error": "",
    }


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def process_row(
    row: dict,
    client: anthropic.Anthropic,
    pdb_dir: str,
    pdb_download_dir: str,
    model: str,
    max_tokens: int,
    api_timeout_sec: int,
) -> dict:
    out = dict(row)
    out[API_CALLED_COL] = False
    out[CLAUDE_TRANSFORMED_COL] = ""
    out[VALIDATION_TRANSFORMED_COL] = False
    out[WARHEAD_COL] = ""
    out[REASON_COL] = ""
    out[REACTION_COMPATIBLE_COL] = ""

    row_smiles = safe_get_any(
        row,
        ["electrophile_smiles", "electrophile-smiles", "smiles", "SMILES"],
    )
    lig_id = safe_get_any(row, ["LigID", "ligid", "lig_id", "LIGID"])
    warhead_type = safe_get_any(
        row,
        ["warhead-type", "warhead_type", "warhead type", "reaction-type", "reaction_type", "reaction type"],
    )
    notes = safe_get(row, "notes")

    working_smiles = row_smiles
    if not working_smiles and lig_id:
        pdb_path = find_local_pdb(lig_id, pdb_dir) if pdb_dir else ""
        if not pdb_path:
            pdb_path = download_pdb(lig_id, pdb_download_dir)
        if pdb_path:
            working_smiles = smiles_from_pdb(pdb_path)

        if not working_smiles:
            sdf_path = download_ligand_component_sdf(lig_id, pdb_download_dir)
            if sdf_path:
                working_smiles = smiles_from_sdf_with_timeout(sdf_path, timeout_sec=10)

    if not working_smiles:
        out[IS_TRANSFORMED_COL] = False
        out[VALIDATION_TRANSFORMED_COL] = False
        out[TRANSFORMED_COL] = ""
        return out

    print(
        "[API CALL] "
        f"LigID={lig_id or 'N/A'} | "
        f"warhead_type={warhead_type or 'N/A'} | "
        f"smiles_len={len(working_smiles)}"
    )

    claude_result = ask_claude_for_prereaction_smiles(
        client=client,
        input_smiles=working_smiles,
        warhead_type=warhead_type,
        notes=notes,
        lig_id=lig_id,
        model=model,
        max_tokens=max_tokens,
        api_timeout_sec=api_timeout_sec,
    )
    out[API_CALLED_COL] = True

    if claude_result.get("api_error"):
        print(f"[API ERROR] LigID={lig_id or 'N/A'} | {claude_result.get('api_error')}")
        out[IS_TRANSFORMED_COL] = False
        out[VALIDATION_TRANSFORMED_COL] = False
        out[TRANSFORMED_COL] = working_smiles
        return out

    print(f"[API RESP] LigID={lig_id or 'N/A'} | received response")
    print(f"[API RAW] LigID={lig_id or 'N/A'} | {claude_result.get('raw', '')}")
    print(
        f"[API PARSED] LigID={lig_id or 'N/A'} | "
        f"pre_reaction_smiles={claude_result.get('pre_reaction_smiles', '')} | "
        f"is_reaction_compatible={claude_result.get('is_reaction_compatible', '')} | "
        f"warhead={claude_result.get('warhead', '')} | "
        f"is_transformed={claude_result.get('is_transformed', '')} | "
        f"reason={claude_result.get('reason', '')}"
    )

    out[WARHEAD_COL] = claude_result.get("warhead", "")
    out[REASON_COL] = claude_result.get("reason", "")
    claude_is_transformed = claude_result.get("is_transformed", None)
    if isinstance(claude_is_transformed, bool):
        out[CLAUDE_TRANSFORMED_COL] = claude_is_transformed
    elif claude_is_transformed in (None, ""):
        out[CLAUDE_TRANSFORMED_COL] = ""
    else:
        out[CLAUDE_TRANSFORMED_COL] = to_bool(claude_is_transformed)

    reaction_compatible = claude_result.get("is_reaction_compatible", None)
    if isinstance(reaction_compatible, bool):
        out[REACTION_COMPATIBLE_COL] = reaction_compatible
    elif reaction_compatible is None:
        out[REACTION_COMPATIBLE_COL] = ""
    else:
        out[REACTION_COMPATIBLE_COL] = str(reaction_compatible)

    transformed_smiles = claude_result.get("pre_reaction_smiles", "")
    if not transformed_smiles:
        transformed_smiles = working_smiles

    input_canon = canonicalize_smiles(working_smiles)
    output_canon = canonicalize_smiles(transformed_smiles)

    if input_canon and output_canon:
        is_transformed = input_canon != output_canon
    else:
        is_transformed = working_smiles.strip() != transformed_smiles.strip()

    if not output_canon:
        # Keep original if Claude output is not parseable SMILES.
        transformed_smiles = working_smiles
        is_transformed = False
    elif not is_transformed:
        transformed_smiles = working_smiles

    out[VALIDATION_TRANSFORMED_COL] = is_transformed
    out[IS_TRANSFORMED_COL] = is_transformed
    out[TRANSFORMED_COL] = transformed_smiles
    return out


def process_csv(
    input_csv: str,
    output_csv: str,
    client: anthropic.Anthropic,
    pdb_dir: str,
    pdb_download_dir: str,
    model: str,
    max_tokens: int,
    api_timeout_sec: int,
) -> None:
    with open(input_csv, "r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")

        required_any = {"electrophile_smiles", "LigID"}
        if not any(h in reader.fieldnames for h in required_any):
            raise ValueError(
                "CSV must contain at least one of these headers: electrophile_smiles or LigID"
            )

        output_headers = list(reader.fieldnames)
        if API_CALLED_COL not in output_headers:
            output_headers.append(API_CALLED_COL)
        if IS_TRANSFORMED_COL not in output_headers:
            output_headers.append(IS_TRANSFORMED_COL)
        if CLAUDE_TRANSFORMED_COL not in output_headers:
            output_headers.append(CLAUDE_TRANSFORMED_COL)
        if VALIDATION_TRANSFORMED_COL not in output_headers:
            output_headers.append(VALIDATION_TRANSFORMED_COL)
        if TRANSFORMED_COL not in output_headers:
            output_headers.append(TRANSFORMED_COL)
        if REACTION_COMPATIBLE_COL not in output_headers:
            output_headers.append(REACTION_COMPATIBLE_COL)
        if WARHEAD_COL not in output_headers:
            output_headers.append(WARHEAD_COL)
        if REASON_COL not in output_headers:
            output_headers.append(REASON_COL)

        rows_out = []
        for row_idx, row in enumerate(reader, start=1):
            row_start = time.perf_counter()
            lig_id_preview = safe_get_any(row, ["LigID", "ligid", "lig_id", "LIGID"]) or "N/A"
            warhead_preview = safe_get_any(
                row,
                ["warhead-type", "warhead_type", "warhead type", "reaction-type", "reaction_type", "reaction type"],
            ) or "N/A"
            print(f"[ROW START] idx={row_idx} | LigID={lig_id_preview} | warhead_type={warhead_preview}")

            processed = process_row(
                row=row,
                client=client,
                pdb_dir=pdb_dir,
                pdb_download_dir=pdb_download_dir,
                model=model,
                max_tokens=max_tokens,
                api_timeout_sec=api_timeout_sec,
            )
            rows_out.append(processed)
            row_elapsed = time.perf_counter() - row_start
            print(
                f"[ROW END] idx={row_idx} | "
                f"api_called={processed.get(API_CALLED_COL, False)} | "
                f"claude_transformed={processed.get(CLAUDE_TRANSFORMED_COL, '')} | "
                f"validation_transformed={processed.get(VALIDATION_TRANSFORMED_COL, False)} | "
                f"elapsed={row_elapsed:.2f}s"
            )

    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=output_headers)
        writer.writeheader()
        writer.writerows(rows_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform ligand SMILES to pre-reaction form using Claude."
    )
    parser.add_argument("--input-csv", required=True, help="Input CSV path")
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument(
        "--pdb-dir",
        default="",
        help="Directory to search for local PDB files by LigID (recursive)",
    )
    parser.add_argument(
        "--pdb-download-dir",
        default="downloaded_pdbs",
        help="Directory to save downloaded PDB files",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Claude model name (overrides ANTHROPIC_MODEL env var)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max tokens for Claude response",
    )
    parser.add_argument(
        "--api-timeout-sec",
        type=int,
        default=45,
        help="Per-row API timeout in seconds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Local .env supports development; host-provided env vars still take precedence.
    load_dotenv(override=False)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. Set it on your host or in a local .env file."
        )

    env_model = (os.getenv("ANTHROPIC_MODEL") or "").strip()
    selected_model = (args.model or "").strip() or env_model or DEFAULT_MODEL

    client = anthropic.Anthropic(api_key=api_key)

    process_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        client=client,
        pdb_dir=args.pdb_dir,
        pdb_download_dir=args.pdb_download_dir,
        model=selected_model,
        max_tokens=args.max_tokens,
        api_timeout_sec=args.api_timeout_sec,
    )
    print(f"Model used: {selected_model}")
    print(f"Wrote transformed results to: {args.output_csv}")


if __name__ == "__main__":
    main()