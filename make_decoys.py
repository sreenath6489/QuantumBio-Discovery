#!/usr/bin/env python3
"""
make_decoys.py - build property-matched decoys for a virtual-screening test set.

What it does (same idea as the DUD-E benchmark):
  1. Reads your ACTIVES (known Mpro inhibitors) and a big POOL of drug-like molecules.
  2. For every active, picks N pool molecules with a similar size, greasiness (logP),
     H-bond donors/acceptors, rotatable bonds and the SAME net charge.
  3. Throws out any pool molecule that is structurally similar to a known active
     (so decoys are very unlikely to be hidden inhibitors).
  4. Writes the decoys and a combined test set (actives + decoys), plus a report that
     shows actives vs decoys have matching properties.

Install:   conda install -c conda-forge rdkit    (then: pip install pandas numpy)

Example:
  python make_decoys.py \
      --actives mpro_actives.csv \
      --pool chembl_pool.csv \
      --exclude chembl_mpro_all_6200_rows.csv \
      --per-active 10 \
      --out-dir testset

Input files
  --actives   CSV with a `smiles` column (and ideally id, name, label, is_reference).
  --pool      CSV (comma or semicolon) with a SMILES column, or a .smi file
              (one SMILES per line, optional ID after a space).
  --exclude   (optional, recommended) other CSV/.smi files of KNOWN Mpro inhibitors,
              e.g. the full ChEMBL download you filtered your top 100 from. These are
              removed from the pool and used in the similarity filter.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

PROPS = ["mw", "logp", "hbd", "hba", "rb"]
# How much each property may differ before it counts as "one unit" of mismatch.
SCALES = np.array([25.0, 0.7, 1.0, 1.0, 1.0])

SMILES_NAMES = ["smiles", "canonical_smiles", "molecule smiles", "smile"]
ID_NAMES = ["id", "molecule_chembl_id", "molecule chembl id", "chembl_id",
            "zinc_id", "zinc id", "compound_id", "name"]

_chooser = rdMolStandardize.LargestFragmentChooser()
_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ----------------------------------------------------------------------------
# Reading files
# ----------------------------------------------------------------------------
def load_table(path):
    """Read a CSV/TSV/semicolon file, or a .smi file, into a DataFrame of strings."""
    if path.lower().endswith((".smi", ".ism", ".txt")):
        rows = []
        with open(path) as fh:
            for line in fh:
                parts = line.strip().split(None, 1)
                if not parts or parts[0].lower() == "smiles":
                    continue
                rows.append((parts[0], parts[1] if len(parts) > 1 else ""))
        return pd.DataFrame(rows, columns=["smiles", "id"])
    try:
        return pd.read_csv(path, sep=None, engine="python", dtype=str)
    except Exception:
        return pd.read_csv(path, dtype=str)  # single-column files can't be sniffed


def find_col(df, names):
    lower = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n in lower:
            return lower[n]
    return None


# ----------------------------------------------------------------------------
# Chemistry helpers
# ----------------------------------------------------------------------------
def to_mol(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    try:
        mol = _chooser.choose(mol)  # keep the biggest piece (drops salts)
    except Exception:
        pass
    return mol


def properties(mol):
    return (
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        Chem.GetFormalCharge(mol),
    )


def featurize(df, label, progress=0):
    """Parse every SMILES and compute properties. Returns a clean DataFrame."""
    scol = find_col(df, SMILES_NAMES)
    if scol is None:
        sys.exit(f"[{label}] no SMILES column found. Columns are: {list(df.columns)}")
    icol = find_col(df, ID_NAMES)

    rows, bad = [], 0
    for n, (_, r) in enumerate(df.iterrows()):
        if progress and n and n % progress == 0:
            print(f"  [{label}] processed {n:,} / {len(df):,}")
        mol = to_mol(r[scol])
        if mol is None:
            bad += 1
            continue
        mw, logp, hbd, hba, rb, charge = properties(mol)
        rows.append({
            "row": r.name,
            "id": (r[icol] if icol and isinstance(r[icol], str) and r[icol].strip() else ""),
            "smiles_in": r[scol],
            "can": Chem.MolToSmiles(mol),
            "mw": mw, "logp": logp, "hbd": hbd, "hba": hba, "rb": rb, "charge": charge,
        })
    if bad:
        print(f"  [{label}] skipped {bad:,} rows with missing/invalid SMILES")
    return pd.DataFrame(rows)


def fingerprint(smiles):
    return _fpgen.GetFingerprint(Chem.MolFromSmiles(smiles))


# ----------------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Make property-matched decoys with RDKit.")
    ap.add_argument("--actives", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="extra files of known inhibitors to keep OUT of the decoys")
    ap.add_argument("--per-active", type=int, default=10, help="decoys per active (default 10)")
    ap.add_argument("--pool-factor", type=int, default=5,
                    help="randomly choose from the closest N*factor matches (default 5)")
    ap.add_argument("--max-sim", type=float, default=0.35,
                    help="reject decoys with Tanimoto >= this to any known active (default 0.35)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reference-smiles", default="",
                    help="SMILES of your headline drug (nirmatrelvir); marks it is_reference=yes")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) actives ------------------------------------------------------------
    print("Reading actives ...")
    act_raw = load_table(args.actives)
    act = featurize(act_raw, "actives")
    if act.empty:
        sys.exit("No valid actives found.")
    print(f"  {len(act)} valid actives")

    # 2) known inhibitors to exclude ---------------------------------------
    known_smiles = set(act["can"])
    ref_smiles = list(act["can"])
    for path in args.exclude:
        print(f"Reading exclude file {path} ...")
        ex = featurize(load_table(path), "exclude", progress=5000)
        known_smiles |= set(ex["can"])
        ref_smiles += list(ex["can"])
    ref_smiles = list(dict.fromkeys(ref_smiles))  # unique, keep order
    print(f"Building fingerprints for {len(ref_smiles):,} known inhibitors ...")
    ref_fps = [fingerprint(s) for s in ref_smiles]

    # 3) pool ---------------------------------------------------------------
    print("Reading pool (this can take a few minutes for 100k+ molecules) ...")
    pool = featurize(load_table(args.pool), "pool", progress=20000)
    before = len(pool)
    pool = pool.drop_duplicates("can")
    pool = pool[~pool["can"].isin(known_smiles)].reset_index(drop=True)
    print(f"  pool: {before:,} -> {len(pool):,} after removing duplicates and known inhibitors")
    if len(pool) < len(act) * args.per_active:
        print("  WARNING: pool is small; use a bigger pool for better matching.")

    pool_props = pool[PROPS].to_numpy(float)
    pool_charge = pool["charge"].to_numpy()
    available = np.ones(len(pool), dtype=bool)

    # 4) match decoys to each active ----------------------------------------
    print("Matching decoys ...")
    chosen_rows = []
    rejected_similar = 0
    shortfalls = []
    n = args.per_active

    for count, i in enumerate(rng.permutation(len(act))):  # random order = fairer
        a = act.iloc[i]
        a_props = np.array([a[p] for p in PROPS], dtype=float)

        idx = np.flatnonzero(available & (pool_charge == a["charge"]))
        if len(idx) == 0:
            shortfalls.append((a["id"] or a["can"][:20], 0))
            continue
        dist = np.sqrt((((pool_props[idx] - a_props) / SCALES) ** 2).sum(axis=1))
        ranked = idx[np.argsort(dist)]

        accepted, pointer = [], 0
        block = n * args.pool_factor
        while len(accepted) < n and pointer < len(ranked):
            chunk = ranked[pointer:pointer + block].copy()
            pointer += block
            rng.shuffle(chunk)
            for j in chunk:
                if len(accepted) >= n:
                    break
                available[j] = False  # never reconsider, accepted or not
                sim = max(DataStructs.BulkTanimotoSimilarity(
                    fingerprint(pool.at[j, "can"]), ref_fps))
                if sim >= args.max_sim:
                    rejected_similar += 1
                    continue
                accepted.append((j, sim))

        if len(accepted) < n:
            shortfalls.append((a["id"] or a["can"][:20], len(accepted)))
        for j, sim in accepted:
            chosen_rows.append({
                "pool_index": j,
                "matched_to": a["id"],
                "max_sim_to_known": sim,
            })
        if (count + 1) % 25 == 0:
            print(f"  matched {count + 1} / {len(act)} actives")

    if not chosen_rows:
        sys.exit("No decoys could be selected. Check your pool file.")

    ch = pd.DataFrame(chosen_rows)
    dec = pool.loc[ch["pool_index"]].reset_index(drop=True)
    dec["max_sim_to_known"] = ch["max_sim_to_known"].values

    # 5) write outputs ------------------------------------------------------
    decoy_ids = [
        (dec.at[k, "id"] if dec.at[k, "id"] else f"DECOY_{k + 1:06d}")
        for k in range(len(dec))
    ]
    decoys_out = pd.DataFrame({
        "id": decoy_ids,
        "name": decoy_ids,
        "smiles": dec["can"],
        "label": "decoy",
        "is_reference": "no",
    })

    # actives in the same 5-column layout (keeps the smiles exactly as you gave them)
    act_id = act["id"].where(act["id"] != "", [f"ACTIVE_{k + 1:04d}" for k in range(len(act))])
    name_col = find_col(act_raw, ["name"])
    ref_col = find_col(act_raw, ["is_reference"])
    src = act_raw.loc[act["row"]].reset_index(drop=True)
    is_ref = src[ref_col].fillna("no").values if ref_col else np.array(["no"] * len(act))
    if args.reference_smiles:
        ref_mol = to_mol(args.reference_smiles)
        if ref_mol is None:
            print("WARNING: --reference-smiles could not be parsed")
        else:
            hit = act["can"].values == Chem.MolToSmiles(ref_mol)
            if hit.any():
                is_ref = np.where(hit, "yes", "no")
            else:
                print("WARNING: the reference drug was not found among your actives")
    actives_out = pd.DataFrame({
        "id": act_id.values,
        "name": src[name_col].fillna("").values if name_col else act_id.values,
        "smiles": act["smiles_in"].values,
        "label": "active",
        "is_reference": is_ref,
    })

    testset = pd.concat([actives_out, decoys_out], ignore_index=True)
    decoys_path = os.path.join(args.out_dir, "mpro_decoys.csv")
    test_path = os.path.join(args.out_dir, "mpro_testset.csv")
    decoys_out.to_csv(decoys_path, index=False)
    testset.to_csv(test_path, index=False)

    # 6) report -------------------------------------------------------------
    lines = []
    lines.append(f"Actives: {len(actives_out)}   Decoys: {len(decoys_out)}   "
                 f"(ratio 1:{len(decoys_out) / len(actives_out):.1f})")
    lines.append(f"Pool molecules rejected for being too similar to a known active: {rejected_similar:,}")
    lines.append("")
    lines.append(f"{'property':<12}{'actives mean+-sd':<22}{'decoys mean+-sd':<22}")
    for p in PROPS + ["charge"]:
        a_vals, d_vals = act[p].astype(float), dec[p].astype(float)
        lines.append(f"{p:<12}{a_vals.mean():>8.2f} +- {a_vals.std():<8.2f}  "
                     f"{d_vals.mean():>8.2f} +- {d_vals.std():<8.2f}")
    lines.append("")
    lines.append(f"Decoy max similarity to any known inhibitor: mean {dec['max_sim_to_known'].mean():.2f}, "
                 f"highest {dec['max_sim_to_known'].max():.2f} (limit {args.max_sim})")
    if shortfalls:
        lines.append("")
        lines.append(f"{len(shortfalls)} actives got fewer than {n} decoys "
                     "(unusual charge or size; a bigger pool fixes this):")
        for name, got in shortfalls[:10]:
            lines.append(f"  {name}: {got}")
    report = "\n".join(lines)

    with open(os.path.join(args.out_dir, "decoy_report.txt"), "w") as fh:
        fh.write(report + "\n")

    print("\n" + report)
    print(f"\nWrote:\n  {decoys_path}\n  {test_path}\n  {os.path.join(args.out_dir, 'decoy_report.txt')}")


if __name__ == "__main__":
    main()
