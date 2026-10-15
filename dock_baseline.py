#!/usr/bin/env python3
"""
dock_baseline.py - dock your whole test set (actives + decoys) with AutoDock Vina and
measure how well plain docking separates real Mpro inhibitors from decoys.

This is your BASELINE. The quantum step later has to beat these numbers.

What it does
  1. Prepares the protein for Vina (adds hydrogens, converts to .pdbqt) with Open Babel.
  2. Finds the pocket automatically from where nirmatrelvir sits (nirmatrelvir.pdb).
  3. For every molecule: makes a 3D shape (RDKit), converts it to .pdbqt (Meeko),
     and docks it (Vina). Many molecules run in parallel on your CPU cores.
  4. Saves scores as it goes (safe to stop with Ctrl+C and re-run: it resumes).
  5. Prints the evaluation: AUC, enrichment, and where nirmatrelvir ranked.

Usage (from your project folder):
  python dock_baseline.py --pilot          # quick test: 20 actives + 200 decoys
  python dock_baseline.py                  # the full run (1,000+ molecules)

If your files are not in the project folder, point to them:
  --protein path/mpro_protein.pdb  --pocket-ligand path/nirmatrelvir.pdb  --testset path/mpro_testset.csv
"""

import argparse
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

COLS = ["id", "name", "smiles", "label", "is_reference"]
_CFG = {}


# ----------------------------------------------------------------------------
# 1. Protein preparation and pocket box
# ----------------------------------------------------------------------------
def prepare_receptor(pdb_path, out_path):
    """Add hydrogens (pH 7.4) and convert the protein to a rigid .pdbqt file."""
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"Using existing receptor file {out_path}")
        return
    print("Preparing protein with Open Babel ...")
    obabel = shutil.which("obabel")
    if obabel:
        res = subprocess.run([obabel, pdb_path, "-O", out_path, "-xr", "-p", "7.4"],
                             capture_output=True, text=True)
        if res.returncode != 0 or not os.path.exists(out_path):
            sys.exit(f"Open Babel failed:\n{res.stderr[-800:]}")
        return
    try:
        from openbabel import pybel
        mol = next(pybel.readfile("pdb", pdb_path))
        mol.OBMol.AddHydrogens(False, True, 7.4)
        mol.write("pdbqt", out_path, overwrite=True, opt={"r": True})
    except Exception as e:
        sys.exit("Could not find Open Babel (`obabel`) or its Python module.\n"
                 f"Details: {e}\nInstall it with: conda install -c conda-forge openbabel")


def pocket_box(ligand_pdb, padding):
    """Box center = middle of the reference drug; size = drug size + padding."""
    xyz = []
    with open(ligand_pdb) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                except ValueError:
                    continue
    if not xyz:
        sys.exit(f"No atoms found in {ligand_pdb}")
    arr = np.array(xyz)
    center = ((arr.max(0) + arr.min(0)) / 2).round(3).tolist()
    extent = arr.max(0) - arr.min(0)
    size = np.clip(extent + padding, 22.0, 36.0).round(1).tolist()
    return center, size


# ----------------------------------------------------------------------------
# 2. Ligand preparation and docking (runs inside worker processes)
# ----------------------------------------------------------------------------
def embed_3d(smiles, seed=42):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    return mol


def to_pdbqt(mol):
    """Convert a 3D RDKit molecule to a Vina .pdbqt string with Meeko (any recent version)."""
    from meeko import MoleculePreparation
    prep = MoleculePreparation()
    try:
        from meeko import PDBQTWriterLegacy
    except ImportError:  # very old Meeko
        prep.prepare(mol)
        return prep.write_pdbqt_string()
    setups = prep.prepare(mol)
    text, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise RuntimeError(f"Meeko: {err}")
    return text


def _init_worker(cfg):
    global _CFG
    _CFG = cfg


def dock_one(row):
    from vina import Vina

    out = {k: row[k] for k in COLS}
    out.update(vina_score=np.nan, seconds=0.0, error="")
    t0 = time.time()
    try:
        mol = embed_3d(row["smiles"], _CFG["seed"])
        if mol is None:
            raise RuntimeError("3D embedding failed")
        pdbqt = to_pdbqt(mol)
        v = Vina(sf_name="vina", cpu=1, seed=_CFG["seed"], verbosity=0)
        v.set_receptor(_CFG["receptor"])
        v.set_ligand_from_string(pdbqt)
        v.compute_vina_maps(center=_CFG["center"], box_size=_CFG["size"])
        v.dock(exhaustiveness=_CFG["exhaustiveness"], n_poses=1)
        out["vina_score"] = float(v.energies(n_poses=1)[0][0])
    except Exception as e:
        out["error"] = str(e).replace("\n", " ")[:200]
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ----------------------------------------------------------------------------
# 3. Evaluation
# ----------------------------------------------------------------------------
def evaluate(results_csv, out_dir):
    from sklearn.metrics import roc_auc_score

    df = pd.read_csv(results_csv)
    df = df.drop_duplicates("id", keep="last")  # a re-run replaces an earlier failure
    # The same molecule can appear under several IDs (nirmatrelvir does): count it once,
    # keeping the copy flagged as the reference.
    df["_is_ref"] = df["is_reference"].astype(str).str.lower().isin(["yes", "true"])
    df = df.sort_values("_is_ref", ascending=False).drop_duplicates("smiles", keep="first")
    df = df.drop(columns="_is_ref").reset_index(drop=True)
    total = len(df)
    ok = df["vina_score"].notna()
    if ok.sum() == 0 or (~ok).mean() > 0.2:
        print("\nMany molecules failed. Most common errors:")
        for msg, c in df.loc[~ok, "error"].fillna("(no message)").value_counts().head(5).items():
            print(f"  {c}x  {msg}")
    df = df[ok].reset_index(drop=True)
    failed = total - len(df)
    y = (df["label"] == "active").astype(int).values
    if y.sum() == 0 or y.sum() == len(y):
        print("Need both actives and decoys with scores to evaluate.")
        return

    # Vina: MORE NEGATIVE = better, so rank ascending.
    df = df.sort_values("vina_score").reset_index(drop=True)
    y = (df["label"] == "active").astype(int).values
    n = len(df)
    base_rate = y.mean()

    lines = [f"Docked and scored: {n} molecules ({int(y.sum())} actives, {int((1 - y).sum())} decoys); "
             f"{failed} failed and were skipped."]

    auc = roc_auc_score(y, -df["vina_score"])
    heavy = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in df["smiles"]]
    size_auc = roc_auc_score(y, heavy)
    lines.append(f"\nVina AUC: {auc:.2f}      (0.50 = random, 1.00 = perfect)")
    lines.append(f"Size-only AUC (just counting atoms): {size_auc:.2f}   <- Vina should beat this")

    lines.append("\nEnrichment (how many times better than picking at random):")
    for frac in (0.01, 0.05, 0.10):
        top = max(1, math.ceil(frac * n))
        hits = int(y[:top].sum())
        ef = (hits / top) / base_rate
        lines.append(f"  top {int(frac * 100):>2}% ({top:>3} molecules): {hits:>2} actives found -> EF {ef:.1f}x")
    lines.append(f"  (best possible EF at 10% here is about {min(1 / base_rate, 1 / 0.10):.1f}x)")

    ref = df[df["is_reference"].astype(str).str.lower().isin(["yes", "true"])]
    if len(ref):
        r = ref.iloc[0]
        rank = int(ref.index[0]) + 1
        lines.append(f"\nNirmatrelvir (reference): score {r['vina_score']:.2f}, "
                     f"rank {rank} of {n} (top {100 * rank / n:.0f}%)")
    else:
        lines.append("\nNirmatrelvir (reference) was not scored - check is_reference or the errors column.")

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(out_dir, "baseline_summary.txt"), "w") as fh:
        fh.write(text + "\n")
    df.to_csv(os.path.join(out_dir, "docking_ranked.csv"), index=False)
    print(f"\nSaved {os.path.join(out_dir, 'baseline_summary.txt')} and "
          f"{os.path.join(out_dir, 'docking_ranked.csv')}")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Dock a test set with AutoDock Vina and evaluate it.")
    ap.add_argument("--testset", default="testset/mpro_testset.csv")
    ap.add_argument("--protein", default="mpro_protein.pdb")
    ap.add_argument("--pocket-ligand", default="nirmatrelvir.pdb",
                    help="PDB of the known drug in the pocket; defines the docking box")
    ap.add_argument("--out-dir", default="docking")
    ap.add_argument("--exhaustiveness", type=int, default=8, help="Vina effort (default 8)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--box-padding", type=float, default=10.0)
    ap.add_argument("--pilot", action="store_true", help="small quick test run")
    ap.add_argument("--pilot-actives", type=int, default=20)
    ap.add_argument("--pilot-decoys", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--evaluate-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results_csv = os.path.join(args.out_dir, "pilot_scores.csv" if args.pilot else "docking_scores.csv")

    if args.evaluate_only:
        evaluate(results_csv, args.out_dir)
        return

    for path in (args.testset, args.protein, args.pocket_ligand):
        if not os.path.exists(path):
            sys.exit(f"Cannot find {path}. Use the --testset / --protein / --pocket-ligand options.")

    df = pd.read_csv(args.testset)
    if args.pilot:
        rng = np.random.default_rng(args.seed)
        is_ref = df["is_reference"].astype(str).str.lower().isin(["yes", "true"])
        act = df[(df["label"] == "active") & ~is_ref]
        dec = df[df["label"] == "decoy"]
        df = pd.concat([
            df[is_ref],
            act.iloc[rng.permutation(len(act))[:max(0, args.pilot_actives - int(is_ref.sum()))]],
            dec.iloc[rng.permutation(len(dec))[:args.pilot_decoys]],
        ]).reset_index(drop=True)
        print(f"PILOT run: {len(df)} molecules")

    receptor = os.path.join(args.out_dir, "mpro_protein.pdbqt")
    prepare_receptor(args.protein, receptor)
    center, size = pocket_box(args.pocket_ligand, args.box_padding)
    print(f"Docking box center {center}, size {size} (angstroms)")

    done = set()
    if os.path.exists(results_csv):
        prev = pd.read_csv(results_csv)
        done = set(prev.loc[prev["vina_score"].notna(), "id"].astype(str))
        print(f"Resuming: {len(done)} molecules already scored (failed ones will be retried)")
    todo = [r for r in df.to_dict("records") if str(r["id"]) not in done]
    if not todo:
        print("Everything is already docked.")
        evaluate(results_csv, args.out_dir)
        return

    cfg = {"receptor": receptor, "center": center, "size": size,
           "exhaustiveness": args.exhaustiveness, "seed": args.seed}
    print(f"Docking {len(todo)} molecules on {args.workers} workers ... (Ctrl+C is safe; re-run to resume)")

    header = not os.path.exists(results_csv)
    start, count = time.time(), 0
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        try:
            for res in pool.imap_unordered(dock_one, todo, chunksize=1):
                pd.DataFrame([res]).to_csv(results_csv, mode="a", header=header, index=False)
                header = False
                count += 1
                if count % 10 == 0 or count == len(todo):
                    elapsed = time.time() - start
                    eta = elapsed / count * (len(todo) - count)
                    print(f"  {count}/{len(todo)} done, about {eta / 60:.0f} min left")
        except KeyboardInterrupt:
            print("\nStopped. Scores so far are saved; re-run the same command to continue.")
            pool.terminate()
            return

    evaluate(results_csv, args.out_dir)


if __name__ == "__main__":
    main()
