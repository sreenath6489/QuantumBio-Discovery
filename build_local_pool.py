#!/usr/bin/env python3
"""
build_local_pool.py - build a pool of virtual, drug-like molecules OFFLINE (no websites).

How: it takes a small library of common building blocks (acids, amines, amino-acid
linkers, sulfonyl chlorides) and joins them with simple, realistic reactions
(amide coupling, sulfonamide formation) using RDKit. Mixing 2, 3 or 4 blocks gives
molecules from small (~250 Da) to drug-sized (~550 Da).

The output (local_pool.csv) is the "pool" that make_decoys.py picks from. make_decoys.py
then keeps only molecules that match your actives' size/charge/properties AND are
structurally different from every known inhibitor.

Run a small test first:
  python build_local_pool.py --n 2000 --out local_pool_test.csv
Then the real thing:
  python build_local_pool.py --n 60000 --out local_pool.csv

Needs: rdkit, pandas   (conda install -c conda-forge rdkit)
"""

import argparse
import random

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

# --- building blocks (each has exactly ONE reactive group unless it is a linker) ----
ACIDS = [
    "OC(=O)c1ccccc1", "OC(=O)c1ccc(F)cc1", "OC(=O)c1ccc(Cl)cc1", "OC(=O)c1ccc(OC)cc1",
    "OC(=O)c1cccnc1", "OC(=O)c1ccncc1", "OC(=O)c1ccc2ccccc2c1", "OC(=O)c1cc2ccccc2[nH]1",
    "OC(=O)c1ccc2[nH]ccc2c1", "OC(=O)c1cnc2ccccn12", "OC(=O)c1ccco1", "OC(=O)c1cccs1",
    "OC(=O)c1ccc(C#N)cc1", "OC(=O)c1ccc(C(F)(F)F)cc1", "OC(=O)c1cc(C)on1",
    "OC(=O)c1cnn(C)c1", "OC(=O)c1nccn1C", "OC(=O)c1csc(C)n1", "OC(=O)C1CC1",
    "OC(=O)C1CCCCC1", "OC(=O)C1CCOCC1", "OC(=O)C(C)(C)C", "OC(=O)Cc1ccccc1",
    "OC(=O)Cc1ccc(F)cc1", "OC(=O)CCc1ccccc1", "OC(=O)COc1ccccc1",
    "OC(=O)C1CCN(C(C)=O)CC1", "OC(=O)c1ccc2OCOc2c1", "OC(=O)c1ccc2c(c1)OCCO2",
    "OC(=O)c1cc2ccccc2o1", "OC(=O)c1cc2ccccc2s1", "OC(=O)c1ccc(cc1)S(C)(=O)=O",
    "OC(=O)c1ccc(N2CCOCC2)cc1", "OC(=O)c1ccnc(Cl)c1", "OC(=O)C=Cc1ccccc1",
    "OC(=O)c1ccc2ncccc2c1", "OC(=O)c1cnc2ccccc2c1", "OC(=O)c1ccc(-c2ccccc2)cc1",
    "OC(=O)C1(c2ccccc2)CC1", "OC(=O)c1n[nH]c2ccccc12",
]

AMINES = [
    "Nc1ccccc1", "Nc1ccc(F)cc1", "Nc1ccc(Cl)cc1", "Nc1cccnc1", "Nc1ccc(OC)cc1",
    "NCc1ccccc1", "NCc1ccc(F)cc1", "NCc1ccccn1", "NCc1cccs1", "NCc1ccco1",
    "NCCc1ccccc1", "NCCc1c[nH]c2ccccc12", "NC1CC1", "NC1CCCCC1", "NC1CCOCC1",
    "NCC1CCCO1", "N1CCCC1", "N1CCCCC1", "N1CCOCC1", "N1CCN(C)CC1",
    "N1CCN(c2ccccc2)CC1", "N1CCN(C(C)=O)CC1", "CNCc1ccccc1", "N1CCC(CC1)c1ccccc1",
    "N1CCC(O)CC1", "NC(C)(C)C", "NCC(F)(F)F", "NCCOC", "NCc1ccc2OCOc2c1",
    "Nc1ccc2[nH]ncc2c1", "Nc1ccc2c(c1)OCO2", "Nc1cnn(C)c1", "Nc1ccc(cc1)N1CCOCC1",
    "NC1CCN(Cc2ccccc2)CC1", "NCC1CC1", "Nc1nccs1", "NC1CCCC1", "C1Cc2ccccc2CN1",
]

# amino-acid-like linkers: one amine + one carboxylic acid
LINKERS = [
    "NCC(=O)O", "NC(C)C(=O)O", "NC(C(C)C)C(=O)O", "NC(CC(C)C)C(=O)O",
    "NC(Cc1ccccc1)C(=O)O", "NC(C(C)(C)C)C(=O)O", "NC1(CC1)C(=O)O", "N1CCCC1C(=O)O",
    "N1CCCCC1C(=O)O", "NC(Cc1c[nH]c2ccccc12)C(=O)O", "NC(C1CCCCC1)C(=O)O",
    "NCCC(=O)O", "Nc1ccc(cc1)C(=O)O", "NCc1ccc(cc1)C(=O)O", "NCc1cccc(c1)C(=O)O",
    "Nc1cccc(c1)C(=O)O", "N1CCC(CC1)C(=O)O", "N1CCCC(C1)C(=O)O",
    "NC1CCC(CC1)C(=O)O", "NCCCC(=O)O", "NC(c1ccccc1)C(=O)O", "N1CCC(C1)C(=O)O",
]

SULFONYL_CHLORIDES = [
    "ClS(=O)(=O)c1ccccc1", "ClS(=O)(=O)c1ccc(C)cc1", "ClS(=O)(=O)c1ccc(F)cc1",
    "ClS(=O)(=O)c1ccc(Cl)cc1", "ClS(=O)(=O)c1ccc(OC)cc1", "ClS(=O)(=O)c1cccs1",
    "ClS(=O)(=O)C", "ClS(=O)(=O)CC", "ClS(=O)(=O)c1ccc2ccccc2c1",
    "ClS(=O)(=O)c1cccc(c1)C(F)(F)F", "ClS(=O)(=O)c1ccc(cc1)C#N",
    "ClS(=O)(=O)c1cnn(C)c1", "ClS(=O)(=O)c1ccc2OCOc2c1",
]

# --- reactions -----------------------------------------------------------------------
AMINE_PATTERN = "[NX3;H2,H1;!$(N-C=[O,N,S]);!$(N-S(=O)=O):{n}]"
RXN_AMIDE = AllChem.ReactionFromSmarts(
    "[CX3:1](=[O:2])[OX2H1]." + AMINE_PATTERN.format(n=3) + ">>[C:1](=[O:2])[N:3]"
)
RXN_SULFONAMIDE = AllChem.ReactionFromSmarts(
    "[SX4:1](=[O:2])(=[O:3])Cl." + AMINE_PATTERN.format(n=4) + ">>[S:1](=[O:2])(=[O:3])[N:4]"
)


def parse_blocks(smiles_list, name):
    mols = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            print(f"  warning: could not parse {name} block {s}")
        else:
            mols.append(m)
    print(f"  {len(mols)} {name} blocks")
    return mols


def run(rxn, a, b):
    """Run a 2-reactant reaction and return the first clean product (or None)."""
    if a is None or b is None:
        return None
    try:
        outs = rxn.RunReactants((a, b))
    except Exception:
        return None
    for o in outs:
        try:
            m = o[0]
            Chem.SanitizeMol(m)
            return Chem.MolFromSmiles(Chem.MolToSmiles(m))
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser(description="Build an offline pool of virtual drug-like molecules.")
    ap.add_argument("--n", type=int, default=60000, help="how many unique molecules (default 60000)")
    ap.add_argument("--out", default="local_pool.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-mw", type=float, default=200.0)
    ap.add_argument("--max-mw", type=float, default=650.0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print("Loading building blocks ...")
    acids = parse_blocks(ACIDS, "acid")
    amines = parse_blocks(AMINES, "amine")
    linkers = parse_blocks(LINKERS, "linker")
    sulfonyls = parse_blocks(SULFONYL_CHLORIDES, "sulfonyl chloride")
    if not (acids and amines and linkers and sulfonyls):
        raise SystemExit("A block list ended up empty - check the warnings above.")

    pick = rng.choice

    def make(kind):
        if kind == "A":  # acid + amine
            return run(RXN_AMIDE, pick(acids), pick(amines))
        if kind == "B":  # acid + linker + amine
            i1 = run(RXN_AMIDE, pick(acids), pick(linkers))
            return run(RXN_AMIDE, i1, pick(amines))
        if kind == "C":  # sulfonyl chloride + linker + amine
            i1 = run(RXN_SULFONAMIDE, pick(sulfonyls), pick(linkers))
            return run(RXN_AMIDE, i1, pick(amines))
        # kind == "D": acid + linker + linker + amine (peptide-like, the biggest)
        i1 = run(RXN_AMIDE, pick(acids), pick(linkers))
        i2 = run(RXN_AMIDE, i1, pick(linkers))
        return run(RXN_AMIDE, i2, pick(amines))

    kinds = ["A"] * 10 + ["B"] * 35 + ["C"] * 20 + ["D"] * 35  # mix of sizes
    seen, rows = set(), []
    attempts, max_attempts = 0, args.n * 30
    print(f"Building {args.n:,} unique molecules ...")
    while len(rows) < args.n and attempts < max_attempts:
        attempts += 1
        mol = make(pick(kinds))
        if mol is None:
            continue
        mw = Descriptors.MolWt(mol)
        if not (args.min_mw <= mw <= args.max_mw):
            continue
        smi = Chem.MolToSmiles(mol)
        if smi in seen:
            continue
        seen.add(smi)
        rows.append({"id": f"LOCAL_{len(rows) + 1:06d}", "smiles": smi})
        if len(rows) % 10000 == 0:
            print(f"  {len(rows):,} built")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df):,} molecules to {args.out}  (after {attempts:,} attempts)")
    if len(df) < args.n:
        print("Note: fewer than requested - the block lists can only make this many unique molecules.")
    mws = [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in df["smiles"].head(2000)]
    print(f"Molecular weight of the first 2,000: mean {sum(mws) / len(mws):.0f}, "
          f"min {min(mws):.0f}, max {max(mws):.0f}")


if __name__ == "__main__":
    main()
