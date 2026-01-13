import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from rdkit import Chem
from tqdm import tqdm


def normalize_smiles(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        return None

    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def batch_normalize_smiles(smiles_list: list[str]) -> list[str | None]:
    normalized_smiles = []
    for smiles in smiles_list:
        result = normalize_smiles(smiles)
        if result is not None:
            normalized_smiles.append(result)
    return normalized_smiles


def main():
    smiles_set = set()

    data_list = []
    with open(args.base, "r") as f:
        for line in tqdm(f.readlines(), desc="Reading base", dynamic_ncols=True):
            data_list.append(line.strip())
    with open(os.path.join(args.molparser, "pretrain_synthetic_7M", "data.jsonl"), "r") as f:
        for line in tqdm(f.readlines(), desc="Reading MolParser pretrain", dynamic_ncols=True):
            smiles = json.loads(line)["smiles"]
            smiles, ext = smiles.split("<sep>")
            if ext == "":  # we only need normal SMILES here
                data_list.append(smiles.strip())
    with open(os.path.join(args.molparser, "sft_real", "data.jsonl"), "r") as f:
        for line in tqdm(f.readlines(), desc="Reading MolParser sft", dynamic_ncols=True):
            smiles = json.loads(line)["smiles"]
            smiles, ext = smiles.split("<sep>")
            if ext == "":  # we only need normal SMILES here
                data_list.append(smiles.strip())

    splited = []  # extract SMILES separated by "."
    for smi in data_list:
        if "." in smi:
            splited.extend(smi.split("."))
    data_list = data_list + splited

    print(f"Total {len(data_list)} SMILES read.")

    batch_jobs = []
    for i in range(0, len(data_list), args.batch_size):
        batch_jobs.append(data_list[i: i + args.batch_size])

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(batch_normalize_smiles, batch) for batch in batch_jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Normalizing SMILES", dynamic_ncols=True):
            normalized_batch = future.result()
            smiles_set.update(normalized_batch)
    print(f"Total {len(smiles_set)} unique normalized SMILES obtained.")

    with open(args.output, "w") as f:
        for smiles in tqdm(smiles_set, desc="Writing output", dynamic_ncols=True):
            f.write(smiles + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=os.cpu_count())
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--base", type=str, default="smiles_large.txt")
    parser.add_argument("--molparser", type=str, default="/home/public_space/zhangxiaohong/public_user/MolParser-7M-Extracted/")
    parser.add_argument("--output", type=str, default="smiles_molparser.txt")
    args = parser.parse_args()
    main()
