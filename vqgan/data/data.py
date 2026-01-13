import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm


def split_dataset(input_file, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05, shuffle=True, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "The sum of the proportions must equal 1"

    with open(input_file, "r") as f:
        file_paths = [line.strip() for line in f.readlines() if line.strip()]

    if shuffle:
        random.seed(seed)
        random.shuffle(file_paths)

    total = len(file_paths)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train_files = file_paths[:train_size]
    val_files = file_paths[train_size: train_size + val_size]
    test_files = file_paths[train_size + val_size:]

    os.makedirs("smiles", exist_ok=True)
    with open("smiles/train.txt", "w") as f:
        f.write("\n".join(train_files))
    with open("smiles/val.txt", "w") as f:
        f.write("\n".join(val_files))
    with open("smiles/test.txt", "w") as f:
        f.write("\n".join(test_files))

    print(f"done:\n"
          f"- train: {len(train_files)}, ({100 * train_ratio:.0f}%)\n"
          f"- val: {len(val_files)}, ({100 * val_ratio:.0f}%)\n"
          f"- test: {len(test_files)}, ({100 * test_ratio:.0f}%)\n")


def mol_to_image(smiles: str, save: str) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        img = Draw.MolToImage(molecule, size=(256, 256))
        img.save(save)


def batch_mol_to_image(batch_data, batch_save) -> None:
    for smiles, save in zip(batch_data, batch_save):
        mol_to_image(smiles, save)


def data_gen():
    files = ["smiles/train.txt", "smiles/val.txt", "smiles/test.txt"]
    for file in files:
        with open(file, "r") as f:
            data = [line.strip() for line in f.readlines() if line.strip()]
        _, file_name = os.path.split(file)
        file_name = file_name.split(".")[0]
        os.makedirs(os.path.join("image", file_name), exist_ok=True)

        batch_jobs = []
        for batch_start in range(0, len(data), args.batch_size):
            batch_data = data[batch_start: batch_start + args.batch_size]
            batch_idx = list(range(batch_start, batch_start + len(batch_data)))
            batch_save = list(map(lambda idx: os.path.join("image", file_name, f"{idx}.png"), batch_idx))
            batch_jobs.append((batch_data, batch_save))

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(batch_mol_to_image, batch_data, batch_save) for batch_data, batch_save in batch_jobs]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {file_name}", dynamic_ncols=True):
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--input", type=str, default="smiles.txt")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    split_dataset(args.input)
    data_gen()
