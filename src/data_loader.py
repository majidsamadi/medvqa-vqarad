from pathlib import Path
import glob
from collections import Counter
from datasets import load_dataset, Image

def load_vqarad(data_dir: str):
    data_dir = str(Path(data_dir).resolve())
    train_files = glob.glob(str(Path(data_dir) / "train-*.parquet"))
    test_files  = glob.glob(str(Path(data_dir) / "test-*.parquet"))
    if not train_files or not test_files:
        raise FileNotFoundError("Missing train-*.parquet or test-*.parquet in data/")

    ds = load_dataset("parquet", data_files={"train": train_files[0], "test": test_files[0]})
    if "image" in ds["train"].column_names:
        ds = ds.cast_column("image", Image())
    return ds

def build_answer_vocab(train_ds, min_freq: int = 1):
    counts = Counter(train_ds["answer"])
    vocab = ["<unk>"] + [a for a, c in counts.most_common() if c >= min_freq]
    answer2id = {a: i for i, a in enumerate(vocab)}
    id2answer = {i: a for a, i in answer2id.items()}
    return answer2id, id2answer

def split_train_val(train_ds, val_ratio: float = 0.15, seed: int = 42):
    split = train_ds.train_test_split(test_size=val_ratio, seed=seed)
    return split["train"], split["test"]
