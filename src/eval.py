import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.data_loader import load_vqarad
from src.preprocess import image_transform
from src.models.cnn_bilstm import CNNBiLSTM
from src.models.transformer_fusion import FusionTransformer
from src.metrics import compute_metrics

class VQATorch(Dataset):
    def __init__(self, hf_ds, tokenizer, answer2id, img_tf, max_len=32):
        self.ds = hf_ds
        self.tok = tokenizer
        self.a2i = answer2id
        self.img_tf = img_tf
        self.max_len = max_len

    def __len__(self): return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        img = ex["image"].convert("RGB")
        img = self.img_tf(img)
        t = self.tok(ex["question"], max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        ans = ex["answer"]
        y = self.a2i.get(ans, self.a2i["<unk>"])
        return img, t["input_ids"].squeeze(0), t["attention_mask"].squeeze(0), torch.tensor(y, dtype=torch.long)

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    preds, golds = [], []
    for img, ids, mask, y in loader:
        img, ids, mask = img.to(device), ids.to(device), mask.to(device)
        logits = model(img, ids, mask)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        golds.extend(y.tolist())
    return preds, golds

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "transformer"], required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--max_len", type=int, default=32)
    args = ap.parse_args()

    out_dir = Path("results")
    ckpt_dir = out_dir / "checkpoints"
    vocab = json.loads((out_dir / "answer_vocab.json").read_text(encoding="utf-8"))
    answer2id = {k: int(v) for k, v in vocab["answer2id"].items()}
    id2answer = {int(k): v for k, v in vocab["id2answer"].items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = load_vqarad(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    img_tf = image_transform(224)

    test_ds = VQATorch(ds["test"], tokenizer, answer2id, img_tf, max_len=args.max_len)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    if args.model == "baseline":
        model = CNNBiLSTM(vocab_size=tokenizer.vocab_size, num_answers=len(answer2id), freeze_cnn=True)
        model.load_state_dict(torch.load(ckpt_dir / "baseline_best.pt", map_location="cpu"))
    else:
        model = FusionTransformer(num_answers=len(answer2id), max_text_len=args.max_len, freeze_backbones=True)
        model.load_state_dict(torch.load(ckpt_dir / "transformer_best.pt", map_location="cpu"))

    model = model.to(device)

    pred, gold = eval_model(model, test_loader, device)
    metrics = compute_metrics(pred, gold, id2answer)

    (out_dir / f"{args.model}_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
