import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from src.data_loader import load_vqarad, build_answer_vocab, split_train_val
from src.preprocess import image_transform
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

def run_epoch(model, loader, opt, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total = 0.0
    for img, ids, mask, y in tqdm(loader, leave=False):
        img, ids, mask, y = img.to(device), ids.to(device), mask.to(device), y.to(device)
        opt.zero_grad()
        logits = model(img, ids, mask)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total += float(loss.item()) * img.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    preds, golds = [], []
    for img, ids, mask, y in loader:
        img, ids, mask = img.to(device), ids.to(device), mask.to(device)
        logits = model(img, ids, mask)
        pred = logits.argmax(dim=1).cpu().tolist()
        preds.extend(pred)
        golds.extend(y.tolist())
    return preds, golds

def main():
    data_dir = "data"
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    max_len = 32
    batch_size = 8
    epochs = 20
    lr = 1e-4

    ds = load_vqarad(data_dir)

    vocab_path = out_dir / "answer_vocab.json"
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        answer2id = {k: int(v) for k, v in vocab["answer2id"].items()}
        id2answer = {int(k): v for k, v in vocab["id2answer"].items()}
    else:
        answer2id, id2answer = build_answer_vocab(ds["train"], min_freq=1)
        (out_dir / "answer_vocab.json").write_text(json.dumps({"answer2id": answer2id, "id2answer": id2answer}, indent=2), encoding="utf-8")

    tr_ds, va_ds = split_train_val(ds["train"], val_ratio=0.15, seed=42)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    img_tf = image_transform(224)

    tr = VQATorch(tr_ds, tokenizer, answer2id, img_tf, max_len=max_len)
    va = VQATorch(va_ds, tokenizer, answer2id, img_tf, max_len=max_len)

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=0)

    model = FusionTransformer(
        num_answers=len(answer2id),
        text_model_name="distilbert-base-uncased",
        d_model=256,
        n_heads=8,
        n_layers=2,
        max_text_len=max_len,
        freeze_backbones=True,
    ).to(device)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    best = -1.0
    history = []

    for ep in range(1, epochs + 1):
        train_loss = run_epoch(model, tr_loader, opt, device)
        pred, gold = eval_model(model, va_loader, device)
        m = compute_metrics(pred, gold, id2answer)
        m["epoch"] = ep
        m["train_loss"] = train_loss
        history.append(m)

        score = m["overall_accuracy"]
        if score > best:
            best = score
            torch.save(model.state_dict(), ckpt_dir / "transformer_best.pt")

    (out_dir / "transformer_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
