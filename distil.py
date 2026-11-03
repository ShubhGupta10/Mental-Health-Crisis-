"""DistilBERT fine-tuning (paper Sec. 4.3, Listing 4). Plain PyTorch loop -> robust across transformers versions.
Requires: pip install torch transformers. Strongly recommended on a GPU (Colab T4)."""
from __future__ import annotations
import numpy as np
from .preprocess import clean


def train_distilbert(train_df, val_df, out_dir="models/distilbert-crisis", epochs=3, lr=2e-5, batch_size=16,
                     max_len=256, crisis_weight=3.0, weight_decay=0.01, warmup_ratio=0.1, seed=42,
                     progress=print):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    from sklearn.metrics import recall_score, f1_score

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(dev)

    def enc(df):
        e = tok([clean(t) for t in df.text], truncation=True, max_length=max_len)
        return e["input_ids"], df.label.tolist()

    def batches(ids, ys, bs, shuffle):
        idx = np.random.permutation(len(ids)) if shuffle else np.arange(len(ids))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            pad = tok.pad({"input_ids": [ids[j] for j in b]}, return_tensors="pt")
            yield pad["input_ids"].to(dev), pad["attention_mask"].to(dev), torch.tensor([ys[j] for j in b]).to(dev)

    tr_ids, tr_y = enc(train_df)
    va_ids, va_y = enc(val_df)
    no_decay = ["bias", "LayerNorm.weight"]
    groups = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": weight_decay},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
               "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr)
    steps = epochs * ((len(tr_ids) + batch_size - 1) // batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(warmup_ratio * steps), steps)
    w = torch.tensor([1.0, crisis_weight], device=dev)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best_rec, best_f1 = -1, -1
    import os
    os.makedirs(out_dir, exist_ok=True)

    for ep in range(epochs):
        model.train()
        tot = 0
        for k, (ii, am, yy) in enumerate(batches(tr_ids, tr_y, batch_size, True)):
            with torch.autocast(device_type=dev, enabled=use_amp):
                logits = model(input_ids=ii, attention_mask=am).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), yy, weight=w)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            if k % 50 == 0:
                progress(f"epoch {ep+1}/{epochs} step {k} loss {loss.item():.4f}")
        model.eval()
        preds = []
        with torch.no_grad():
            for ii, am, yy in batches(va_ids, va_y, 64, False):
                preds.append(model(input_ids=ii, attention_mask=am).logits.argmax(-1).cpu().numpy())
        preds = np.concatenate(preds)
        rec, f1 = recall_score(va_y, preds), f1_score(va_y, preds)
        progress(f"epoch {ep+1}: val recall {rec:.4f} f1 {f1:.4f}")
        if (rec, f1) > (best_rec, best_f1):          # model selection on recall, as in paper
            best_rec, best_f1 = rec, f1
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
    return out_dir
