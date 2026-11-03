"""CLI:  python train.py --source synthetic | kaggle | csv --csv path  [--backend tfidf|distilbert]"""
import argparse, json, os
from mhc import data
from mhc.models import train_screener

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["synthetic", "kaggle", "csv"], default="synthetic")
ap.add_argument("--csv")
ap.add_argument("--dataset", default=data.KAGGLE_DATASET)
ap.add_argument("--max-rows", type=int, default=40000)
ap.add_argument("--backend", choices=["tfidf", "distilbert"], default="tfidf")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--target-recall", type=float, default=0.95)
ap.add_argument("--out", default="models/screener.joblib")
a = ap.parse_args()

if a.source == "synthetic":
    df = data.make_synthetic()
elif a.source == "kaggle":
    df = data.load_kaggle(a.dataset, a.max_rows)
else:
    df = data.load_csv(a.csv, a.max_rows)
print(f"{len(df)} rows, crisis share {df.label.mean():.2%}")

distil_dir = None
if a.backend == "distilbert":
    from sklearn.model_selection import train_test_split
    from mhc.distil import train_distilbert
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=42)
    va, _ = train_test_split(tmp, test_size=0.50, stratify=tmp.label, random_state=42)
    distil_dir = train_distilbert(tr, va, "models/distilbert-crisis", epochs=a.epochs, max_len=a.max_len)

scr = train_screener(df, a.backend, distil_dir, a.target_recall)
scr.report["source"] = a.source
scr.save(a.out)
for k, v in scr.report["test"].items():
    print(f"{k:26s} acc {v['accuracy']:.3f} P {v['precision']:.3f} R {v['recall']:.3f} F1 {v['f1']:.3f} cm {v['cm']}")
print("saved", a.out)
