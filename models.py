"""Phase 2-3: text branch (TF-IDF+LR or DistilBERT), metadata branch, late fusion (paper Sec. 4.3-4.4)."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .preprocess import clean

META_COLS = ["hour", "comments", "posts_per_week"]


# ------------------------------------------------------------------ branches
class TfidfText:
    kind = "tfidf"

    def __init__(self):
        self.pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0))

    def fit(self, texts, y):
        self.pipe.fit([clean(t) for t in texts], y)
        return self

    def predict_proba(self, texts):
        return self.pipe.predict_proba([clean(t) for t in texts])[:, 1]

    def top_terms(self, k=15):
        tf, lr = self.pipe.steps[0][1], self.pipe.steps[1][1]
        names, coef = np.array(tf.get_feature_names_out()), lr.coef_[0]
        order = np.argsort(coef)
        return (pd.DataFrame({"term": names[order[::-1][:k]], "weight": coef[order[::-1][:k]]}),
                pd.DataFrame({"term": names[order[:k]], "weight": coef[order[:k]]}))


class DistilBertText:
    """Inference wrapper for a model fine-tuned by mhc.distil.train_distilbert (needs torch + transformers)."""
    kind = "distilbert"

    def __init__(self, model_dir: str, max_len: int = 256, batch_size: int = 32):
        self.model_dir, self.max_len, self.batch_size = model_dir, max_len, batch_size
        self._tok = self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self._torch = torch
            self._tok = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).eval()
            self._dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._dev)

    def predict_proba(self, texts):
        self._load()
        torch, out = self._torch, []
        texts = [clean(t) for t in texts]
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self._tok(texts[i:i + self.batch_size], truncation=True, max_length=self.max_len,
                                padding=True, return_tensors="pt").to(self._dev)
                out.append(torch.softmax(self._model(**enc).logits, -1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_tok"] = d["_model"] = None
        d.pop("_torch", None)
        return d


def meta_feats(m: np.ndarray) -> np.ndarray:
    """m = [hour, comments, posts_per_week] -> [late_night, comments, posts_per_week] (paper Listing 6)."""
    m = np.asarray(m, dtype=float)
    night = ((m[:, 0] >= 22) | (m[:, 0] <= 4)).astype(float)
    return np.c_[night, m[:, 1], m[:, 2]]


class MetaModel:
    def __init__(self):
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))

    def fit(self, M, y):
        self.pipe.fit(meta_feats(M), y)
        return self

    def predict_proba(self, M):
        return self.pipe.predict_proba(meta_feats(M))[:, 1]

    def importance(self):
        coef = self.pipe.steps[1][1].coef_[0]
        return pd.DataFrame({"feature": ["late_night", "comments", "posts_per_week"], "coef (standardised)": coef})


# ------------------------------------------------------------------- metrics
def evaluate(y, p, thr=0.5) -> dict:
    y, pred = np.asarray(y), (np.asarray(p) >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    out = dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
               recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
               cm=cm.tolist(), threshold=float(thr))
    out["auc"] = float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")
    return out


def pick_threshold(y, p, target_recall=0.95):
    """Highest threshold that still keeps recall >= target (paper Listing 5 / Sec. 4.3.3)."""
    y, p = np.asarray(y), np.asarray(p)
    best = 0.05
    for t in np.linspace(0.02, 0.98, 97):
        if recall_score(y, (p >= t).astype(int), zero_division=0) >= target_recall:
            best = float(t)
    return best


# -------------------------------------------------------------------- bundle
class CrisisScreener:
    """Late fusion  p = alpha*p_text + (1-alpha)*p_meta  (paper Sec. 4.4.2)."""

    def __init__(self, text_model, meta_model=None, alpha=1.0, threshold=0.5):
        self.text_model, self.meta_model = text_model, meta_model
        self.alpha, self.threshold = alpha, threshold
        self.report: dict = {}

    def predict_proba(self, texts, meta=None):
        pt = self.text_model.predict_proba(texts)
        if self.meta_model is None or meta is None or self.alpha >= 1.0:
            return pt, pt, None
        pm = self.meta_model.predict_proba(meta)
        return self.alpha * pt + (1 - self.alpha) * pm, pt, pm

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def train_screener(df: pd.DataFrame, text_backend="tfidf", distil_dir=None, target_recall=0.95,
                   seed=42, use_meta=None) -> CrisisScreener:
    """70/15/15 stratified split. Alpha and threshold are chosen on validation only; test is touched once."""
    has_meta = all(c in df.columns for c in META_COLS)
    use_meta = has_meta if use_meta is None else (use_meta and has_meta)
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.label, random_state=seed)

    if text_backend == "distilbert":
        if not distil_dir or not os.path.isdir(distil_dir):
            raise ValueError("Provide distil_dir with a model trained by mhc.distil.train_distilbert")
        text = DistilBertText(distil_dir)
    else:
        text = TfidfText().fit(tr.text, tr.label)
    meta = MetaModel().fit(tr[META_COLS].values, tr.label) if use_meta else None

    pt_va = text.predict_proba(va.text.tolist())
    alpha = 1.0
    if meta is not None:
        pm_va = meta.predict_proba(va[META_COLS].values)
        scores = [(f1_score(va.label, ((a * pt_va + (1 - a) * pm_va) >= .5).astype(int)), a)
                  for a in np.linspace(0, 1, 11)]
        best_f1 = max(s for s, _ in scores)
        alpha = max(a for s, a in scores if s >= best_f1 - 1e-9)   # prefer more text weight on ties
        p_va = alpha * pt_va + (1 - alpha) * pm_va
    else:
        p_va = pt_va
    thr = pick_threshold(va.label, p_va, target_recall)
    scr = CrisisScreener(text, meta, alpha, thr)

    # ---- test evaluation
    meta_te = te[META_COLS].values if use_meta else None
    p_f, p_t, _ = scr.predict_proba(te.text.tolist(), meta_te)
    rep = {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "alpha": float(alpha),
           "threshold": float(thr), "text_backend": text_backend, "has_meta": bool(use_meta),
           "target_recall": target_recall, "test": {}}
    rep["test"]["Text only"] = evaluate(te.label, p_t, 0.5)
    if use_meta:
        rep["test"]["Metadata only"] = evaluate(te.label, meta.predict_proba(te[META_COLS].values), 0.5)
        rep["test"]["Late fusion (thr 0.5)"] = evaluate(te.label, p_f, 0.5)
    rep["test"][f"Final (thr {thr:.2f})"] = evaluate(te.label, p_f, thr)
    rep["curve"] = {"y": te.label.tolist(), "p": [float(x) for x in p_f]}
    if text_backend == "tfidf":
        rep["top_crisis_terms"], rep["top_noncrisis_terms"] = [t.to_dict("records") for t in text.top_terms()]
    scr.report = rep
    return scr
