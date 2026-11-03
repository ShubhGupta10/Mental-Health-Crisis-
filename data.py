"""Data loading: Kaggle (real), user CSV, or synthetic demo corpus (paper Sec. 5.3.1)."""
import os
import glob
import numpy as np
import pandas as pd

# Reddit r/SuicideWatch (crisis) vs r/teenagers (non-crisis): the closest public match to paper Sec. 4.2.1
KAGGLE_DATASET = "nikhileswarkomati/suicide-watch"
CRISIS_WORDS = {"suicide", "suicidal", "crisis", "crisis_indicator", "depression", "1", "true", "yes", "positive"}


def _pick_cols(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    text_col = next((cols[c] for c in ("text", "post", "content", "body", "selftext", "clean_text") if c in cols), None)
    if text_col is None:  # fall back to longest-string column
        obj = [c for c in df.columns if df[c].dtype == object]
        text_col = max(obj, key=lambda c: df[c].astype(str).str.len().mean())
    lab_col = next((cols[c] for c in ("class", "label", "target", "category", "y") if c in cols), None)
    if lab_col is None:
        raise ValueError(f"No label column found in {list(df.columns)}")
    return text_col, lab_col


def normalise_labels(s: pd.Series) -> pd.Series:
    def f(v):
        v = str(v).strip().lower()
        if v in ("non-suicide", "non-crisis", "non_suicide", "0", "false", "no", "negative"):
            return 0
        return 1 if v in CRISIS_WORDS or "suicid" in v else 0
    return s.map(f).astype(int)


def load_csv(path: str, max_rows: int | None = None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol, lcol = _pick_cols(df)
    out = pd.DataFrame({"text": df[tcol].astype(str), "label": normalise_labels(df[lcol])})
    out = out[out.text.str.strip().str.len() > 0].drop_duplicates("text")
    if max_rows and len(out) > max_rows:
        out = out.groupby("label", group_keys=False).apply(
            lambda g: g.sample(int(max_rows * len(g) / len(out)), random_state=seed))
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_kaggle(dataset: str = KAGGLE_DATASET, max_rows: int | None = 40000) -> pd.DataFrame:
    """Download via kagglehub (needs internet + Kaggle credentials: KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json)."""
    import kagglehub
    path = kagglehub.dataset_download(dataset)
    csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {path}")
    return load_csv(max(csvs, key=os.path.getsize), max_rows=max_rows)


# ---------------------------------------------------------------- synthetic
_CRISIS = [
    "i have been feeling like a burden to everyone. my family would be better off without me. i'm planning to end it {t}.",
    "nobody cares about me, i am always alone and i cannot sleep or stop crying",
    "i can't do this anymore. there is nothing left for me and i want to die",
    "i feel completely hopeless. everything is pointless and i want the pain to stop forever",
    "i wrote goodbye letters to my family. i don't see a reason to keep living",
    "i am tired of being alive. no one would even notice if i disappeared {t}",
    "i keep thinking about ending my life. i never feel okay anymore",
]
_HARD_NEG = [
    "this exam is killing me, i have not slept but i will survive it",
    "my boss is going to kill me if i miss this deadline lol",
    "i'm dead after that workout, my legs are dying",
    "this traffic makes me want to scream, i could die of boredom in this meeting",
    "i used to want to die, but now i want to live. therapy really helped",
]
_NEUTRAL = [
    "just got back from a great hike with my dog, the weather was perfect",
    "what is everyone having for dinner tonight? thinking about making pasta",
    "i started learning guitar last week and my fingers hurt but it's fun",
    "anyone have recommendations for a good sci-fi book to read this weekend?",
    "my team won the game today, we're celebrating at the pizza place",
    "looking for tips on studying for finals, any good apps?",
    "bought a new laptop today and the screen is amazing",
]
_TIMES = ["tonight", "tomorrow", "soon", "this week"]


def make_synthetic(n: int = 1200, crisis_frac: float = 0.375, hard_frac: float = 0.15, seed: int = 0) -> pd.DataFrame:
    """Template corpus + simulated metadata. NOT real data; only validates the pipeline."""
    rng = np.random.default_rng(seed)
    rows = []
    n_c, n_h = int(n * crisis_frac), int(n * hard_frac)
    n_n = n - n_c - n_h
    def noise(t):
        extra = rng.choice(["", " idk.", " it is what it is", " honestly", " ..."])
        return (t + extra).strip()
    for _ in range(n_c):
        t = noise(rng.choice(_CRISIS).format(t=rng.choice(_TIMES)))
        night = rng.random() < 0.7
        rows.append((t, 1, int(rng.choice([22, 23, 0, 1, 2, 3, 4])) if night else int(rng.integers(5, 22)),
                     int(rng.poisson(2)), float(rng.gamma(4, 1.5))))
    for t in rng.choice(_HARD_NEG, n_h):
        rows.append((noise(t), 0, int(rng.integers(5, 24)), int(rng.poisson(6)), float(rng.gamma(2, 1.2))))
    for t in rng.choice(_NEUTRAL, n_n):
        night = rng.random() < 0.15
        rows.append((noise(t), 0, int(rng.choice([22, 23, 0, 1, 2])) if night else int(rng.integers(5, 22)),
                     int(rng.poisson(6)), float(rng.gamma(2, 1.2))))
    df = pd.DataFrame(rows, columns=["text", "label", "hour", "comments", "posts_per_week"])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
