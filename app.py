"""Streamlit front end for the explainable crisis-indicator screening framework."""
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve

from mhc import data
from mhc.explain import textrank, highlight_flags
from mhc.models import CrisisScreener, train_screener, META_COLS
from mhc.preprocess import clean

MODEL_PATH = "models/screener.joblib"
DEMO_PATH = "models/demo.joblib"

st.set_page_config(page_title="Crisis Indicator Screening", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.flag-high {background:#fdecea;border-left:6px solid #d93025;padding:.8rem 1rem;border-radius:6px;}
.flag-low  {background:#e8f5e9;border-left:6px solid #2e7d32;padding:.8rem 1rem;border-radius:6px;}
.chip {display:inline-block;background:#eef2ff;border:1px solid #c7d2fe;border-radius:14px;padding:2px 10px;margin:2px;font-size:.9rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_model(path):
    return CrisisScreener.load(path) if os.path.exists(path) else None


def active_model():
    for p in (MODEL_PATH, DEMO_PATH):
        m = load_model(p)
        if m is not None:
            return m, p
    return None, None


st.title("🧠 Explainable Crisis-Indicator Screening")
st.caption("Term-paper implementation: cleaning → text model (TF-IDF or DistilBERT) → metadata late fusion → TextRank explanation. "
           "**Assistive triage for a human moderator — not a diagnostic tool.**")

with st.sidebar:
    st.header("Model")
    model, mpath = active_model()
    if model is None:
        st.warning("No trained model yet. Open the **Train** tab (synthetic demo works offline).")
    else:
        r = model.report
        st.success(f"Loaded `{os.path.basename(mpath)}`")
        st.write(f"Text branch: **{r.get('text_backend','?')}**  \nData: **{r.get('source','?')}**")
        st.write(f"Fusion α = **{model.alpha:.1f}**  \nDecision threshold = **{model.threshold:.2f}**")
        if r.get("source") == "synthetic":
            st.error("Trained on SYNTHETIC template data. Scores demonstrate the pipeline only, not real-world accuracy.")
    st.divider()
    st.info("If you or someone you know is in crisis, contact local emergency services or a crisis line "
            "(India: Tele-MANAS 14416; US: call/text 988).")

tab_an, tab_batch, tab_perf, tab_train, tab_about = st.tabs(
    ["🔎 Analyse a post", "📄 Batch CSV", "📊 Model performance", "🛠 Train", "⚖ Ethics & limits"])

# ------------------------------------------------------------------ analyse
with tab_an:
    if model is None:
        st.info("Train a model first.")
    else:
        examples = {
            "— choose an example —": "",
            "Genuine distress": "I have been feeling like a burden to everyone. My family would be better off without me. I'm planning to end it tonight.",
            "Figurative language": "this exam is killing me, i have not slept but i will survive it",
            "Absolutist / isolation": "nobody cares about me, i am always alone and i cannot sleep or stop crying",
            "Recovery narrative": "I used to want to die, but now I want to live. Therapy really helped.",
            "Neutral": "just got back from a great hike with my dog, the weather was perfect",
        }
        c1, c2 = st.columns([3, 2])
        with c1:
            pick = st.selectbox("Example", list(examples))
            text = st.text_area("Post text", value=examples[pick], height=170, key=f"t_{pick}")
        with c2:
            use_meta = st.checkbox("Use interaction metadata (late fusion)", value=model.meta_model is not None and model.alpha < 1,
                                   disabled=model.meta_model is None or model.alpha >= 1,
                                   help="Only available when the model was trained with metadata and fusion gave it weight (α<1).")
            hour = st.slider("Posting hour (0-23)", 0, 23, 14, disabled=not use_meta)
            comments = st.number_input("Comments received", 0, 500, 3, disabled=not use_meta)
            ppw = st.number_input("User posts / week", 0.0, 100.0, 4.0, disabled=not use_meta)
            thr = st.slider("Decision threshold", 0.01, 0.99, float(model.threshold), 0.01,
                            help="Lower = higher recall, more false alarms (paper prioritises recall).")
        if st.button("Analyse", type="primary") and text.strip():
            meta = np.array([[hour, comments, ppw]]) if use_meta else None
            pf, pt, pm = model.predict_proba([text], meta)
            p = float(pf[0])
            if p >= thr:
                st.markdown(f"<div class='flag-high'><b>⚠ Flag for human review</b> — fused crisis probability "
                            f"<b>{p:.3f}</b> ≥ threshold {thr:.2f}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='flag-low'><b>Not flagged</b> — fused crisis probability "
                            f"<b>{p:.3f}</b> &lt; threshold {thr:.2f}</div>", unsafe_allow_html=True)
            g1, g2 = st.columns(2)
            with g1:
                fig = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#d93025" if p >= thr else "#2e7d32"},
                           "threshold": {"line": {"color": "black", "width": 3}, "value": thr * 100}}))
                fig.update_layout(height=250, margin=dict(t=20, b=10))
                st.plotly_chart(fig, width="stretch")
                st.write(f"Text probability: **{float(pt[0]):.3f}**" + (f" · Metadata probability: **{float(pm[0]):.3f}**" if pm is not None else ""))
            with g2:
                st.subheader("Why? (TextRank)")
                kw = textrank(clean(text))
                st.markdown(" ".join(f"<span class='chip'>{w}</span>" for w, _ in kw) or "_no keywords_", unsafe_allow_html=True)
                if kw:
                    kdf = pd.DataFrame(kw, columns=["keyword", "score"]).sort_values("score")
                    st.plotly_chart(px.bar(kdf, x="score", y="keyword", orientation="h", height=220), width="stretch")
                fl = highlight_flags(text)
                if fl["absolutist"]:
                    st.write("Absolutist language:", ", ".join(fl["absolutist"]))
                if fl["urgency"]:
                    st.write("Urgency markers:", ", ".join(fl["urgency"]))
            st.caption("Keywords explain *which words are central to the post*, not the classifier's internal reasoning. "
                       "A human must review every flag.")

# -------------------------------------------------------------------- batch
with tab_batch:
    if model is None:
        st.info("Train a model first.")
    else:
        up = st.file_uploader("CSV with a text column (optional: hour, comments, posts_per_week)", type="csv")
        if up:
            d = pd.read_csv(up)
            tcol = st.selectbox("Text column", d.columns, index=list(d.columns).index("text") if "text" in d.columns else 0)
            meta = d[META_COLS].values if all(c in d.columns for c in META_COLS) and model.meta_model is not None else None
            pf, pt, _ = model.predict_proba(d[tcol].astype(str).tolist(), meta)
            d["p_crisis"] = pf
            d["flagged"] = d.p_crisis >= model.threshold
            st.write(f"**{int(d.flagged.sum())}** of {len(d)} posts flagged at threshold {model.threshold:.2f}")
            st.dataframe(d.sort_values("p_crisis", ascending=False), width="stretch", height=400)
            st.download_button("Download scored CSV", d.to_csv(index=False).encode(), "scored.csv", "text/csv")

# --------------------------------------------------------------------- perf
with tab_perf:
    if model is None:
        st.info("Train a model first.")
    else:
        r = model.report
        st.write(f"Split: {r['n_train']} train / {r['n_val']} validation / {r['n_test']} test (stratified). "
                 f"α and threshold chosen on validation; target recall {r['target_recall']:.0%}.")
        rows = [{"Model": k, "Accuracy": v["accuracy"], "Precision": v["precision"], "Recall": v["recall"],
                 "F1": v["f1"], "AUC": v["auc"], "TN/FP/FN/TP": f"{v['cm'][0][0]}/{v['cm'][0][1]}/{v['cm'][1][0]}/{v['cm'][1][1]}"}
                for k, v in r["test"].items()]
        st.dataframe(pd.DataFrame(rows).style.format({c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1", "AUC"]}),
                     width="stretch", hide_index=True)
        m = pd.DataFrame(rows).melt(id_vars="Model", value_vars=["Precision", "Recall", "F1"])
        st.plotly_chart(px.bar(m, x="Model", y="value", color="variable", barmode="group", range_y=[0, 1.05]), width="stretch")
        y, p = np.array(r["curve"]["y"]), np.array(r["curve"]["p"])
        pr, rc, th = precision_recall_curve(y, p)
        fig = go.Figure(go.Scatter(x=rc, y=pr, mode="lines"))
        fig.update_layout(title="Precision–Recall (test set, final model)", xaxis_title="Recall", yaxis_title="Precision", height=350)
        st.plotly_chart(fig, width="stretch")
        cm = np.array(r["test"][list(r["test"])[-1]]["cm"])
        st.plotly_chart(px.imshow(cm, text_auto=True, x=["pred non-crisis", "pred crisis"], y=["true non-crisis", "true crisis"],
                                  title="Confusion matrix (final model)", height=350), width="stretch")
        if "top_crisis_terms" in r:
            a, b = st.columns(2)
            a.subheader("Top crisis terms"); a.dataframe(pd.DataFrame(r["top_crisis_terms"]), hide_index=True)
            b.subheader("Top non-crisis terms"); b.dataframe(pd.DataFrame(r["top_noncrisis_terms"]), hide_index=True)
        if r.get("source") == "synthetic":
            st.warning("Perfect scores here are expected: template-generated classes share almost no vocabulary (paper §6.1.1).")

# -------------------------------------------------------------------- train
with tab_train:
    st.subheader("Train a screener")
    src = st.radio("Data source", ["Synthetic demo (offline)", "Kaggle (download)", "Upload CSV"], horizontal=True)
    backend = st.radio("Text branch", ["tfidf", "distilbert"], horizontal=True,
                       help="DistilBERT needs `pip install torch transformers`, internet for the base model, and ideally a GPU.")
    tr_recall = st.slider("Target recall on validation set", 0.80, 0.99, 0.95, 0.01)
    df = None
    if src.startswith("Synthetic"):
        n = st.slider("Posts", 400, 5000, 1200, 100)
    elif src.startswith("Kaggle"):
        slug = st.text_input("Kaggle dataset slug", data.KAGGLE_DATASET)
        mr = st.number_input("Max rows (stratified sample)", 2000, 250000, 40000, 2000)
        st.caption("Needs Kaggle credentials (`KAGGLE_USERNAME`/`KAGGLE_KEY` env vars or `~/.kaggle/kaggle.json`). "
                   "Default = Reddit r/SuicideWatch vs. r/teenagers. Kaggle data has no metadata, so fusion is disabled.")
    else:
        f = st.file_uploader("Labelled CSV (text + label/class column)", type="csv", key="trainup")
    if backend == "distilbert":
        ep = st.slider("Epochs", 1, 5, 3)
        ml = st.select_slider("Max tokens", [64, 128, 256, 512], 256)
    if st.button("Train", type="primary"):
        try:
            with st.status("Working…", expanded=True) as s:
                if src.startswith("Synthetic"):
                    df, source = data.make_synthetic(n), "synthetic"
                elif src.startswith("Kaggle"):
                    st.write("Downloading from Kaggle…"); df, source = data.load_kaggle(slug, int(mr)), "kaggle"
                else:
                    if f is None: raise ValueError("Upload a CSV first")
                    os.makedirs("models", exist_ok=True)
                    with open("models/_upload.csv", "wb") as fh: fh.write(f.getbuffer())
                    df, source = data.load_csv("models/_upload.csv"), "upload"
                st.write(f"{len(df)} rows · crisis share {df.label.mean():.1%}")
                dd = None
                if backend == "distilbert":
                    from sklearn.model_selection import train_test_split
                    from mhc.distil import train_distilbert
                    a_, b_ = train_test_split(df, test_size=.3, stratify=df.label, random_state=42)
                    v_, _ = train_test_split(b_, test_size=.5, stratify=b_.label, random_state=42)
                    dd = train_distilbert(a_, v_, "models/distilbert-crisis", epochs=ep, max_len=ml, progress=st.write)
                st.write("Fitting fusion + threshold…")
                scr = train_screener(df, backend, dd, tr_recall)
                scr.report["source"] = source
                scr.save(MODEL_PATH)
                load_model.clear()
                s.update(label="Done", state="complete")
            st.success("Model saved — reload the page or switch tabs to use it.")
            st.rerun()
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")

# -------------------------------------------------------------------- about
with tab_about:
    st.markdown("""
**Detection ≠ diagnosis.** This tool surfaces linguistic patterns associated with crisis for a trained human moderator.
It must never act autonomously, auto-message users, or report to authorities.

- **Privacy:** Reddit posts are public but authors didn't consent to research. URLs and user/subreddit mentions are stripped; do not republish raw posts.
- **Bias:** Reddit skews young, male and Western; indirect or culturally specific expressions of distress may be missed (false negatives).
- **Language drift:** slang ("unalive", "kms") evolves; models need monitoring and retraining.
- **Truncation:** long posts are cut at the model's token limit.
- **Explanations:** TextRank shows central words, not the model's internal reasoning.
- **Synthetic data:** results on the demo corpus validate code only.
""")
