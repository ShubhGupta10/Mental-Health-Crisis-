# Explainable Crisis-Indicator Screening (Streamlit)

Transformer-based NLP for mental-health crisis detection:
cleaning -> text model (TF-IDF+LR baseline **or** fine-tuned DistilBERT) -> metadata late fusion -> TextRank explanation.
Assistive triage for a human moderator, **not** a diagnostic tool.

## Run
    pip install -r requirements.txt
    python train.py --source synthetic --out models/demo.joblib   # offline pipeline check
    streamlit run app.py

## Real data (Kaggle)
Default dataset: `nikhileswarkomati/suicide-watch` (Reddit r/SuicideWatch vs r/teenagers, ~232k posts).
Set credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY`, or `~/.kaggle/kaggle.json`), then:

    python train.py --source kaggle --max-rows 40000 --out models/screener.joblib
    # DistilBERT (GPU strongly recommended; pip install torch transformers):
    python train.py --source kaggle --backend distilbert --epochs 3 --max-len 256

Or use the **Train** tab in the app. Any CSV with a text column and a label/class column also works (`--source csv --csv file.csv`).
The app loads `models/screener.joblib` if present, otherwise `models/demo.joblib`.

## Notes
- Kaggle data has no metadata, so late fusion is disabled (alpha = 1) there; metadata only exists in the synthetic corpus.
- Threshold is the highest value keeping validation recall >= target (default 95%), per paper Sec. 4.3.3.
- Scores on the synthetic corpus validate code only, not real-world accuracy.
