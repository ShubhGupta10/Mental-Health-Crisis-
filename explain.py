"""TextRank explainer (paper Sec. 3.4, 5.6) with the fixes listed in Sec. 6.2:
contraction handling and retention of urgency words ("tonight")."""
import re
import networkx as nx
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP = set(ENGLISH_STOP_WORDS) | {"just", "really", "like", "maybe", "today", "im", "ive", "dont", "cant", "wont"}
# absolutist words (Al-Mosaiwi & Johnstone) + temporal urgency markers
KEEP_ABS = {"nothing", "nobody", "everyone", "always", "never", "anymore", "forever",
            "completely", "totally", "entire", "everything", "all"}
KEEP_URGENT = {"tonight", "now", "soon", "tomorrow"}
KEEP = KEEP_ABS | KEEP_URGENT


def _tokens(text: str):
    text = re.sub(r"(\w)'(\w)", r"\1\2", text.lower())   # i'm -> im (contraction fix)
    return re.findall(r"[a-z]+", text)


def textrank(text: str, window: int = 3, top_n: int = 5):
    toks = _tokens(text)
    cand = [w for w in toks if (w not in STOP or w in KEEP) and len(w) > 2]
    G = nx.Graph()
    for i, w in enumerate(cand):
        for j in range(i + 1, min(i + window, len(cand))):
            if w != cand[j]:
                G.add_edge(w, cand[j])
    if G.number_of_nodes() == 0:
        return []
    pr = nx.pagerank(G, alpha=0.85)
    # boost clinically meaningful words slightly so they are not lost to graph position
    pr = {w: s * (1.25 if w in KEEP else 1.0) for w, s in pr.items()}
    return sorted(pr.items(), key=lambda x: -x[1])[:top_n]


def highlight_flags(text: str):
    """Return which absolutist / urgency markers occur in the post."""
    toks = set(_tokens(text))
    return {"absolutist": sorted(toks & KEEP_ABS), "urgency": sorted(toks & KEEP_URGENT)}
