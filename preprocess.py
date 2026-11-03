"""Phase 1: cleaning + anonymisation (paper Sec. 4.2.3, Listing 2)."""
import html
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"\b/?u/\w+")
_SUB = re.compile(r"\b/?r/\w+")


def clean(text: str) -> str:
    t = html.unescape(str(text))
    t = _URL.sub(" ", t)
    t = _USER.sub(" ", t)
    t = _SUB.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t
