"""
Loaders for four specialized lexicons: WorryWords, NRC-WCST, Yelp
AFFLEX-NEGLEX, and the NRC Emotion Intensity Lexicon (distinct from the NRC
Emotion Lexicon / NRCLex proportions real_feature_engineering.py computes).

Each loader parses the lexicon file's actual column layout directly:

  * WorryWords: parses the real TSV columns and uses the `Term` column,
    filtered to terms with MajorityLabel >= 2 (worry/anxiety-associated per
    human annotation).

  * Yelp AFFLEX-NEGLEX: this file has FOUR tab-separated columns (term,
    score, count, a 4th field) - parsed directly rather than assumed to be
    two columns. Terms carry `_NEGFIRST` suffixes for negation-context
    variants (a real feature of this lexicon), kept as distinct keys.

  * NRC-WCST: averages the lexicon's real four-dimensional
    warmth/competence/sociability/trust scores per matched word, rather than
    just counting matches.

  * NRC Emotion Intensity: term -> {emotion: intensity}, straightforward TSV parse.
"""

from io import BytesIO
import pandas as pd


def _tabular_source(source):
    if isinstance(source, (bytes, bytearray)):
        return BytesIO(bytes(source))
    return source


def load_worry_words(path: str) -> set:
    """WorryWords lexicon: {term} for terms with MajorityLabel >= 2 (i.e. at
    least mildly worry/anxiety-associated per human annotation) - narrower
    than "any annotated word at all", to keep this a meaningful "worry word"
    flag rather than a blanket match."""
    df = pd.read_csv(
        _tabular_source(path), sep="\t", keep_default_na=False, na_values=[]
    )
    df.columns = [c.strip() for c in df.columns]
    hits = df[df["MajorityLabel"].astype(float) >= 2]
    return set(hits["Term"].astype(str).str.lower())


def load_wcst(path: str) -> dict:
    """term -> {warmth, competence, sociability, trust} (real per-dimension scores).

    The actual NRC-WCST-Lexicon-v1.0.txt header carries a parenthetical
    abbreviation per column ("warmth (W)", "competence (C)", "sociability (S)",
    "trust (T)") - lowercasing alone leaves those suffixes in place, so the
    column name has to be split on " (" as well, not just lowercased."""
    df = pd.read_csv(
        _tabular_source(path), sep="\t", keep_default_na=False, na_values=[]
    )
    df.columns = [c.strip().lower().split(" (")[0] for c in df.columns]
    return df.set_index("term")[["warmth", "competence", "sociability", "trust"]].to_dict("index")


def load_yelp_afflex(path: str) -> dict:
    """term -> sentiment score. File has 4 tab-separated columns (term, score,
    count, unused). Terms carry `_NEGFIRST` suffixes for negation-context
    variants (a real feature of this lexicon) - kept as distinct keys, matched
    only on an exact token basis like the rest of this pipeline's lexicon lookups."""
    rows = []
    if isinstance(path, (bytes, bytearray)):
        lines = bytes(path).decode("utf-8", errors="replace").splitlines()
    else:
        with open(path, encoding="utf-8", errors="replace") as stream:
            lines = list(stream)
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        term, score = parts[0], parts[1]
        try:
            rows.append((term, float(score)))
        except ValueError:
            continue
    return dict(rows)


def load_nrc_intensity(path: str) -> dict:
    """term -> {emotion: intensity}."""
    df = pd.read_csv(
        _tabular_source(path),
        sep="\t",
        names=["term", "emotion", "score"],
        keep_default_na=False,
        na_values=[],
    )
    out = {}
    for term, emotion, score in df.itertuples(index=False):
        out.setdefault(term, {})[emotion] = float(score)
    return out


def load_all(worry_path: str, wcst_path: str, yelp_path: str, nrc_intensity_path: str) -> dict:
    return {
        "worry": load_worry_words(worry_path),
        "wcst": load_wcst(wcst_path),
        "yelp": load_yelp_afflex(yelp_path),
        "nrc_intensity": load_nrc_intensity(nrc_intensity_path),
    }


def load_all_from_payloads(payloads: dict[str, bytes]) -> dict:
    """Parse executor-local byte payloads without filesystem or SparkContext APIs."""
    return load_all(
        worry_path=payloads["worry_path"],
        wcst_path=payloads["wcst_path"],
        yelp_path=payloads["yelp_path"],
        nrc_intensity_path=payloads["nrc_intensity_path"],
    )
