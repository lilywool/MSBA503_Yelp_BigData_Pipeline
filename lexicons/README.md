# lexicons/

Empty by design — put your downloaded lexicon files here, under the exact
filenames below, and point `--lexicons-dir lexicons` at this folder. Not
committed to the repo itself (see the licensing note at the bottom).

## What goes here

| Exact filename | What it's for | Source |
|---|---|---|
| `NRC-VAD-Lexicon-v2.1.txt` | Valence/Arousal/Dominance (`Valence_avg`, `Arousal_avg`, `Dominance_avg`) | saifmohammad.com/WebPages/nrc-vad.html |
| `NRC-WCST-Lexicon-v1.0.txt` | Warmth/Competence/Sociability/Trust (`wcst_*_avg`) | saifmohammad.com/WebPages/warmth.html |
| `worrywords-v1.txt` | Worry/anxiety word flag (`worry_word_count`) | saifmohammad.com/WebPages/worrywords.html |
| `Yelp-restaurant-reviews-AFFLEX-NEGLEX-unigrams.txt` | Restaurant-review sentiment score (`yelp_sentiment_avg`) | contact: Saif Mohammad (saif.mohammad@nrc-cnrc.gc.ca) - see the `readme.txt` shipped with the lexicon download for the full citation |
| `NRC-Emotion-Intensity-Lexicon-v1.txt` | Ranked emotion intensity (`primary_emotion_lex`/`secondary_emotion_lex` + confidences, `nrc_{emotion}_lex`) | saifmohammad.com/WebPages/AffectIntensity.htm |

All five are from Dr. Saif Mohammad's lexicon collection
(saifmohammad.com/WebPages/lexicons.html) - free for research use, but each
comes with a no-redistribution license, which is why they're downloaded
locally rather than committed here.

**Not needed:** the base `NRC-Emotion-Lexicon` files (word-level/sense-level
association data). `dominant_emotion` and the `{emotion}_count` columns come
from the `nrclex` pip package instead, which bundles that data itself.

**Not used:** the AFFLEX-NEGLEX `-bigrams.txt` file that ships alongside the
unigrams file above. This pipeline's lexicon lookups are single-token
matches only, so only the unigrams file is read. This is a real, known scope
limit, not a bug - the bigrams file would let `yelp_sentiment_avg` also
catch phrase-level sentiment (e.g. two-word expressions whose combined
sentiment isn't just the sum of each word's), which the current
implementation doesn't do. Worth adding if that signal matters to you.

## Pointing the pipeline at this folder

Every entrypoint (`spark_feature_engineering.py`, `tests/parity_check.py`)
takes a single `--lexicons-dir` flag once all 5 files above are in one
folder under those exact names:

    python spark_feature_engineering.py --files ... --lexicons-dir lexicons ...

Individual `--vad-lexicon` / `--worry-lexicon` / `--wcst-lexicon` /
`--yelp-lexicon` / `--nrc-intensity-lexicon` flags still exist if your files
live somewhere else or use different names - each one overrides just that
file, `--lexicons-dir` covers the rest. See `pipeline/lexicon_cli.py`.

On Databricks/EMR: upload these 5 files to a DBFS/Volumes path or S3, and
pass `--lexicons-dir` (or the individual flags) pointing there instead.

## Verified against the real files (2026-08-27)

worry (5,959 terms), WCST (31,332 terms - after fixing a column-header bug,
see below), Yelp AFFLEX unigrams (39,274 terms), NRC intensity (5,891
terms), VAD (44,728 single-word terms after multi-word rows are filtered -
matches the count documented in `real_feature_engineering.py`). The lexicon
loaders were exercised against these real files. A prior end-to-end run
reported 68/68 matching output columns, but its comparator could miss absent
columns and null/value mismatches. Re-run the corrected strict comparator
before treating local and Spark outputs as equivalent.

The corrected comparator was rerun in the pinned WSL2 environment on
2026-09-03. Both disjoint 1,500-row Chipotle slices matched all 68 expected
columns, schemas, review-key sets, null positions, values, and row counts.

**Literal `null` term bug found and fixed:** pandas treats the English word
`null` as a missing-value marker by default. NRC-VAD, WorryWords, and NRC-WCST
each contain that literal term, so one loader crashed and two silently omitted
it. All four pandas-based lexicon readers now disable default NA-string
coercion, and a synthetic regression test proves that `null` remains scoreable
in VAD, WorryWords, WCST, and NRC Emotion Intensity inputs.

**Bug found and fixed:** `load_wcst()` assumed clean column names
(`warmth`, `competence`, `sociability`, `trust`) after lowercasing. The real
file's header carries a parenthetical abbreviation per column (`warmth (W)`,
`competence (C)`, etc.) that survives lowercasing, so the loader raised a
`KeyError` against the actual file. Fixed by also splitting each column name
on `" ("` after lowercasing.
