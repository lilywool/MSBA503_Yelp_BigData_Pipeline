"""Platform-neutral Spark transforms for Yelp Bronze, Gold, and dashboards."""

from __future__ import annotations

from functools import reduce

import pyspark.sql.functions as F


GOLD_LEVELS = (
    "brand-sample", "sample", "business", "user", "date", "industry",
    "sentiment", "emotion",
)
DATE_GRANULARITIES = ("day", "week", "month", "quarter", "year")
EMOTION_COLUMNS = ("dominant_emotion", "primary_emotion_lex", "hf_emotion_label")
SENTIMENT_COLUMNS = (
    "vader_sentiment_score", "hf_computed_sentiment", "yelp_sentiment_avg",
    "Valence_avg",
)
AVERAGE_METRICS = (
    "stars", "weighted_star", "vader_sentiment_score", "Valence_avg",
    "Arousal_avg", "Dominance_avg", "word_count", "subjectivity_score",
    "anger_int_avg", "fear_int_avg", "joy_int_avg", "sadness_int_avg",
)


def require_columns(frame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def require_nonempty(frame, label: str) -> int:
    count = frame.count()
    if count == 0:
        raise ValueError(f"{label} is empty")
    return count


def sample_reviews(frame, sample_size: int | None, seed: int):
    """Return a deterministic review subset, or the complete review frame."""
    if sample_size is None:
        return frame
    if sample_size < 1:
        raise ValueError("--silver-sample-size must be at least 1")
    require_columns(frame, {"review_id"}, "Deterministic Silver sample")
    return (
        frame.withColumn(
            "_sample_hash",
            F.xxhash64(F.lit(int(seed)), F.col("review_id").cast("string")),
        )
        .orderBy("_sample_hash", "review_id")
        .limit(sample_size)
        .drop("_sample_hash")
    )


def scope_related_dataset(frame, dataset_name: str, selected_reviews, *, use_broadcasts=True):
    """Keep dimension/fact rows related to the selected review population."""
    if dataset_name == "review":
        return selected_reviews
    if dataset_name == "user":
        keys = selected_reviews.select("user_id").where(F.col("user_id").isNotNull()).distinct()
        return frame.join(F.broadcast(keys) if use_broadcasts else keys, "user_id", "left_semi")
    if dataset_name in {"business", "checkin", "tip"}:
        keys = selected_reviews.select("business_id").where(
            F.col("business_id").isNotNull()
        ).distinct()
        return frame.join(
            F.broadcast(keys) if use_broadcasts else keys,
            "business_id",
            "left_semi",
        )
    raise ValueError(f"Unsupported Yelp dataset: {dataset_name}")


def _contains_any(column, values: list[str]):
    normalized = F.lower(F.regexp_replace(F.coalesce(column, F.lit("")), r"\s+", " "))
    predicates = [normalized.contains(" ".join(value.lower().split())) for value in values]
    return reduce(lambda left, right: left | right, predicates)


def _sentiment_label(column_name: str):
    score = F.col(column_name)
    if column_name == "Valence_avg":
        return F.when(score >= 0.55, "positive").when(
            score <= 0.45, "negative"
        ).otherwise("neutral")
    return F.when(score >= 0.05, "positive").when(
        score <= -0.05, "negative"
    ).otherwise("neutral")


def apply_gold_filters(
    silver,
    *,
    business_ids=None,
    business_names=None,
    user_ids=None,
    industries=None,
    cities=None,
    states=None,
    start_date=None,
    end_date=None,
    sentiment_labels=None,
    emotion_labels=None,
    emotion_column="dominant_emotion",
    sentiment_column="vader_sentiment_score",
    min_sentiment=None,
    max_sentiment=None,
    min_stars=None,
    max_stars=None,
):
    frame = silver
    if business_ids:
        require_columns(frame, {"business_id"}, "Business ID filtering")
        frame = frame.where(F.col("business_id").isin(business_ids))
    if business_names:
        require_columns(frame, {"name"}, "Business name filtering")
        frame = frame.where(_contains_any(F.col("name"), business_names))
    if user_ids:
        require_columns(frame, {"user_id"}, "User filtering")
        frame = frame.where(F.col("user_id").isin(user_ids))
    if industries:
        predicates = [
            _contains_any(F.col(column), industries)
            for column in ("primary_industry", "secondary_industry", "tertiary_industry")
            if column in frame.columns
        ]
        if not predicates:
            raise ValueError("Industry filtering requires at least one industry column")
        frame = frame.where(reduce(lambda left, right: left | right, predicates))
    if cities:
        require_columns(frame, {"city"}, "City filtering")
        frame = frame.where(F.lower(F.col("city")).isin([value.lower() for value in cities]))
    if states:
        require_columns(frame, {"state"}, "State filtering")
        frame = frame.where(F.upper(F.col("state")).isin([value.upper() for value in states]))
    if start_date:
        require_columns(frame, {"review_date"}, "Start-date filtering")
        frame = frame.where(F.to_date("review_date") >= F.to_date(F.lit(start_date)))
    if end_date:
        require_columns(frame, {"review_date"}, "End-date filtering")
        frame = frame.where(F.to_date("review_date") <= F.to_date(F.lit(end_date)))
    if sentiment_labels:
        require_columns(frame, {sentiment_column}, "Sentiment filtering")
        frame = frame.where(_sentiment_label(sentiment_column).isin(sentiment_labels))
    if emotion_labels:
        require_columns(frame, {emotion_column}, "Emotion filtering")
        frame = frame.where(
            F.lower(F.col(emotion_column)).isin([value.lower() for value in emotion_labels])
        )
    if min_sentiment is not None:
        require_columns(frame, {sentiment_column}, "Sentiment score filtering")
        frame = frame.where(F.col(sentiment_column) >= min_sentiment)
    if max_sentiment is not None:
        require_columns(frame, {sentiment_column}, "Sentiment score filtering")
        frame = frame.where(F.col(sentiment_column) <= max_sentiment)
    if min_stars is not None:
        require_columns(frame, {"stars"}, "Star rating filtering")
        frame = frame.where(F.col("stars") >= min_stars)
    if max_stars is not None:
        require_columns(frame, {"stars"}, "Star rating filtering")
        frame = frame.where(F.col("stars") <= max_stars)
    return frame


def deterministic_sample(frame, sample_size: int, seed: int):
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    require_columns(frame, {"review_id"}, "Deterministic Gold sample")
    return (
        frame.withColumn(
            "_sample_hash",
            F.xxhash64(F.lit(int(seed)), F.col("review_id").cast("string")),
        )
        .orderBy("_sample_hash", "review_id")
        .limit(sample_size)
        .drop("_sample_hash")
    )


def effective_sample_size(requested_size: int | None, eligible_rows: int) -> int | None:
    if requested_size is None:
        return None
    if requested_size < 1:
        raise ValueError("--gold-sample-size must be at least 1")
    if eligible_rows < 0:
        raise ValueError("eligible_rows cannot be negative")
    return min(requested_size, eligible_rows)


def build_brand_sample(frame, brands: list[str], sample_size: int, seed: int):
    require_columns(frame, {"review_id", "name"}, "Brand Gold sample")
    pieces = []
    for brand in brands:
        pieces.append(
            deterministic_sample(frame.where(_contains_any(F.col("name"), [brand])), sample_size, seed)
            .withColumn("gold_variant_level", F.lit("brand-sample"))
            .withColumn("gold_variant_value", F.lit(brand))
            .withColumn("gold_sample_seed", F.lit(seed))
        )
    return reduce(lambda left, right: left.unionByName(right), pieces)


def _aggregate_metrics(frame, group_columns: list[str]):
    expressions = [
        F.count("review_id").alias("review_count"),
        F.avg(F.when(F.col("stars") <= 2, 1.0).otherwise(0.0)).alias("low_star_rate"),
        F.countDistinct("business_id").alias("unique_business_count"),
        F.countDistinct("user_id").alias("unique_user_count"),
    ]
    for column in AVERAGE_METRICS:
        if column in frame.columns:
            expressions.append(F.avg(F.col(column)).alias(f"avg_{column}"))
    return frame.groupBy(*group_columns).agg(*expressions)


def build_gold_variant(
    frame,
    *,
    gold_level: str,
    sample_size: int | None,
    sample_seed: int,
    brands: list[str] | None,
    date_granularity: str,
    emotion_column: str,
    sentiment_column: str,
):
    if gold_level == "brand-sample":
        if not brands:
            raise ValueError("brand-sample requires at least one --brand")
        if sample_size is None:
            raise ValueError("brand-sample requires --gold-sample-size")
        return build_brand_sample(frame, brands, sample_size, sample_seed)
    if gold_level == "sample":
        if sample_size is None:
            raise ValueError("sample requires --gold-sample-size")
        return (
            deterministic_sample(frame, sample_size, sample_seed)
            .withColumn("gold_variant_level", F.lit("sample"))
            .withColumn("gold_variant_value", F.lit("filtered_reviews"))
            .withColumn("gold_sample_seed", F.lit(sample_seed))
        )
    if gold_level == "business":
        require_columns(frame, {"business_id"}, "Business Gold")
        dimensions = frame.groupBy("business_id").agg(
            F.first("name", ignorenulls=True).alias("business_name"),
            F.first("city", ignorenulls=True).alias("city"),
            F.first("state", ignorenulls=True).alias("state"),
            F.first("primary_industry", ignorenulls=True).alias("primary_industry"),
        )
        result = _aggregate_metrics(frame, ["business_id"]).join(dimensions, "business_id", "left")
        return result.withColumn("gold_variant_level", F.lit("business")).withColumn(
            "gold_variant_value", F.col("business_id")
        )
    if gold_level == "user":
        require_columns(frame, {"user_id"}, "User Gold")
        dimensions = frame.groupBy("user_id").agg(
            F.first("user_name", ignorenulls=True).alias("user_name")
        )
        result = _aggregate_metrics(frame, ["user_id"]).join(dimensions, "user_id", "left")
        return result.withColumn("gold_variant_level", F.lit("user")).withColumn(
            "gold_variant_value", F.col("user_id")
        )
    if gold_level == "date":
        require_columns(frame, {"review_date"}, "Date Gold")
        dated = frame.withColumn(
            "date_period", F.date_trunc(date_granularity, F.to_timestamp("review_date"))
        ).where(F.col("date_period").isNotNull())
        return _aggregate_metrics(dated, ["date_period"]).withColumn(
            "gold_variant_level", F.lit("date")
        ).withColumn("gold_variant_value", F.col("date_period").cast("string"))
    if gold_level == "industry":
        require_columns(frame, {"primary_industry"}, "Industry Gold")
        return _aggregate_metrics(
            frame.where(F.col("primary_industry").isNotNull()), ["primary_industry"]
        ).withColumn("gold_variant_level", F.lit("industry")).withColumn(
            "gold_variant_value", F.col("primary_industry")
        )
    if gold_level == "sentiment":
        require_columns(frame, {sentiment_column}, "Sentiment Gold")
        labeled = frame.withColumn("sentiment_label", _sentiment_label(sentiment_column))
        return _aggregate_metrics(labeled, ["sentiment_label"]).withColumn(
            "gold_variant_level", F.lit("sentiment")
        ).withColumn("gold_variant_value", F.col("sentiment_label"))
    require_columns(frame, {emotion_column}, "Emotion Gold")
    labeled = frame.withColumn("emotion_label", F.lower(F.col(emotion_column))).where(
        F.col("emotion_label").isNotNull()
    )
    return _aggregate_metrics(labeled, ["emotion_label"]).withColumn(
        "gold_variant_level", F.lit("emotion")
    ).withColumn("gold_variant_value", F.col("emotion_label"))


def validate_gold(frame, gold_level: str, sample_size: int | None, expected_variants=None) -> dict:
    rows = frame.count()
    failures = []
    if rows == 0:
        failures.append("Gold variant is empty")
    key_map = {
        "sample": ["review_id"],
        "brand-sample": ["gold_variant_value", "review_id"],
        "business": ["business_id"],
        "user": ["user_id"],
        "date": ["date_period"],
        "industry": ["primary_industry"],
        "sentiment": ["sentiment_label"],
        "emotion": ["emotion_label"],
    }
    keys = key_map[gold_level]
    if frame.groupBy(*keys).count().where(F.col("count") > 1).limit(1).count():
        failures.append(f"Gold keys are not unique: {keys}")
    if gold_level == "sample" and sample_size is not None and rows > sample_size:
        failures.append("Sample Gold contains more rows than requested")
    if gold_level == "brand-sample" and sample_size is not None:
        if frame.groupBy("gold_variant_value").count().where(
            F.col("count") > sample_size
        ).limit(1).count():
            failures.append("A brand sample contains more rows than requested")
        if expected_variants:
            actual = {row[0] for row in frame.select("gold_variant_value").distinct().collect()}
            missing = set(expected_variants) - actual
            if missing:
                failures.append(f"No eligible reviews for requested brands: {sorted(missing)}")
    if failures:
        raise AssertionError("Gold validation failed: " + "; ".join(failures))
    return {"passed": True, "gold_level": gold_level, "gold_rows": rows, "keys": keys}


def _avg_if_present(columns: set[str], name: str, alias: str):
    if name in columns:
        return F.avg(F.col(name)).alias(alias)
    return F.first(F.lit(None).cast("double")).alias(alias)


def build_business_summary(gold):
    columns = set(gold.columns)
    expressions = [
        F.first("name", ignorenulls=True).alias("business_name"),
        F.first("city", ignorenulls=True).alias("city"),
        F.first("state", ignorenulls=True).alias("state"),
        F.first("primary_industry", ignorenulls=True).alias("primary_industry"),
        F.count("review_id").alias("review_count"),
        F.avg("stars").alias("avg_stars"),
        F.avg(F.when(F.col("stars") <= 2, 1.0).otherwise(0.0)).alias("low_star_rate"),
        _avg_if_present(columns, "weighted_star", "avg_weighted_star"),
        _avg_if_present(columns, "vader_sentiment_score", "avg_sentiment"),
        _avg_if_present(columns, "anger_int_avg", "avg_anger"),
        _avg_if_present(columns, "fear_int_avg", "avg_fear"),
        _avg_if_present(columns, "sadness_int_avg", "avg_sadness"),
        _avg_if_present(columns, "joy_int_avg", "avg_joy"),
    ]
    summary = gold.groupBy("gold_variant_value", "business_id").agg(*expressions)
    risk_score = F.least(
        F.lit(1.0),
        F.greatest(F.lit(0.0), (F.lit(5.0) - F.col("avg_stars")) / F.lit(4.0)) * 0.45
        + F.col("low_star_rate") * 0.35
        + F.greatest(F.lit(0.0), -F.coalesce(F.col("avg_sentiment"), F.lit(0.0))) * 0.20,
    )
    return (
        summary.withColumn("risk_score", F.round(risk_score, 4))
        .withColumn(
            "attention_tier",
            F.when(F.col("risk_score") >= 0.65, "high")
            .when(F.col("risk_score") >= 0.40, "moderate")
            .otherwise("monitor"),
        )
        .withColumn(
            "recommended_focus",
            F.when(F.col("low_star_rate") >= 0.35, "Review recurring low-star pain points")
            .when(F.col("avg_sentiment") < -0.15, "Investigate negative sentiment drivers")
            .when(F.col("avg_joy") < F.col("avg_sadness"), "Strengthen positive service moments")
            .otherwise("Maintain performance and monitor trends"),
        )
    )


def build_monthly_summary(gold):
    columns = set(gold.columns)
    month = F.coalesce(
        F.col("year_month") if "year_month" in columns else F.lit(None),
        F.date_format(F.col("review_date"), "yyyy-MM"),
    )
    return (
        gold.withColumn("year_month", month)
        .groupBy("gold_variant_value", "year_month", "primary_industry")
        .agg(
            F.count("review_id").alias("review_count"),
            F.avg("stars").alias("avg_stars"),
            F.avg(F.when(F.col("stars") <= 2, 1.0).otherwise(0.0)).alias("low_star_rate"),
            _avg_if_present(columns, "vader_sentiment_score", "avg_sentiment"),
        )
    )
