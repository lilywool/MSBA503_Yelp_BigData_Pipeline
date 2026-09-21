"""
Distributed feature engineering for Yelp reviews - Databricks / Spark version.

The feature logic itself is NOT reimplemented in Spark:
`corrected_feature_engineering.py` is imported UNCHANGED and is the single
source of truth for every feature computation. Spark's only job is
distribution: read the data, split it into partitions, and run ordinary
Python/pandas code (via `mapInPandas`) on each partition. Zero Spark NLP,
zero JVM-side NLP, zero JARs beyond Spark itself.

Usage (local test / smoke run - no Databricks needed):
    python spark_feature_engineering.py --files ../data/chipotle_sample_15000.csv \
        --lexicons-dir ../lexicons --outdir ../data/processed --master local[4]

On Databricks: same script, run as a Job (not an interactive notebook) against a
JVM-free Premium job cluster. `spark` is provided by the runtime; SparkSession.builder
.getOrCreate() picks it up automatically, so this file needs zero changes to run there -
just point --lexicons-dir (or the individual --*-lexicon flags) at DBFS/Volumes
instead of local paths.

= Lexicon loading: broadcast, not re-parsed per partition =
The four specialized lexicons (worry/WCST/Yelp/NRC-intensity - plain dicts and
sets, a few MB total) are parsed ONCE here at the driver and shipped to every
executor via a Spark broadcast variable, cached there for reuse across every
partition that executor runs - not re-read from DBFS/S3 and re-parsed on every
single partition, which is what calling cfe.init_models(worry_path=..., ...)
directly inside the mapInPandas function would do. spaCy/VADER/NRCLex/the VAD
lexicon remain worker-local because they are stateful model objects, not
broadcastable plain data. They are cached when Spark reuses a Python worker.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

from pyspark import SparkFiles
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, DoubleType
)

sys.path.insert(0, str(Path(__file__).parent))
import corrected_feature_engineering as cfe  # the frozen, combined payload
import iteration2_lexicons as il
from lexicon_cli import add_lexicon_args, resolve_lexicon_paths

DOUBLE_COLS = {
    "avg_word_len", "avg_sentence_len",
    "anger_int_avg", "fear_int_avg", "joy_int_avg", "sadness_int_avg",
    "anticipation_int_avg", "disgust_int_avg", "surprise_int_avg", "trust_int_avg",
    "Valence_avg", "Arousal_avg", "Dominance_avg",
    "vader_sentiment_score", "vader_pos", "vader_neu", "vader_neg",
    "hf_sentiment_confidence", "hf_computed_sentiment", "hf_emotion_confidence",
    "noun_pct", "verb_pct", "adj_pct", "adv_pct", "subjectivity_score", "type_token_ratio",
    "wcst_warmth_avg", "wcst_competence_avg", "wcst_sociability_avg", "wcst_trust_avg",
    "yelp_sentiment_avg", "nrc_anger_lex", "nrc_fear_lex", "nrc_joy_lex", "nrc_sadness_lex",
    "nrc_anticipation_lex", "nrc_disgust_lex", "nrc_surprise_lex", "nrc_trust_lex",
    "primary_emotion_lex_conf", "secondary_emotion_lex_conf", "weighted_star",
}
LONG_COLS = {
    "word_count", "char_count", "num_excl", "num_ques", "num_caps", "num_at",
    "num_hash", "sentence_count", "negation_count",
    "anger_count", "fear_count", "joy_count", "sadness_count", "anticipation_count",
    "disgust_count", "surprise_count", "trust_count", "positive_count", "negative_count",
    "person_count", "location_count", "product_count", "vad_matched_words",
    "hf_sentiment_label", "worry_word_count", "yelp_matched_words",
}


def _generated_schema(use_transformers: bool, feature_groups=None) -> StructType:
    cols = cfe.output_columns_for(feature_groups, use_transformers)
    fields = []
    for c in cols:
        if c in DOUBLE_COLS:
            fields.append(StructField(c, DoubleType(), True))
        elif c in LONG_COLS:
            fields.append(StructField(c, LongType(), True))
        else:
            fields.append(StructField(c, StringType(), True))
    return StructType(fields)


def make_mapper(
    text_col: str,
    vad_lexicon_filename: str | None,
    lexicons_bc,
    use_transformers: bool,
    reference_date,
    feature_groups=None,
    lexicon_payloads: dict[str, bytes] | None = None,
):
    """Returns the function passed to mapInPandas.

    Each Spark task calls this with an iterator of pandas batches.
    spaCy/VADER/NRCLex/VAD are initialized worker-locally and reused when the
    Python worker survives across tasks. Classic Spark uses ``lexicons_bc``;
    Spark Connect/serverless receives the five small lexicons as exact byte
    payloads in the mapper closure.

    `reference_date` is computed ONCE at the driver (see run()) from the full
    dataset and passed in here as a plain closed-over value - every partition
    uses the SAME decay anchor for weighted_star. Computing it independently
    per-partition (e.g. each partition's own local max review_date) would
    silently give different reviews different decay anchors depending on
    which partition they happened to land in - a distributed-aggregate bug
    of exactly the "looks fine, is quietly wrong" shape this repo exists to
    prevent.
    """
    def _map(batches):
        if lexicon_payloads is not None:
            # Spark Connect/serverless has no SparkContext or SparkFiles. The
            # five licensed resources are small enough to travel once with the
            # Python closure and are parsed once per reused Python worker.
            cfe.init_models(lexicon_payloads=lexicon_payloads)
        else:
            # Classic Spark distributes the VAD file with addFile(); resolve its
            # executor-local path rather than closing over a driver path.
            vad_lexicon_path = SparkFiles.get(vad_lexicon_filename)
            cfe.init_models(
                vad_lexicon_path=vad_lexicon_path,
                preloaded_lexicons=lexicons_bc.value,
            )
        if use_transformers:
            cfe.init_transformers()
        out_cols = cfe.output_columns_for(feature_groups, use_transformers)
        for pdf in batches:
            input_cols = list(pdf.columns)
            result = cfe.process_dataframe(
                pdf,
                text_col=text_col,
                use_transformers=use_transformers,
                reference_date=reference_date,
                feature_groups=feature_groups,
            )
            # schema = input columns (passed through) + generated columns, in that
            # order - mapInPandas requires every yielded batch to carry the FULL
            # declared schema, not just the newly-computed columns.
            yield result[input_cols + out_cols]
    return _map


def _classic_spark_context(spark):
    """Return SparkContext for classic Spark, or None for Spark Connect/serverless."""
    try:
        return spark.sparkContext
    except AttributeError:
        return None
    except Exception as exc:
        message = str(exc)
        if (
            "JVM_ATTRIBUTE_NOT_SUPPORTED" in message
            or "not supported in Spark Connect" in message
        ):
            return None
        raise


def _stage_lexicons(spark, lexicon_paths: dict) -> tuple[dict, dict]:
    """Distribute every lexicon with Spark and return driver-local paths + filenames.

    This makes local paths, DBFS paths, and S3 URIs follow the same execution
    model. Loaders parse the staged driver-local files; workers resolve their own
    copy through SparkFiles instead of trying to call Python ``open()`` on S3.
    """
    driver_paths, filenames = {}, {}
    for key, source in lexicon_paths.items():
        filename = Path(source.rstrip("/\\")).name
        if not filename:
            raise ValueError(f"Cannot determine a filename for lexicon path: {source}")
        # spark-submit --files may already have staged this basename. Avoid
        # adding the downloaded copy a second time under a different source URI.
        if not Path(SparkFiles.get(filename)).exists():
            spark.sparkContext.addFile(source)
        driver_paths[key] = SparkFiles.get(filename)
        filenames[key] = filename
    return driver_paths, filenames


def _add_industry_columns(sdf):
    """Derive the first three comma-separated Yelp categories when available."""
    if "categories" not in sdf.columns:
        return sdf
    categories = F.split(F.coalesce(F.col("categories"), F.lit("")), r"\s*,\s*")
    for index, name in enumerate(
        ("primary_industry", "secondary_industry", "tertiary_industry")
    ):
        value = F.trim(categories.getItem(index))
        sdf = sdf.withColumn(name, F.when(value != "", value))
    return sdf.drop("categories")


def read_prepared_input(spark, input_path: str, input_format: str = "csv"):
    """Read an already-joined review-level table used by smoke/parity runs."""
    if input_format == "parquet":
        return spark.read.parquet(input_path)
    if input_format != "csv":
        raise ValueError(f"Unsupported prepared input format: {input_format}")
    return (spark.read
            .option("header", True)
            .option("inferSchema", True)
            .option("multiLine", True)
            .option("escape", '"')
            .csv(input_path))


def build_silver_frame(reviews, businesses, users):
    """Build the canonical Silver review table from three Spark DataFrames.

    Keeping the relational work in one function lets local Parquet, AWS S3, and
    Unity Catalog Delta inputs share the exact same join, aliases, and
    deduplication behavior.
    """
    reviews = (reviews.select(
                   "review_id", "user_id", "business_id",
                   F.col("stars").cast("double").alias("stars"),
                   F.to_date("date").alias("review_date"),
                   F.col("text").alias("raw_review"),
               )
               .dropDuplicates(["review_id"]))

    businesses = (businesses.select(
                      "business_id", F.col("name").alias("name"), "city", "state",
                      F.col("review_count").cast("long").alias("business_review_count"),
                      F.col("is_open").cast("long").alias("is_open"), "categories",
                  )
                  .dropDuplicates(["business_id"]))

    users = (users.select(
                 "user_id", F.col("name").alias("user_name"),
                 F.col("review_count").cast("long").alias("user_review_count"),
                 F.col("average_stars").cast("double").alias("user_stars_avg"),
             )
             .dropDuplicates(["user_id"]))

    joined = (reviews
              .join(F.broadcast(businesses), "business_id", "left")
              .join(users, "user_id", "left")
              .withColumn("year_month", F.date_format("review_date", "yyyy-MM")))
    return _add_industry_columns(joined)


def read_joined_input(spark, reviews_path: str, businesses_path: str, users_path: str):
    """Build the Silver review-level table from three Yelp Parquet datasets."""
    return build_silver_frame(
        spark.read.parquet(reviews_path),
        spark.read.parquet(businesses_path),
        spark.read.parquet(users_path),
    )


def apply_features(spark, sdf, text_col: str, lexicon_paths: dict,
                   use_transformers: bool, num_partitions: int | None = None,
                   feature_groups=None):
    """Apply the canonical feature payload to an already prepared Spark frame."""
    spark_context = _classic_spark_context(spark)
    if spark_context is not None:
        # Classic Spark workers receive source modules through addPyFile. The
        # serverless Job installs the repository itself as a Python project.
        pipeline_dir = Path(__file__).parent
        for fname in ("real_feature_engineering.py", "iteration2_features.py",
                      "iteration2_lexicons.py", "corrected_feature_engineering.py"):
            module_path = pipeline_dir / fname
            if module_path.exists():
                spark_context.addPyFile(str(module_path))

    if text_col not in sdf.columns:
        raise ValueError(
            f"Text column {text_col!r} is missing. Available columns: {sdf.columns}"
        )

    staged_filenames = None
    lexicons_bc = None
    lexicon_payloads = None
    if spark_context is None:
        # Google Drive files have already been materialized as temporary local
        # files by the Databricks entry point. Close over their exact bytes so
        # Spark Connect can distribute them without SparkContext/SparkFiles.
        lexicon_payloads = {}
        for key, source in lexicon_paths.items():
            path = Path(source)
            if not path.is_file():
                raise ValueError(
                    "Serverless lexicons must be driver-local files before feature "
                    f"distribution; not found: {source}"
                )
            lexicon_payloads[key] = path.read_bytes()
    else:
        # Classic Spark retains its executor-safe SparkFiles + broadcast path.
        staged_paths, staged_filenames = _stage_lexicons(spark, lexicon_paths)
        lexicons = il.load_all(
            worry_path=staged_paths["worry_path"],
            wcst_path=staged_paths["wcst_path"],
            yelp_path=staged_paths["yelp_path"],
            nrc_intensity_path=staged_paths["nrc_intensity_path"],
        )
        lexicons_bc = spark_context.broadcast(lexicons)

    # Never trust old feature values - drop and fully regenerate every
    # pipeline-owned column from raw text, same rule as the CLI pipelines.
    keep = [c for c in sdf.columns if c not in cfe.ALL_GENERATED_COLUMNS]
    sdf = sdf.select(keep)

    # Global reference date for weighted_star's decay, computed ONCE here at the
    # driver (a single Spark aggregate over the whole dataset) - see make_mapper()'s
    # docstring for why this must not be computed independently per partition.
    reference_date = None
    if "review_date" in sdf.columns:
        max_row = sdf.agg(F.max(F.to_timestamp("review_date")).alias("m")).first()
        if max_row and max_row["m"] is not None:
            reference_date = max_row["m"].replace(tzinfo=timezone.utc)

    if num_partitions:
        sdf = sdf.repartition(num_partitions)

    resolved_groups = cfe.normalize_feature_groups(feature_groups, use_transformers)
    schema = StructType(
        sdf.schema.fields + _generated_schema(use_transformers, resolved_groups).fields
    )
    mapper = make_mapper(
        text_col,
        (
            staged_filenames["vad_lexicon_path"]
            if staged_filenames is not None
            else None
        ),
        lexicons_bc,
        use_transformers,
        reference_date,
        resolved_groups,
        lexicon_payloads=lexicon_payloads,
    )
    result = sdf.mapInPandas(mapper, schema=schema)
    return result


def run(spark, input_path: str, text_col: str, lexicon_paths: dict,
        use_transformers: bool, num_partitions: int | None = None,
        input_format: str = "csv", feature_groups=None):
    """Backward-compatible prepared-input entry point used by parity tests."""
    sdf = read_prepared_input(spark, input_path, input_format=input_format)
    sdf = _add_industry_columns(sdf)
    return apply_features(
        spark, sdf, text_col, lexicon_paths, use_transformers, num_partitions,
        feature_groups,
    )


def run_joined(spark, reviews_path: str, businesses_path: str, users_path: str,
               text_col: str, lexicon_paths: dict, use_transformers: bool,
               num_partitions: int | None = None, feature_groups=None):
    """Full Bronze-Parquet -> Silver join -> distributed feature entry point."""
    sdf = read_joined_input(spark, reviews_path, businesses_path, users_path)
    return apply_features(
        spark, sdf, text_col, lexicon_paths, use_transformers, num_partitions,
        feature_groups,
    )


def _output_path(base: str, name: str) -> str:
    if "://" in base:
        return f"{base.rstrip('/')}/{name}"
    return str(Path(base) / name)


def _write_result(result, output_path: str, output_format: str) -> None:
    writer = result.write.mode("overwrite")
    if output_format == "csv":
        writer.option("header", True).csv(output_path)
    elif output_format == "parquet":
        writer.parquet(output_path)
    elif output_format == "delta":
        writer.format("delta").save(output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributed Spark wrapper around the feature-engineering pipeline."
    )
    parser.add_argument("--files", nargs="+", help="Prepared review-level CSV/Parquet inputs")
    parser.add_argument("--input-format", choices=("csv", "parquet"), default="csv")
    parser.add_argument("--reviews-path", help="Full Yelp review Parquet path")
    parser.add_argument("--businesses-path", help="Full Yelp business Parquet path")
    parser.add_argument("--users-path", help="Full Yelp user Parquet path")
    parser.add_argument("--text-col", default="raw_review")
    add_lexicon_args(parser)
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--output-format", choices=("parquet", "csv", "delta"), default="parquet")
    parser.add_argument(
        "--at-scale", action="store_true",
        help="Use Spark-native validation; required for the full joined run",
    )
    parser.add_argument(
        "--log-path", default=None,
        help="Audit-log path. URI paths are append-only Spark JSON directories.",
    )
    parser.add_argument("--delta-log-path", default=None,
                        help="Optional Databricks Delta audit-table path")
    parser.add_argument("--partitions", type=int, default=None)
    parser.add_argument("--transformers", action="store_true")
    parser.add_argument(
        "--master", default=None,
        help="Override Spark master (for example local[4]); leave unset on a cluster.",
    )
    args = parser.parse_args()

    joined_args = (args.reviews_path, args.businesses_path, args.users_path)
    if args.files and any(joined_args):
        parser.error("Use either --files or the three joined-input paths, not both.")
    if not args.files and not any(joined_args):
        parser.error("Pass --files or all of --reviews-path/--businesses-path/--users-path.")
    if any(joined_args) and not all(joined_args):
        parser.error("The full joined run requires all three Parquet paths.")
    if all(joined_args) and not args.at_scale:
        parser.error("The full joined run requires --at-scale validation.")

    lexicon_paths = resolve_lexicon_paths(args)

    builder = SparkSession.builder.appName("yelp-review-intelligence-distributed")
    if args.master:
        builder = builder.master(args.master)
    spark = builder.getOrCreate()

    if args.files:
        jobs = [
            {
                "name": Path(infile.rstrip("/\\")).stem,
                "input": infile,
                "build": lambda infile=infile: run(
                    spark, infile, args.text_col, lexicon_paths, args.transformers,
                    args.partitions, input_format=args.input_format,
                ),
            }
            for infile in args.files
        ]
    else:
        jobs = [{
            "name": "yelp_full",
            "input": {
                "reviews": args.reviews_path,
                "businesses": args.businesses_path,
                "users": args.users_path,
            },
            "build": lambda: run_joined(
                spark, args.reviews_path, args.businesses_path, args.users_path,
                args.text_col, lexicon_paths, args.transformers, args.partitions,
            ),
        }]

    from run_logger import log_run
    from spark_validate import validate_spark, validate_spark_at_scale

    log_path = args.log_path
    if log_path is None:
        log_path = (_output_path(args.outdir, "pipeline_run_log")
                    if "://" in args.outdir else
                    str(Path(args.outdir) / "pipeline_run_log.json"))

    try:
        for job in jobs:
            name = job["name"]
            print(f"\n{'='*60}\nDISTRIBUTED PROCESSING: {job['input']}\n{'='*60}")
            result = None
            metrics = None
            outfile = _output_path(
                args.outdir,
                f"{name}_REAL_distributed.{args.output_format}",
            )
            entry = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "input": job["input"],
                "output": outfile,
                "lexicon_paths": lexicon_paths,
                "transformers": args.transformers,
                "engine": "spark_mapInPandas",
                "validation_mode": "spark_native" if args.at_scale else "driver_collect",
            }
            try:
                # Persist before validation: validation, row count, and output write
                # must not run the expensive NLP map more than once.
                result = job["build"]().persist()
                metrics = (validate_spark_at_scale(result, text_col=args.text_col)
                           if args.at_scale else
                           validate_spark(result, text_col=args.text_col))
                entry["rows"] = result.count()
                entry["validation"] = metrics
                _write_result(result, outfile, args.output_format)
                entry["outcome"] = "success"
                log_run(entry, log_path=log_path, spark=spark,
                        delta_table_path=args.delta_log_path)
                print(f"Saved: {outfile}")
            except Exception as exc:
                entry["outcome"] = "failed"
                entry["validation"] = metrics or {
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                try:
                    log_run(entry, log_path=log_path, spark=spark,
                            delta_table_path=args.delta_log_path)
                except Exception as log_exc:
                    print(f"ERROR: failed to record failed run: {log_exc}", file=sys.stderr)
                raise
            finally:
                if result is not None:
                    result.unpersist()
    finally:
        spark.stop()
