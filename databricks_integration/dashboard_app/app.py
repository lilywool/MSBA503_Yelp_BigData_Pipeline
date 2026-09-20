"""Streamlit presentation layer for the Yelp Databricks serving tables."""

from __future__ import annotations

import os
import re

import altair as alt
import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config


IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
CATALOG = os.getenv("YELP_CATALOG", "workspace")
SCHEMA = os.getenv("YELP_SCHEMA", "default")
PREFIX = os.getenv("YELP_TABLE_PREFIX", "yelp")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")


def _qualified_table(suffix: str) -> str:
    parts = (CATALOG, SCHEMA, f"{PREFIX}_{suffix}")
    if not all(IDENTIFIER.fullmatch(part) for part in parts):
        raise ValueError("Dashboard table identifiers contain unsupported characters")
    return ".".join(f"`{part}`" for part in parts)


def _connection():
    if not WAREHOUSE_ID:
        raise RuntimeError(
            "Attach a SQL warehouse resource with key 'sql_warehouse' before deploying."
        )
    config = Config()
    hostname = config.host.removeprefix("https://").removeprefix("http://")
    return sql.connect(
        server_hostname=hostname,
        http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
        credentials_provider=lambda: config.authenticate,
    )


@st.cache_data(ttl=300, show_spinner="Loading validated Yelp metrics...")
def query_frame(query: str) -> pd.DataFrame:
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [column[0] for column in cursor.description]
            return pd.DataFrame(rows, columns=columns)


def load_businesses() -> pd.DataFrame:
    return query_frame(
        f"""
        SELECT gold_variant_value, business_id, business_name, city, state, primary_industry,
               review_count, avg_stars, low_star_rate, avg_weighted_star,
               avg_sentiment, avg_anger, avg_fear, avg_sadness, avg_joy,
               risk_score, attention_tier, recommended_focus
        FROM {_qualified_table('dashboard_business_summary')}
        ORDER BY review_count DESC
        LIMIT 10000
        """
    )


def load_monthly() -> pd.DataFrame:
    return query_frame(
        f"""
        SELECT gold_variant_value, year_month, primary_industry, review_count, avg_stars,
               low_star_rate, avg_sentiment
        FROM {_qualified_table('dashboard_monthly_summary')}
        WHERE year_month IS NOT NULL
        ORDER BY year_month
        LIMIT 50000
        """
    )


def load_reviews() -> pd.DataFrame:
    return query_frame(
        f"""
        SELECT gold_variant_level, gold_variant_value, review_id, business_id,
               name AS business_name, city, state,
               primary_industry, stars, review_date, raw_review,
               vader_sentiment_score, dominant_emotion
        FROM {_qualified_table('dashboard_review_sample')}
        ORDER BY review_date DESC
        LIMIT 30000
        """
    )
st.set_page_config(page_title="Yelp Review Intelligence", page_icon="★", layout="wide")
st.title("Yelp Review Intelligence")
st.caption("Validated review features, business priorities, and customer-experience trends")

try:
    businesses = load_businesses()
    monthly = load_monthly()
    reviews = load_reviews()
except Exception as exc:
    st.error(f"Dashboard data is unavailable: {exc}")
    st.stop()

states = sorted(value for value in businesses["state"].dropna().unique())
industries = sorted(value for value in businesses["primary_industry"].dropna().unique())
variants = sorted(value for value in businesses["gold_variant_value"].dropna().unique())
selected_variants = st.sidebar.multiselect("Gold variant", variants, default=variants)
selected_states = st.sidebar.multiselect("State", states)
selected_industries = st.sidebar.multiselect("Industry", industries)
minimum_reviews = st.sidebar.number_input("Minimum reviews", min_value=1, value=10, step=5)

filtered = businesses[businesses["review_count"] >= minimum_reviews].copy()
if selected_variants:
    filtered = filtered[filtered["gold_variant_value"].isin(selected_variants)]
if selected_states:
    filtered = filtered[filtered["state"].isin(selected_states)]
if selected_industries:
    filtered = filtered[filtered["primary_industry"].isin(selected_industries)]

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Businesses", f"{len(filtered):,}")
kpi2.metric("Reviews represented", f"{int(filtered['review_count'].sum()):,}")
kpi3.metric("Average stars", f"{filtered['avg_stars'].mean():.2f}" if len(filtered) else "—")
kpi4.metric("High-attention businesses", f"{int((filtered['attention_tier'] == 'high').sum()):,}")

overview, variant_comparison, trends, priorities, explorer = st.tabs(
    [
        "Business overview",
        "Gold variant comparison",
        "Monthly trends",
        "Priority actions",
        "Review explorer",
    ]
)

with overview:
    scatter = (
        alt.Chart(filtered)
        .mark_circle(opacity=0.65)
        .encode(
            x=alt.X("avg_stars:Q", title="Average stars", scale=alt.Scale(domain=[1, 5])),
            y=alt.Y("risk_score:Q", title="Risk score"),
            size=alt.Size("review_count:Q", title="Review count"),
            color=alt.Color("attention_tier:N", title="Attention tier"),
            tooltip=[
                "gold_variant_value:N", "business_name:N", "city:N", "state:N",
                "primary_industry:N",
                alt.Tooltip("review_count:Q", format=","),
                alt.Tooltip("avg_stars:Q", format=".2f"),
                alt.Tooltip("risk_score:Q", format=".3f"),
            ],
        )
        .properties(height=460)
        .interactive()
    )
    st.altair_chart(scatter, use_container_width=True)

with variant_comparison:
    comparison_reviews = reviews.copy()
    if selected_variants:
        comparison_reviews = comparison_reviews[
            comparison_reviews["gold_variant_value"].isin(selected_variants)
        ]
    variant_summary = (
        comparison_reviews.groupby("gold_variant_value", as_index=False)
        .agg(
            reviews=("review_id", "count"),
            avg_stars=("stars", "mean"),
            avg_sentiment=("vader_sentiment_score", "mean"),
        )
    )
    st.dataframe(variant_summary, use_container_width=True, hide_index=True)
    rating_counts = (
        comparison_reviews.groupby(["gold_variant_value", "stars"], as_index=False)
        .size()
        .rename(columns={"size": "reviews"})
    )
    ratings = (
        alt.Chart(rating_counts)
        .mark_bar()
        .encode(
            x=alt.X("stars:O", title="Star rating"),
            y=alt.Y("reviews:Q", title="Reviews"),
            color=alt.Color("gold_variant_value:N", title="Gold variant"),
            xOffset="gold_variant_value:N",
            tooltip=[
                "gold_variant_value:N",
                "stars:O",
                alt.Tooltip("reviews:Q", format=","),
            ],
        )
        .properties(height=420)
    )
    st.altair_chart(ratings, use_container_width=True)

with trends:
    trend = monthly.copy()
    if selected_variants:
        trend = trend[trend["gold_variant_value"].isin(selected_variants)]
    if selected_industries:
        trend = trend[trend["primary_industry"].isin(selected_industries)]
    else:
        top_industries = (
            trend.groupby("primary_industry", dropna=True)["review_count"]
            .sum().nlargest(8).index
        )
        trend = trend[trend["primary_industry"].isin(top_industries)]
    trend["year_month"] = pd.to_datetime(trend["year_month"], errors="coerce")
    chart = (
        alt.Chart(trend.dropna(subset=["year_month"]))
        .mark_line(point=False)
        .encode(
            x=alt.X("year_month:T", title="Month"),
            y=alt.Y("avg_stars:Q", title="Average stars", scale=alt.Scale(domain=[1, 5])),
            color=alt.Color("primary_industry:N", title="Industry"),
            strokeDash=alt.StrokeDash("gold_variant_value:N", title="Gold variant"),
            tooltip=["year_month:T", "gold_variant_value:N", "primary_industry:N", "review_count:Q", alt.Tooltip("avg_stars:Q", format=".2f")],
        )
        .properties(height=460)
    )
    st.altair_chart(chart, use_container_width=True)

with priorities:
    priority_columns = [
        "business_name", "city", "state", "primary_industry", "review_count",
        "avg_stars", "low_star_rate", "avg_sentiment", "risk_score",
        "attention_tier", "recommended_focus",
    ]
    st.dataframe(
        filtered.sort_values(["risk_score", "review_count"], ascending=[False, False])[
            priority_columns
        ],
        use_container_width=True,
        hide_index=True,
    )

with explorer:
    visible_reviews = reviews.copy()
    if selected_variants:
        visible_reviews = visible_reviews[
            visible_reviews["gold_variant_value"].isin(selected_variants)
        ]
    if selected_states:
        visible_reviews = visible_reviews[visible_reviews["state"].isin(selected_states)]
    if selected_industries:
        visible_reviews = visible_reviews[
            visible_reviews["primary_industry"].isin(selected_industries)
        ]
    st.dataframe(visible_reviews, use_container_width=True, hide_index=True)
