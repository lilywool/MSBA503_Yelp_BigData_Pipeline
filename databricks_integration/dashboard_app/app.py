"""Streamlit presentation layer for the Yelp Databricks serving tables."""

from __future__ import annotations

import os
import re

import altair as alt
import pandas as pd
import plotly.express as px
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
               latitude, longitude,
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

overview, geography, variant_comparison, trends, priorities, explorer = st.tabs(
    [
        "Business overview",
        "Geographic view",
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

with geography:
    mapped = filtered.dropna(subset=["latitude", "longitude"]).copy()
    if mapped.empty:
        st.info("No business coordinates are available for the selected Gold variant.")
    else:
        st.subheader("Business locations")
        largest = max(float(mapped["review_count"].max()), 1.0)
        mapped["marker_size"] = 80.0 + (mapped["review_count"] / largest) * 920.0
        st.map(
            mapped,
            latitude="latitude",
            longitude="longitude",
            size="marker_size",
            use_container_width=True,
        )

        st.subheader("United States summary")
        mapped["weighted_stars"] = mapped["avg_stars"] * mapped["review_count"]
        mapped["weighted_sentiment"] = mapped["avg_sentiment"] * mapped["review_count"]
        mapped["weighted_risk"] = mapped["risk_score"] * mapped["review_count"]
        state_summary = (
            mapped.groupby("state", as_index=False)
            .agg(
                review_count=("review_count", "sum"),
                weighted_stars=("weighted_stars", "sum"),
                weighted_sentiment=("weighted_sentiment", "sum"),
                weighted_risk=("weighted_risk", "sum"),
            )
        )
        state_summary["avg_stars"] = (
            state_summary["weighted_stars"] / state_summary["review_count"]
        )
        state_summary["avg_sentiment"] = (
            state_summary["weighted_sentiment"] / state_summary["review_count"]
        )
        state_summary["avg_risk"] = (
            state_summary["weighted_risk"] / state_summary["review_count"]
        )
        metric_options = {
            "Review volume": ("review_count", "Reds", None),
            "Average rating": ("avg_stars", "YlGn", (1.0, 5.0)),
            "Average sentiment": ("avg_sentiment", "RdYlGn", (-1.0, 1.0)),
            "Average risk": ("avg_risk", "Reds", (0.0, 1.0)),
        }
        metric_label = st.selectbox("Color states by", list(metric_options))
        metric, color_scale, range_color = metric_options[metric_label]
        choropleth = px.choropleth(
            state_summary,
            locations="state",
            locationmode="USA-states",
            color=metric,
            scope="usa",
            color_continuous_scale=color_scale,
            range_color=range_color,
            hover_name="state",
            hover_data={
                "review_count": ":,",
                "avg_stars": ":.2f",
                "avg_sentiment": ":.3f",
                "avg_risk": ":.3f",
                "state": False,
            },
            labels={metric: metric_label},
        )
        choropleth.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=450)
        st.plotly_chart(choropleth, use_container_width=True)
        st.caption(
            "The choropleth renders U.S. state codes; Canadian provinces and other "
            "regions remain available in the tables and filters."
        )

        st.dataframe(
            mapped[
                [
                    "business_name", "city", "state", "primary_industry",
                    "review_count", "avg_stars", "risk_score",
                ]
            ].sort_values("review_count", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

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
