# Original team dashboard

`yelp_universal_dashboard.ipynb` is the original coursework dashboard notebook,
preserved without code or output edits. Its SHA-256 digest at archival time is:

```text
E79AEEF1833E7C4763E11D6D4B8E4586ACFFDFE2B54B761E1AB21FA3E9640C09
```

The prototype combined work from three team members:

- **Alex:** dashboard layout, navigation, data loading, Altair visualizations,
  overview KPIs, filters, and time-series forecasting.
- **Eddie:** diagnostic pain-point analysis, emotion aggregation, at-risk
  business signals, and prescriptive recommendations.
- **Lily:** the NLP/data pipeline and the early pain-point and positive-driver
  feature prototypes used by the dashboard analysis.

The notebook is preserved as a coursework artifact because its master app shell
was not completed. The live 2026 Databricks implementation adapts the analytical
ideas into governed serving tables and a separately deployable Streamlit app;
it does not alter the historical notebook or misrepresent its completion state.

`prepare_dashboard_data.py` is also archived here. It prepares the original
file-based Chipotle/Great Clips samples for offline compatibility and is not an
input to the 2026 Databricks workflow.

This named-contributor coursework artifact is retained for portfolio and
educational reference and is not relicensed by the repository's Apache license.
