# Databricks dashboard app

This Streamlit app turns the three dashboard-serving Delta tables into a
shareable Databricks Apps URL. The URL is stable across pipeline runs; the
`data_science_dashboard` job task refreshes the tables behind it.

## Deploy once

1. In **Databricks Apps**, create an app sourced from this GitHub repository and
   select `databricks_integration/dashboard_app` as the source directory.
2. Add a SQL warehouse resource with **Can use** permission and the resource key
   `sql_warehouse`.
3. Grant the app service principal `USE CATALOG` on `workspace`, `USE SCHEMA` on
   `workspace.default`, and `SELECT` on the three `yelp_dashboard_*` tables.
4. Deploy the app. Databricks displays the app URL when deployment completes.
5. Share that URL using the app permissions appropriate for the intended audience.

No personal access token or password belongs in this folder. Databricks injects
the app service principal credentials and the attached warehouse identifier.

The app includes business KPIs, an interactive risk overview, a dynamic Gold
variant comparison, monthly industry trends, prioritized actions, and a review
explorer. It reads only the bounded serving tables created by the final pipeline
task, never the full raw or Silver corpus. Chipotle versus Great Clips appears
only when that example Gold variant is selected.

Platform references:

- [Deploy a Databricks app](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/deploy)
- [Streamlit app tutorial](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/tutorial-streamlit)
- [Add a SQL warehouse resource](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/sql-warehouse)
