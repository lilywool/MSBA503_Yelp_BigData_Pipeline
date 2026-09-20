# Dashboard layer

The maintained cloud presentation path is
`databricks_integration/dashboard_app/`. It reads bounded Delta serving tables
produced by the third Databricks job task and deploys as a Streamlit-based
Databricks App with a stable, shareable workspace URL.

The original team notebook is preserved unchanged in
`archival/coursework_dashboard/yelp_universal_dashboard.ipynb`. Its archive
README records module ownership and the notebook's prototype status. The old
file-based sample preparation utility lives beside it in the archive and is not
used by the live Databricks pipeline.
