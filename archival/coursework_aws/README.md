# Original MSBA 503 AWS implementation

The coursework version of this project was developed as an AWS data pipeline.
Lily used Amazon S3 as the data lake, AWS Glue and Athena for cataloging and
querying, and Amazon EMR with PySpark for distributed preparation and feature
processing. After the distributed data work, Amazon SageMaker provided the
separate environment used for transformer-oriented sentiment and emotion
experimentation.

That sequence matters to the project history: EMR was the distributed processing
layer used during the course, and SageMaker followed for the heavier model work.
The current repository does not rewrite that history as if the original class
submission ran on Databricks.

The 2026 revision preserves the AWS pathway in `aws_emr/` while adding a governed
Databricks Delta implementation in `databricks_integration/`. Both use the same
canonical feature code. The Databricks work is a later platform upgrade, not a
claim about what was submitted during the course.

Only this edited architectural record is published from the larger set of
coursework notes. Raw diary-style setup logs, transient errors, local outputs,
credentials, and obsolete environment folders are intentionally excluded from
the public repository.
