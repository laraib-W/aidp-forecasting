"""One Spark session for the Spark versions of the models: AIDP's own, or a local one on a laptop."""
import os
import sys


def get_spark(spark=None):
    """Configure and return `spark` (pass the notebook's session on AIDP), or start a local one on a laptop."""
    if spark is None:  # laptop: start a local one (needs pyspark and Java 17)
        from pyspark.sql import SparkSession

        os.environ["PYSPARK_PYTHON"] = sys.executable  # workers must use this same Python
        spark = SparkSession.builder.master("local[*]").config("spark.driver.memory", "6g").getOrCreate()

    # Arrow off: on AIDP the duplicate numpy/pandas installed by requirements.txt break the Arrow transfer
    # between pandas and Spark (AttributeError: 'numpy.ndarray' object has no attribute 'isna').
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
    # Keep one task per group-partition: adaptive execution would merge small groups and lose the parallelism.
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "false")
    # About 4 tasks per worker core: every core busy, without hundreds of empty tasks (Spark's default is 200).
    spark.conf.set("spark.sql.shuffle.partitions", spark.sparkContext.defaultParallelism * 4)
    return spark
