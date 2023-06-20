package SparkMLLearn

import org.apache.spark.ml.clustering.{BisectingKMeans, BisectingKMeansModel}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo29_BisectingKMeans {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()
    val data: DataFrame = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")
    val bkm = new BisectingKMeans()
      .setK(2)
      .setMaxIter()
      .setSeed(1)
      .setMinDivisibleClusterSize()
      .setDistanceMeasure()
      .setFeaturesCol()
      .setPredictionCol()
    val model: BisectingKMeansModel = bkm.fit(data)
    val prediction: DataFrame = model.transform(data)
    val evaluator = new ClusteringEvaluator()
    val silhouette: Double = evaluator.evaluate(prediction)
    println(silhouette)
    spark.stop()
  }
}
