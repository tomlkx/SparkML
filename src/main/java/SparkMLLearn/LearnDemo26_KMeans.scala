package SparkMLLearn

import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo26_KMeans {
  def main(args: Array[String]): Unit = {
    // 创建SparkSession
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()
    // 加载数据
    val data: DataFrame = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")
    data.show()
    // 创建KMeans 模型设置参数
    val kmeans: KMeans = new KMeans()
      .setK(2) //设置聚类的簇数
    //训练KMeans模型
    val model: KMeansModel = kmeans.fit(data)
    //进行预测
    val predictions: DataFrame = model.transform(data)
    predictions.show(false)
    //评估聚类结果,计算Silhouette系数
    val evaluator = new ClusteringEvaluator()
      .setMetricName("silhouette")
    val silhouette: Double = evaluator.evaluate(predictions)
    println(silhouette)
    model.clusterCenters.foreach(println(_))
    spark.stop()
  }
}
