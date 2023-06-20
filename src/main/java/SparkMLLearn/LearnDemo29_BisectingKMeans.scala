package SparkMLLearn

import org.apache.spark.ml.clustering.{BisectingKMeans, BisectingKMeansModel}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo29_BisectingKMeans {
  def main(args: Array[String]): Unit = {
    // 创建SparkSession
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()

    // 加载数据集
    val data: DataFrame = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")

    // 创建BisectingKMeans对象并设置参数
    val bkm = new BisectingKMeans()
      .setK(2) // 设置簇的数量
      .setSeed(1) // 设置随机种子

    // 使用数据集训练模型
    val model: BisectingKMeansModel = bkm.fit(data)

    // 对数据集进行预测
    val prediction: DataFrame = model.transform(data)

    // 创建ClusteringEvaluator评估器
    val evaluator = new ClusteringEvaluator()

    // 计算轮廓系数评估模型效果
    val silhouette: Double = evaluator.evaluate(prediction)

    // 输出轮廓系数
    println(silhouette)

    // 停止SparkSession
    spark.stop()
  }
}
