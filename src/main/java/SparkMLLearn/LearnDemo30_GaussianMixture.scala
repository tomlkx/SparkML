package SparkMLLearn

import org.apache.spark.ml.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo30_GaussianMixture {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()

    // 加载数据
    val data: DataFrame = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")

    // 创建 GaussianMixture 对象并设置参数
    val gmm: GaussianMixture = new GaussianMixture()
      .setK(2) // 设置聚类簇的数量

    // 使用数据训练模型
    val model: GaussianMixtureModel = gmm.fit(data)

    // 遍历每个聚类簇
    for (i <- 0 until model.getK) {
      // 输出每个聚类簇的权重
      println(s"Weight for cluster $i: ${model.weights(i)}")

      // 输出每个聚类簇的均值
      println(s"Mean for cluster $i:")
      println(model.gaussians(i).mean)

      // 输出每个聚类簇的协方差矩阵
      println(s"Covariance matrix for cluster $i:")
      println(model.gaussians(i).cov)
    }

    spark.stop()
  }
}
