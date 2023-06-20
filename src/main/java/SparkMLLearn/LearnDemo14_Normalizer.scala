package SparkMLLearn

import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo14_Normalizer {
  /**
   *
   * Normalizer是Spark ML中的一个特征转换器，用于对向量特征进行标准化或归一化处理。它将每个特征向量缩放到单位范数（长度为1）。这种标准化方法通常用于对特征向量进行长度归一化，使其不会因为向量的维度不同而产生偏差。
   *
   * @param args
   */
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("demo").getOrCreate()
    import spark.implicits._
    import org.apache.spark.sql.functions._
    val dataFrame: DataFrame = spark.createDataFrame(
      Seq(
        (0, Vectors.dense(1.0, 0.5, -1.0)),
        (1, Vectors.dense(2.0, 1.0, 1.0)),
        (2, Vectors.dense(4.0, 10.0, 2.0))
      )
    ).toDF("id", "features")
    val normalizer: Normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      // 0: 零范数,表示非零元素的数量
      // 1: $L^1$范数,表示所有元素的绝对值之和
      // 2: $L^2$范数,也称为欧几里得范数,表示向量元素的平方根
      .setP(1)
    val l1NormData: DataFrame = normalizer.transform(dataFrame)
    l1NormData.show(false)
    // Normalize each Vector using $L^\infty$ norm.
    val lInfNormData = normalizer.transform(dataFrame, normalizer.p -> Double.PositiveInfinity)
    lInfNormData.show(false)
    spark.stop()
  }
}
