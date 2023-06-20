package SparkMLLearn

import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo08_PolynomialExpansion {
  /**
   *
   * PolynomialExpansion（多项式展开）是 Spark ML 中的一个特征转换器，用于生成输入特征的多项式组合。它可以将原始特征的高阶多项式组合作为新的特征，从而引入更多的非线性关系。
   */
  def main(args: Array[String]): Unit = {
    // 创建 SparkSession
    val spark: SparkSession = SparkSession.builder()
      .master("local[*]")
      .appName("demo")
      .getOrCreate()

    // 创建输入数据
    val data: Array[linalg.Vector] = Array(
      Vectors.dense(2.0, 1.0),
      Vectors.dense(0.0, 0.0),
      Vectors.dense(3.0, -1.0)
    )

    // 将数据转换为 DataFrame
    val df: DataFrame = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    // 创建 PolynomialExpansion 实例
    val polyExpansion: PolynomialExpansion = new PolynomialExpansion()
      .setInputCol("features")
      .setOutputCol("polyFeatures")
    //将多项式阶数设置为 3。这意味着 PolynomialExpansion 将在原始特征的基础上生成所有阶数小于等于 3 的多项式组合。
      .setDegree(3)

    // 应用 PolynomialExpansion 转换器并显示结果
    val transformedDF: DataFrame = polyExpansion.transform(df)
    transformedDF.show(false)

    // 停止 SparkSession
    spark.stop()

  }
}
