package SparkMLLearn

import org.apache.spark.ml.feature.{PCA, PCAModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo07_PCA {
  /**
   * Spark ML中的PCA（Principal Component Analysis）是一种常用的降维技术，用于将高维数据转换为低维数据，同时保留数据的主要特征。PCA通过线性变换将原始数据投影到一个新的特征空间，新的特征空间中的维度数量比原始数据的维度数量少，从而实现数据的降维。
   */
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("demo").getOrCreate()
    val data = Array(
      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    )
    val df: DataFrame = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    val pac: PCAModel = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(3)
      .fit(df)
    pac.transform(df).printSchema()
    spark.stop()
  }
}
