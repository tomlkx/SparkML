package SparkMLLearn

import org.apache.spark.ml.feature.DCT
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo09_DCT {
  def main(args: Array[String]): Unit = {
    // 创建 SparkSession
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()

    // 定义输入数据
    val data: Seq[linalg.Vector] = Seq(
      Vectors.dense(0.0, 1.0, -2.0, 3.0),
      Vectors.dense(-1.0, 2.0, 4.0, -7.0),
      Vectors.dense(14.0, -2.0, -5.0, 1.0)
    )

    // 创建 DataFrame
    val df: DataFrame = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    // 创建 DCT 实例
    val dct: DCT = new DCT()
      .setInputCol("features") // 设置输入列
      .setOutputCol("featuresDCT") // 设置输出列
      .setInverse(false) // 设置是否进行反向变换

    // 进行 DCT 变换并展示结果
    dct.transform(df).show(false)

    // 停止 SparkSession
    spark.stop()
  }
}
