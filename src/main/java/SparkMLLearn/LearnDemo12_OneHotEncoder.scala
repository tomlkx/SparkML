package SparkMLLearn

import org.apache.spark.ml.feature.{OneHotEncoder, OneHotEncoderModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo12_OneHotEncoder extends Serializable {
  /**
   *  OneHotEncoder 用于将数值类型的索引列转换为独热编码（One-Hot Encoding）的特征向量。它将每个索引值转换为一个二进制的稀疏向量，其中只有一个元素为 1，表示该索引值对应的分类特征。OneHotEncoder 的输出是一个稀疏向量类型的特征列，通常用于训练机器学习模型。
   */
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("demo").getOrCreate()
    val df: DataFrame = spark.createDataFrame(
      Seq(
        (0.0, 1.0),
        (1.0, 0.0),
        (2.0, 1.0),
        (0.0, 2.0),
        (0.0, 1.0),
        (2.0, 0.0)
      )
    ).toDF("categoryIndex1", "categoryIndex2")
    @transient
    val model: OneHotEncoderModel = new OneHotEncoder().setInputCols(Array("categoryIndex1", "categoryIndex2"))
      .setOutputCols(Array("categoryVec1", "categoryVec2"))
      .fit(df)
    model.transform(df).show(false)
    spark.stop()
  }
}
