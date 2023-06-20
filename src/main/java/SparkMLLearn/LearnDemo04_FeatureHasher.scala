package SparkMLLearn

import org.apache.spark.ml.feature.FeatureHasher
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo04_FeatureHasher {
  def main(args: Array[String]): Unit = {
    // 创建 SparkSession
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("demo").getOrCreate()

    // 创建输入数据 DataFrame
    val data: DataFrame = spark.createDataFrame(
      Seq(
        (2.2, true, "1", "张三"),
        (3.3, false, "2", "李四"),
        (4.4, false, "3", "张明"),
        (5.5, false, "4", "张三")
      )
    ).toDF("real", "bool", "stringNum", "string")

    // 创建 FeatureHasher 实例
    val hasher: FeatureHasher = new FeatureHasher()
      .setInputCols("real", "bool", "stringNum", "string") // 设置输入列名
      .setOutputCol("features") // 设置输出列名
      .setNumFeatures(300) // 设置特征维度

    // 使用 FeatureHasher 进行转换，生成特征向量
    val transformedData = hasher.transform(data)

    // 显示转换后的数据
    transformedData.show(false)

    // 停止 SparkSession
    spark.stop()
  }
}
