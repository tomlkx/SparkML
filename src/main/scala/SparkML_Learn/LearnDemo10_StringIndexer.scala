package SparkML_Learn

import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo10_StringIndexer extends Serializable {
  /**
   * StringIndexer 是一种特征转换器，用于将字符串类型的类别特征转换为数值索引。它将字符串类别映射到唯一的整数值，从而便于在机器学习算法中处理。
   */
  def main(args: Array[String]): Unit = {
    // 创建 SparkSession
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("demo").getOrCreate()

    // 定义输入数据
    val data: DataFrame = spark.createDataFrame(
      Seq(
        (0, "男"), (1, "男"), (2, "女"), (3, "男"), (4, "女"), (5, "女")
      )
    ).toDF("id", "category")

    // 创建 StringIndexer 实例
    val indexer: StringIndexer = new StringIndexer()
      .setInputCol("categoryIndexer") // 设置输入列
      .setOutputCol("categoryIndex") // 设置输出列
      //frequencyDesc 按照标签频率降序(默认)
      //frequencyAsc  按标签频率升序
      //alphabetDesc  降序字母顺序
      //alphabetAsc 升序字母顺序
      .setStringOrderType("frequencyDesc")

    // 拟合模型并进行转换
    val model: StringIndexerModel = indexer.fit(data)
    val transformedData: DataFrame = model.transform(data)

    // 展示转换结果
    transformedData.show(false)

    // 停止 SparkSession
    spark.stop()
  }
}
