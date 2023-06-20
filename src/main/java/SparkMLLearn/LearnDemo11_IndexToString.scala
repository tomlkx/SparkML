package SparkMLLearn

import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo11_IndexToString {
  /**
   * IndexToString 是一个特征转换器，用于将索引列转换回原始的标签列。它的作用与 StringIndexer 相反，StringIndexer 将标签列转换为索引列，而 IndexToString 将索引列转换回原始的标签列。
   */
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()

    // 创建DataFrame
    val df: DataFrame = spark.createDataFrame(
      Seq(
        (0, "红"),
        (1, "黄"),
        (2, "蓝"),
        (3, "蓝"),
        (4, "蓝"),
        (5, "红")
      )
    ).toDF("id", "category")

    // 创建StringIndexer并进行转换
    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .fit(df)
    val indexed: DataFrame = indexer.transform(df)
    indexed.show()

    // 打印StringIndexer的元数据信息
    println(Attribute.fromStructField(indexed.schema(indexer.getOutputCol)))

    // 创建IndexToString并进行转换  会 按照StringIndexer转换会原本的数据
    val converter: IndexToString = new IndexToString()
      .setInputCol("categoryIndex")
      .setOutputCol("originalCategory")

    converter.transform(indexed).show()

    spark.stop()

  }
}
