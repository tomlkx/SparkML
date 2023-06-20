package SparkMLLearn

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, StringIndexerModel, VectorAssembler, VectorIndexer, Word2Vec, Word2VecModel}
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo13_VectorIndexer {
  /**
   * VectorIndexer是Spark ML中的一个特征转换器，用于自动识别和编码数值型特征中的分类特征。它可以根据指定的阈值将数值型特征中的分类特征转换为数值编码，以便在机器学习算法中使用。
   */
  def main(args: Array[String]): Unit = {
    // 创建 SparkSession
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("demo").getOrCreate()
    import spark.implicits._
    import org.apache.spark.sql.functions._
    val datSeq = Seq(
      (1, Vectors.dense(1, 2, 3, 4)),
      (2, Vectors.dense(0, 2, 0, 4)),
      (3, Vectors.dense(4, 5, 2, 4)),
      (4, Vectors.dense(2, 2, 1, 3))
    )
    // 定义输入数据
    val data: DataFrame = spark.createDataFrame(
      datSeq
    ).toDF("id", "category")
    val indexer: VectorIndexer = new VectorIndexer()
      .setInputCol("category")
      .setOutputCol("categoryEncode")
      /**
       * VectorIndexer是Spark ML中的一个特征转换器，用于自动识别和编码分类特征。setMaxCategories方法是
       * VectorIndexer的一个参数，用于指定将具有多少个不同取值的特征被认为是分类特征，而不是连续特征。
       * 在给定的数据集中，如果一个特征的取值个数超过了setMaxCategories设定的阈值，那么该特征将被认为是连
       * 续特征，而不是分类特征。对于连续特征，VectorIndexer不会对其进行编码。
       */
      .setMaxCategories(2)
    indexer.fit(data).transform(data).show(false)
    spark.stop()
  }
}
