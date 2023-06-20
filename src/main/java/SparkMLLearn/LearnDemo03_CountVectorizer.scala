package SparkMLLearn

import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo03_CountVectorizer {
  /**
   * CountVectorizer 是一个用于将文本数据转换为特征向量的特征提取器。它将文本数据中的词语进行计数，并将每个词语映射到一个唯一的整数索引，然后生成一个稀疏向量表示文本数据。
   */
  def main(args: Array[String]): Unit = {
    import org.apache.spark.sql.{SparkSession, DataFrame}
    import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

    // 创建 SparkSession
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()

    // 创建 DataFrame，包含 id 和 words 列
    val df: DataFrame = spark.createDataFrame(
      Seq(
        (0, Array("a", "b", "c")), // 第一行数据
        (1, Array("a", "b", "b", "c", "a","f","g","a")) // 第二行数据
      )
    ).toDF("id", "words")
    /**
     * 最小文档频率（MinDF）是以整个数据集合为单位进行计算的。
     *
     * 在CountVectorizer中，最小文档频率是指一个词语在整个数据集合中出现的最小次数。换句话说，一个词语只有在至少达到最小文档频率指定的次数时，才会被纳入作为特征进行考虑。
     *
     * 最小文档频率的计算是基于整个数据集合中的文档统计信息完成的，而不是基于单个文档。通过对整个数据集合进行分析，可以确定哪些词语的出现频率达到了最小文档频率的要求，从而进行特征提取的筛选和过滤操作。
     *
     * 因此，在设置最小文档频率时，需要考虑整个数据集合中词语的出现情况，以确定合适的阈值。这样可以确保只有在足够数量的文档中出现过的词语被纳入考虑，从而提高特征的质量和准确性。
     */
    // 创建 CountVectorizerModel
    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("words") // 指定输入列为 "words"
      .setOutputCol("features") // 指定输出列为 "features"
      .setVocabSize(3) // 设置词汇表(理解为提取特征数量)大小为 3
      .setMinDF(2) // 设置最小文档频率为 2
      .fit(df) // 在 DataFrame 上拟合 CountVectorizerModel

    // 使用 CountVectorizerModel 对 DataFrame 进行转换，并展示结果
    cvModel.transform(df).show(false)
    import spark.implicits._
    import org.apache.spark.sql.functions._
    val vocabulary: Array[String] = cvModel.vocabulary
    val topicsWithKeywords = cvModel.transform(df)
      .select($"topic", $"termIndices")
      .as[(Int, Array[Int])]
      .map { case (topicId, termIndices) =>
        val keywords = termIndices.map(vocabulary(_))
        (topicId, keywords)
      }
    // 停止 SparkSession
    spark.stop()

  }
}
