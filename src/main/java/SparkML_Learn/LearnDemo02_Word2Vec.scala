package SparkML_Learn

import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.ml.linalg
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object LearnDemo02_Word2Vec {
  /**
   * Word2Vec 模型是一种基于神经网络的词嵌入算法，用于将单词映射到连续向量空间中。通过学习上下文语境中单词之间的关系，Word2Vec 可以捕捉到单词的语义和语法关联。这使得具有相似上下文的单词在向量空间中距离较近，从而可以应用于词义相似度计算、文本分类、命名实体识别等任务。Word2Vec 模型适合于在大型语料库上进行训练，并且能够处理未登录词（Out-of-Vocabulary）。
   * @param args
   */
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("demo").getOrCreate()
    import spark.implicits._
    import org.apache.spark.sql.functions._
    val documentDF: DataFrame = spark.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)
    ).toDF("text")
    /**
     * 如果你设置 Word2Vec 模型只提取一个句子中的三个向量，提取规则通常基于词语的重要性或者词语的频率。
     *
     * 具体来说，Word2Vec 模型会对句子进行处理，将其中的每个词语转换为一个向量。然后，根据某种规则或算法，选择句子中最重要或者最频繁出现的词语的向量作为输出。这个规则可以基于词语的权重、词频、词语在句子中的位置等因素。
     *
     * 具体的提取规则取决于你如何定义和实现。你可以根据自己的需求和数据特点，选择适合的方法来提取句子中的三个向量。例如，你可以基于词语的 TF-IDF 值进行排序，选择前三个权重最高的词语的向量作为输出；或者根据词语在句子中的位置，选择最靠前或最靠后的三个词语的向量。
     *
     * 总之，提取句子中的三个向量的具体规则是根据你的需求和实际情况而定，可以根据词语的重要性、频率、位置等因素进行选择。
     */
    val word2Vec: Word2Vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(3) //设置向量的维度为 3
      .setMinCount(0)   //设置最小出现次数为 0
    val word2VecModel: Word2VecModel = word2Vec.fit(documentDF)
    word2VecModel.transform(documentDF).collect().foreach{case Row(text:Seq[_],features:linalg.Vector)=>println(s"Text: [${text.mkString(", ")}] => \nVector: $features\n")}
    spark.stop()
  }
}
