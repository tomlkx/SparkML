package SparkMLLearn

import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel, Tokenizer}
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo01_TF_IDF {
  /**
   * TF-IDF 是一种常用的文本特征表示方法，它结合了词频和逆文档频率的概念，用于衡量一个单词在文档中的重要程度。TF（词频）表示单词在文档中的出现频率，IDF（逆文档频率）表示单词在整个语料库中的稀有程度。TF-IDF 将这两个概念相乘，得到单词的权重，用于表示单词在文档中的重要性。TF-IDF 在信息检索、文本分类、关键词提取等任务中被广泛应用。
   * @param args
   */
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()
    val sentenceData: DataFrame = spark.createDataFrame(
      Seq(
        (0.0, "Hi I heard about Spark"),
        (0.0, "I wish Java could use case classes"),
        (1.0, "Logistic regression models are neat")
      )
    ).toDF("label", "sentence")
    //默认按照空格进行分词
    val tokenizer: Tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData: DataFrame = tokenizer.transform(sentenceData)
    // 词频 设置输入列words同时设置输出列rawFeatures 并 设置提取特征数量为20
    val hashingTF: HashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
    val featurizedData: DataFrame = hashingTF.transform(wordsData)
    // 逆文档率
    val idf: IDF = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    // 拟合
    val idfModel: IDFModel = idf.fit(featurizedData)
    // 变换
    val rescaleData: DataFrame = idfModel.transform(featurizedData)
    rescaleData.show()
    spark.stop()
  }
}
