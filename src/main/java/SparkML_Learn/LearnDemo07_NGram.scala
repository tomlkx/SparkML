package SparkML_Learn

import org.apache.spark.ml.feature.NGram
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo07_NGram {
  /**
   * n-gram模型中的n-gram是指由n个连续的单词组成的短语。在自然语言处理中，n-gram被用于分析文本中的连续单词序列，从而捕捉语言中的局部上下文信息。
   */
  def main(args: Array[String]): Unit = {
    // 创建SparkSession
    val spark = SparkSession.builder()
      .appName("demo")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // 创建DataFrame，包含id和words列
    val wordDataFrame = Seq(
      (0, Array("Hi", "I", "heard", "about", "Spark")),
      (1, Array("I", "wish", "Java", "could", "use", "case", "classes")),
      (2, Array("Logistic", "regression", "models", "are", "neat"))
    ).toDF("id", "words")

    // 创建NGram实例，设置n-gram的n值为2，输入列为"words"，输出列为"ngrams"
    val ngram = new NGram().setN(2).setInputCol("words").setOutputCol("ngrams")

    // 应用n-gram转换器，生成包含n-gram结果的新DataFrame
    val ngramDataFrame = ngram.transform(wordDataFrame)

    // 展示n-gram结果的DataFrame
    ngramDataFrame.show(false)

    // 停止SparkSession
    spark.stop()
  }
}
