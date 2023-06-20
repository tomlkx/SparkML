package SparkMLLearn

import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo05_Tokenizer {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()
    import spark.implicits._
    import org.apache.spark.sql.functions._
    val sentenceDataFrame: DataFrame = spark.createDataFrame(
      Seq(
        (0, "Hi I heard about Spark"),
        (1, "I wish Java could use case classes"),
        (2, "Logistic,regression,models,are,neat")
      )
    ).toDF("id", "sentence")
    val tokenizer: Tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val regexTokenizer: RegexTokenizer = new RegexTokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")
      .setPattern("\\W")
    //自定义UDF
    val countTokens: UserDefinedFunction = udf { (words: Seq[String]) => words.length }
    val tokenized: DataFrame = tokenizer.transform(sentenceDataFrame)
    tokenized.select("sentence", "words")
      .withColumn("tokens", countTokens(col("words"))).show(false)
    val regexTokenized: DataFrame = regexTokenizer.transform(sentenceDataFrame)
    regexTokenized.select("sentence", "words")
      .withColumn("tokens", countTokens(col("words"))).show(false)
    spark.stop()
  }
}
