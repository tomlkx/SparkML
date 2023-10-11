package SparkML_Learn

import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo06_StopWordsRemover {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("StopWordsRemoverExample")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._
    //加载中文停用词
    val chineseStopWords=Array("我","这个")

    val data = Seq(
      (0, Seq("我", "喜欢", "这个", "电影")),
      (1, Seq("这本", "书", "非常", "有趣"))
    ).toDF("id", "words")

    val remover = new StopWordsRemover()
      //设置输入列名
      .setInputCol("words")
      //设置输出列名
      .setOutputCol("filteredWords")
      //载入中文停用用词
      .setStopWords(chineseStopWords)

    val removedData = remover.transform(data)
    removedData.show(false)

    spark.stop()

  }
}
