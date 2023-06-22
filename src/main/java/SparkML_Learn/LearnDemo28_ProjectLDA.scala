package SparkML_Learn
import com.hankcs.hanlp.HanLP
import com.hankcs.hanlp.seg.common.Term
import org.apache.spark.ml.clustering.{LDA, LDAModel}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, IDF, IDFModel, StopWordsRemover, StringIndexer, StringIndexerModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

import java.util
import scala.jdk.CollectionConverters.asScalaBufferConverter

object LearnDemo28_ProjectLDA {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()
    import spark.implicits._
    import org.apache.spark.sql.functions._
    val data: DataFrame = spark.read.option("header","true").csv("/media/liukaixin/机械2盘/数据/情感数据微信公众号.csv")
    //自定义分词
    def segment(text:String):List[String]={
      val list: List[Term] = HanLP.segment(text).asScala.toList
      //不包含 w 的意思是不包含标点符号
      list.filter(item=>item.nature.toString != "w" && item.word.length > 1).map(item=>{
        item.word
      })
    }
    //注册分词器
    spark.udf.register("segment",segment _)
    //分词
    val words: DataFrame = data.select("content").selectExpr("segment(content) as words")
    //祛除停用词
    val remover: StopWordsRemover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("StopWords")
      .setStopWords(spark.read.text("data/stopwords.txt").rdd.collect().map(_.get(0).toString))
    val removeStopData: DataFrame = remover.transform(words)
    //特征提取
    val model: CountVectorizerModel = new CountVectorizer()
      .setInputCol("StopWords")
      .setOutputCol("features")
      .setVocabSize(1000)
      .fit(removeStopData)
    val exData: DataFrame = model.transform(removeStopData)
    val idf: IDF = new IDF()
      .setInputCol("features")
      .setOutputCol("idfFeatures")
    val idfModel: IDFModel = idf.fit(exData)
    val idfData: DataFrame = idfModel.transform(exData)
    //构建主题模型
    val lda: LDA = new LDA()
      .setK(30)
      .setMaxIter(200)
      .setFeaturesCol("idfFeatures")
    val ldaModel: LDAModel = lda.fit(idfData)
    val ll: Double = ldaModel.logLikelihood(idfData)
    println(ll)
    // 特征提取 词汇表
    val vocabulary: Array[String] = model.vocabulary
    // 输出每个主题对应的关键词
    val topics = ldaModel.describeTopics(20)
    topics.show(false)
    val topicsWithKeywords = topics
      .select($"topic", $"termIndices")
      .as[(Int, Array[Int])]
      .map { case (topicId, termIndices) =>
        val keywords = termIndices.map(vocabulary(_))
        (topicId, keywords)
      }
    topicsWithKeywords.show(false)
    spark.stop()
  }

}
