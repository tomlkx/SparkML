package SparkMLLearn

import org.apache.spark.ml.{Pipeline, linalg}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.{Row, SparkSession}

object LearnDemo32_CrossValidator {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()

    import org.apache.spark.sql.functions._
    import spark.implicits._

    // 创建训练数据集 DataFrame
    val training = spark.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0),
      (4L, "b spark who", 1.0),
      (5L, "g d a y", 0.0),
      (6L, "spark fly", 1.0),
      (7L, "was mapreduce", 0.0),
      (8L, "e spark program", 1.0),
      (9L, "a e c l", 0.0),
      (10L, "spark compile", 1.0),
      (11L, "hadoop software", 0.0)
    )).toDF("id", "text", "label")

    // 创建分词器 Tokenizer
    val tokenizer: Tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    // 输入列为"text"，输出列为"words"，将输入的文本按空格进行分词

    // 创建HashingTF特征转换器
    val hashingTF = new HashingTF()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    // 输入列为分词器的输出列"words"，输出列为"features"，将分词后的文本转换为特征向量

    // 创建逻辑回归模型
    val lr: LogisticRegression = new LogisticRegression()
      //标签列为"label"
      .setLabelCol("label")
      //特征列为"features"
      .setFeaturesCol("features")
      //最大迭代次数为10
      .setMaxIter(10)

    // 创建Pipeline管道
    val pipeline: Pipeline = new Pipeline()
      // 设置管道的阶段，按顺序包括分词器、特征转换器和逻辑回归模型
      .setStages(Array(tokenizer, hashingTF, lr))

    // 创建参数网格
    val paramGrid: Array[ParamMap] = new ParamGridBuilder()
      // 对于特征转换器，尝试不同的特征数量（numFeatures）
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      // 对于逻辑回归模型，尝试不同的正则化参数（regParam）
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()
    // 创建交叉验证器
    val cv: CrossValidator = new CrossValidator()
      // 使用管道作为估计器（estimator）
      .setEstimator(pipeline)
      // 使用二分类评估器（evaluator）进行模型评估
      .setEvaluator(new BinaryClassificationEvaluator)
      // 使用参数网格（paramGrid）进行参数搜索
      .setEstimatorParamMaps(paramGrid)
      //设置交叉验证的折数（numFolds）为2
      .setNumFolds(2)
      //设置并行度（parallelism）为2
      .setParallelism(2)

    // 在训练数据上进行交叉验证
    val cvModel: CrossValidatorModel = cv.fit(training)

    // 创建测试数据集 DataFrame
    val test = spark.createDataFrame(Seq(
      (4L, "spark i j k"),
      (5L, "l m n"),
      (6L, "mapreduce spark"),
      (7L, "apache hadoop")
    )).toDF("id", "text")

    // 使用训练好的模型对测试数据进行预测
    val predictions = cvModel.transform(test)

    // 打印预测结果
    predictions
      .select("id", "text", "probability", "prediction")
      .collect()
      .foreach { case Row(id: Long, text: String, prob: linalg.Vector, prediction: Double) =>
        println(s"($id, $text) --> prob=$prob, prediction=$prediction")
      }

    spark.stop()
  }
}
