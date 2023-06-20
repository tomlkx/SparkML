package SparkMLLearn

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, StringIndexerModel, VectorIndexer, VectorIndexerModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo22_LogisticRegression {
  def main(args: Array[String]): Unit = {
    // 创建SparkSession
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("demo").getOrCreate()
    import spark.implicits._
    import org.apache.spark.sql.functions._

    // 加载数据
    val data: DataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    // 随机拆分训练集和测试集
    val Array(data1, data2) = data.randomSplit(Array(0.7, 0.3))

    // 对标签进行索引，并添加标签列的元数据
    val indexerLabel: StringIndexerModel = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("IndexerLabel")
      .fit(data)

    // 自动识别分类特征，并对它们进行索引
    val IndexerFeature: VectorIndexerModel = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("IndexerFeatures")
      .setMaxCategories(4)
      .fit(data)

    // 创建逻辑回归模型
    val lor: LogisticRegression = new LogisticRegression()
      .setLabelCol("IndexerLabel")
      .setFeaturesCol("IndexerFeatures")
      .setPredictionCol("prediction")

    // 将索引标签转换回原始标签
    val labelConverter: IndexToString = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(indexerLabel.labelsArray(0))

    // 创建Pipeline，并设置各个阶段
    val pipeline: Pipeline = new Pipeline()
      .setStages(Array(indexerLabel, IndexerFeature, lor, labelConverter))

    // 在训练集上训练Pipeline模型
    val model: PipelineModel = pipeline.fit(data1)

    // 在测试集上进行预测
    val frame: DataFrame = model.transform(data2)

    // 创建多分类分类器评估器
    val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("IndexerLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    // 计算模型在测试集上的准确率
    val accuracy: Double = evaluator.evaluate(frame)

    // 打印预测结果和准确率
    frame.show()
    println(s"Accuracy: $accuracy")

    // 停止SparkSession
    spark.stop()
  }
}
