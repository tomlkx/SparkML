package SparkML_Learn

import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object LearnDemo19_NaiveBayes extends Serializable {
  def main(args: Array[String]): Unit = {
    // 创建 SparkSession
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("NaiveBayExample")
      .getOrCreate()

    // 加载数据集，格式为libsvm
    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    // 将数据集分割为训练集和测试集
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)

    // 创建朴素贝叶斯模型对象，并设置参数
    val model = new NaiveBayes()
      .setLabelCol("label") // 设置标签列名
      .setFeaturesCol("features") // 设置特征列名
      .setSmoothing(1.0) // 设置平滑参数，用于处理零概率
      .setModelType("multinomial") // 设置模型类型为多项式朴素贝叶斯
      .fit(trainingData) // 在训练集上拟合模型

    // 在测试集上进行预测
    val predictions = model.transform(testData)

    // 创建多类分类评估器，并设置参数
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction")

    // 计算预测结果的准确率
    evaluator.setMetricName("accuracy")
    var accuracy = evaluator.evaluate(predictions)
    println(s"Test set accuracy = $accuracy")

    // 计算召回率
    evaluator.setMetricName("recallByLabel")
    accuracy = evaluator.evaluate(predictions)
    println(s"Test set recall = $accuracy")

    // 计算精准率
    evaluator.setMetricName("precision")
    accuracy = evaluator.evaluate(predictions)
    println(s"Test set precision = $accuracy")

    spark.stop()
  }
}
