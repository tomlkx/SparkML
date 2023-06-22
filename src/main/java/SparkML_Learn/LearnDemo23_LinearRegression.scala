package SparkML_Learn

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel, LinearRegressionTrainingSummary}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object LearnDemo23_LinearRegression {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()
    // 加载训练数据
    val training: DataFrame = spark.read.format("libsvm").load("data/mllib/sample_linear_regression_data.txt")
    // 划分数据集
    val Array(data1,data2) = training.randomSplit(Array(0.7, 0.3))
    // 创建线性回归对象并设置参数
    val lr: LinearRegression = new LinearRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(100)
    val lrModel: LinearRegressionModel = lr.fit(data1)
    // 对训练集上的模型进行总结并打印一些指标
    val trainingSummary: LinearRegressionTrainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")          // 打印总迭代次数
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")  // 打印目标函数历史记录
    trainingSummary.residuals.show()                                       // 打印残差
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")              // 打印均方根误差
    println(s"r2: ${trainingSummary.r2}")                                  // 打印R平方值
    // 创建回归模型评估器

    val evaluator: RegressionEvaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val frame: DataFrame = lrModel.transform(data2)
    val d: Double = evaluator.evaluate(frame)
    println(d)
    spark.stop()
  }
}
