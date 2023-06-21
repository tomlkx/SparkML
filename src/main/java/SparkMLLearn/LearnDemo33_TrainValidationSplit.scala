package SparkMLLearn

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo33_TrainValidationSplit {
  def main(args: Array[String]): Unit = {
    // 创建SparkSession对象
    val spark: SparkSession = SparkSession.builder()
      .appName("demo")
      .master("local[*]")
      .getOrCreate()

    // 读取数据
    val data: DataFrame = spark.read.format("libsvm").load("data/mllib/sample_linear_regression_data.txt")

    // 将数据集划分为训练集和测试集
    val Array(training, test) = data.randomSplit(Array(0.9, 0.1), seed = 123456)

    // 创建线性回归模型
    val lr: LinearRegression = new LinearRegression()
      .setMaxIter(10)

    // 创建参数网格
    val paramGrid: Array[ParamMap] = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept, Array(true, false))
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    // 创建训练验证拆分对象
    val trainValidationSplit: TrainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)
      .setParallelism(2)

    // 在训练集上拟合模型
    val model: TrainValidationSplitModel = trainValidationSplit.fit(training)

    // 在测试集上进行预测
    val predictions: DataFrame = model.transform(test)

    // 选择需要显示的列
    predictions.select("features", "label", "prediction").show()

    // 停止SparkSession，释放资源
    spark.stop()
  }
}
