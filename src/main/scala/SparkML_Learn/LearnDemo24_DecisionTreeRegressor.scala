package SparkML_Learn

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{VectorIndexer, VectorIndexerModel}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.sql.{DataFrame,SparkSession}

object LearnDemo24_DecisionTreeRegressor {
  def main(args: Array[String]): Unit = {

    // 创建SparkSession
    val spark: SparkSession = SparkSession.builder()
      .appName("demo")
      .master("local[*]")
      .getOrCreate()

    // 加载数据
    val data: DataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    // 特征索引
    val featuresIndexer: VectorIndexerModel = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    // 将数据集分割为训练集和测试集
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // 创建决策树回归模型
    val dt = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    // 创建Pipeline，包括特征索引和决策树回归
    val pipeline: Pipeline = new Pipeline()
      .setStages(Array(featuresIndexer, dt))

    // 在训练数据上拟合Pipeline
    val model: PipelineModel = pipeline.fit(trainingData)

    // 在测试数据上进行预测
    val frame: DataFrame = model.transform(testData)
    frame.show()

    // 使用均方根误差（RMSE）评估模型性能
    val evaluator: RegressionEvaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse: Double = evaluator.evaluate(frame)
    println(s"Root Mean Squared Error (RMSE): $rmse")

    // 停止SparkSession
    spark.stop()
  }
}
