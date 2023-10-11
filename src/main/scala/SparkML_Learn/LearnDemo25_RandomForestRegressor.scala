package SparkML_Learn

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{VectorIndexer, VectorIndexerModel}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.sql.{DataFrame,SparkSession}

object LearnDemo25_RandomForestRegressor {
  def main(args: Array[String]): Unit = {
    // 创建SparkSession
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("demo").getOrCreate()

    // 加载数据
    val data: DataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    // 特征向量索引
    val vectorIndexerModel: VectorIndexerModel = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeature")
      .setMaxCategories(4)
      .fit(data)

    // 将数据集拆分为训练集和测试集
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // 随机森林回归模型
    val rf: RandomForestRegressor = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeature")
    // 构建Pipeline，将特征向量索引和随机森林回归模型串联起来
    val pipeline: Pipeline = new Pipeline()
      .setStages(Array(vectorIndexerModel, rf))

    // 训练模型
    val model: PipelineModel = pipeline.fit(trainingData)

    // 进行预测
    val prediction: DataFrame = model.transform(testData)

    // 评估模型性能
    val evaluator: RegressionEvaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse: Double = evaluator.evaluate(prediction)

    // 输出均方根误差（RMSE）
    println(rmse)

    // 停止SparkSession
    spark.stop()
  }
}
