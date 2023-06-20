package SparkMLLearn

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorIndexerModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo20_RandomForestClassifier {
  def main(args: Array[String]): Unit = {
    //创建SparkSession
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("demo").getOrCreate()
    //加载并解析数据文件,将其转换为DataFrame
    val data: DataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    data.show
    //对标签进行索引,并为标签列添加元数据.整个数据集进行拟合包含所有标签在索引中
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    //自动识别分类特征,对其进行索引.设置maxCategories,一遍具有>4个不同值的特征视为连续特征
    val featureIndexer: VectorIndexerModel = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)
    //将数据集拆分为训练和测试集(30%)
    val Array(trainingData,testData) = data.randomSplit(Array(0.7,0.3))
    //训练一个随机森林模型
    val rf: RandomForestClassifier = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)
    //将索标签转换回原始标签
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labelsArray(0))
    //在管道中串联索引器和随机森林模型
    val pipeline: Pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
    //训练模型,这里会运行索引器
    val model: PipelineModel = pipeline.fit(trainingData)
    //进行预测
    val predictions: DataFrame = model.transform(testData)
    //选择示例进行现实
    predictions.show()
    //选择(预测值,真实标签)并计算测试误差
    val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")
    spark.stop()
  }
}
