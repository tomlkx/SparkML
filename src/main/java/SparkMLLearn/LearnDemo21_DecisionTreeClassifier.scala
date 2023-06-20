package SparkMLLearn

import org.apache.spark.ml.{Pipeline, PipelineModel, classification}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, StringIndexerModel, VectorIndexer, VectorIndexerModel}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object LearnDemo21_DecisionTreeClassifier {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("demo").getOrCreate()
    import spark.implicits._
    import org.apache.spark.sql.functions._
    //加载数据
    val data: DataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    data.show()
    // 对标签进行索引,并添加标签列的元数据
    val labelIndexer: StringIndexerModel = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    // 自动识别分类特征,并对他们进行索引
    val featureIndexer: VectorIndexerModel = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4) // 将具有大于4个不同值的特征视为连续特征
      .fit(data)

    // 将数据集合拆分为训练和测试集(30% 用于测试)
    val Array(trainingData,testData) = data.randomSplit(Array(0.7, 0.3))
    // 训练一个决策树模型
    val dt: DecisionTreeClassifier = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
    //将索引标签转换回原始标签
    val labelConverter: IndexToString = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      /***
       * .setLabels(labelIndexer.labelsArray(0)) 是一个方法调用，用于设置转换器或模型中的标签。
       * 在这个特定的代码片段中，labelIndexer 是一个 StringIndexerModel 对象，而 labelsArray(0) 是该对象的一个方法，
       * 用于获取索引到标签的映射数组。.setLabels() 方法将这个标签数组作为参数，用于设置转换器或模型中的标签。
       *
       * 具体来说，这段代码中使用了一个标签编码器 labelIndexer 来将标签列中的字符串标签转换为数值索引。
       * 然后，通过调用 .labelsArray(0) 方法，从标签编码器中获取索引到标签的映射数组。最后，调用 .setLabels() 方法，将这
       * 个标签数组设置为转换器或模型的标签。
       */
      .setLabels(labelIndexer.labelsArray(0))
    //在Pipeline 中链接索引和决策树
    val pipeline: Pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
    // 训练模型,同时执行索引器的操作
    val model: PipelineModel = pipeline.fit(trainingData)
    //进行预测
    val predictions: DataFrame = model.transform(testData)
    //选择要显示的示例行
    predictions.select("predictedLabel", "label", "features").show()
    //选择(预测值,真实标签),并计算测试误差
    val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy: Double = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0) - accuracy}")

    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
    spark.stop()
  }
}
