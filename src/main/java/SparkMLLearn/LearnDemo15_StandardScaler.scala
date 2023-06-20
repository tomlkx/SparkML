package SparkMLLearn

import org.apache.spark.ml.feature.{StandardScaler, StandardScalerModel, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo15_StandardScaler {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("demo").getOrCreate()

    // 导入所需的类和函数
    import spark.implicits._
    import org.apache.spark.sql.functions._

    // 创建示例数据集
    val data = Seq(
      (0, 2.0, 3.0),
      (1, -1.0, 5.0),
      (2, 4.0, 7.0),
      (3, 6.0, 9.0),
      (4, 8.0, 11.0)
    )
    val df: DataFrame = spark.createDataFrame(data).toDF("id", "feature1", "feature2")

    // 创建一个VectorAssembler，将特征列合并为一个向量列
    val va: VectorAssembler = new VectorAssembler()
      .setInputCols(Array("feature1", "feature2"))
      .setOutputCol("features")

    // 将DataFrame转换为包含向量特征列的新DataFrame
    val asDataFrame: DataFrame = va.transform(df)

    // 创建一个StandardScaler，对特征进行标准化
    val scaler: StandardScaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaleFeatures")
      .setWithStd(true)    // 设置是否对特征进行标准差缩放
      .setWithMean(false)  // 设置是否从特征中减去均值

    // 使用StandardScaler对数据进行拟合，得到一个StandardScalerModel
    val scalerModel: StandardScalerModel = scaler.fit(asDataFrame)

    // 使用StandardScalerModel对数据进行转换，得到标准化后的特征列
    val scaledData: DataFrame = scalerModel.transform(asDataFrame)

    // 展示标准化后的数据
    scaledData.show(false)

    // 停止Spark会话
    spark.stop()

  }
}
