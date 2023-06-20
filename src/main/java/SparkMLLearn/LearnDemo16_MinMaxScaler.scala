package SparkMLLearn

import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object LearnDemo16_MinMaxScaler {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("demo").getOrCreate()

    // 导入所需的类和函数
    import spark.implicits._
    import org.apache.spark.sql.functions._
    val dataFrame = spark.createDataFrame(Seq(
      (1, Vectors.dense(2.0, 1.1, 1.0)),
      (0, Vectors.dense(1.0, 0.1, -1.0)),
      (2, Vectors.dense(3.0, 10.1, 3.0))
    )).toDF("id", "features")
    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
    // Compute summary statistics and generate MinMaxScalerModel
    val scalerModel = scaler.fit(dataFrame)

    // rescale each feature to range [min, max].
    val scaledData = scalerModel.transform(dataFrame)
    println(s"Features scaled to range: [${scaler.getMin}, ${scaler.getMax}]")
    scaledData.select("features", "scaledFeatures").show()
      spark.stop()
  }
}
