package SparkML_Learn

import org.apache.spark.ml.linalg.{Matrix, Vector, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.SparkSession

object LearnDemo37_PearsonCorrelation {
  def main(args: Array[String]): Unit = {
    // 创建 SparkSession
    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("PearsonCorrelation")
      .getOrCreate()

    import spark.implicits._

    // 创建示例向量
    val v1 = Vectors.dense(1.0, 2.0, 3.0, 4.0)
    val v2 = Vectors.dense(2.0, 3.0, 4.0, 5.0)

    // 将向量转化为数据帧
    val data = Seq((0, v1), (1, v2)).toDF("id", "features")

    // 计算皮尔逊相关系数
    val correlation = Correlation.corr(data, "features", "pearson").head
    val matrix = correlation.getAs[Matrix](0)
    // 获取皮尔逊相关系数
    val pearsonCorrelation = matrix(0, 1)

    println(s"皮尔逊相关系数: $pearsonCorrelation")

    // 停止 SparkSession
    spark.stop()
  }
}
