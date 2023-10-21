package SparkML_Learn

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession

object LearnDemo36_Euclidean_distance {
  def main(args: Array[String]): Unit = {
    // 创建 SparkSession
    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("CosineSimilarity")
      .getOrCreate()
    /**
     * 欧式距离越接近0表示两个向量在欧几里得空间中越接近
     * @param v1
     * @param v2
     * @return
     */
    def cosineSimilarity(v1: Vector, v2: Vector): Double = {
      // 计算平方欧式距离
      val squaredEuclideanDistance = Vectors.sqdist(v1, v2)
      // 计算欧式距离（平方根）
      math.sqrt(squaredEuclideanDistance)
    }
    // 示例向量
    val v1 = Vectors.dense(1.0, 1.0, 1.0)
    val v2 = Vectors.dense(1.0, 1.0, 1.0)
    print(v1.getClass.getSimpleName)
    // 计算夹欧式距离
    val similarity = cosineSimilarity(v1, v2)
    println(s"Euclidean_distance: $similarity")
    // 停止 SparkSession
    spark.stop()
  }
}