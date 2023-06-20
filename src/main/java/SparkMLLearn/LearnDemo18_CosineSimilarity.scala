package SparkMLLearn

import org.apache.spark.ml.feature.{BucketedRandomProjectionLSH, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession

object LearnDemo18_CosineSimilarity {
  def main(args: Array[String]): Unit = {
    // 创建 SparkSession
    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("CosineSimilarity")
      .getOrCreate()
    import spark.implicits._
    import org.apache.spark.sql.functions._
    // 计算夹角余弦相似度
    def cosineSimilarity(v1: Vector, v2: Vector): Double = {
      //点积运算
      val dotProduct = v1.dot(v2)
      //使用欧几里得范数标准化规范化  计算两个向量模的乘积
      val normProduct = Vectors.norm(v1, 2) * Vectors.norm(v2, 2)
      /**
       *  将点积除以两个向量的 L2 范数的乘积，得到夹角余弦相似度。夹角余弦相似度是一个介于 -1 和 1 之间的值，
       *  表示了两个向量之间的相似程度。值越接近 1，表示两个向量越相似；值越接近 -1，表示两个向量越相反；
       *  值接近 0，表示两个向量之间没有明显的相关性。
       */
      dotProduct / normProduct
    }

    // 示例向量
    val v1 = Vectors.dense(1.0, 2.0, 3.0)
    val v2 = Vectors.dense(4.0, 5.0, 6.0)

    // 计算夹角余弦相似度
    val similarity = cosineSimilarity(v1, v2)
    println(s"Cosine similarity: $similarity")
    // 停止 SparkSession
    spark.stop()
  }
}
