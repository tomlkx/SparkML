package SparkML_ExcavteActualCombat

import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object test4 {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()
    // 创建用户-商品数据，10个商品，0表示未购买，1表示已购买
    val data = Seq(
      Vectors.dense(0, 1, 0, 1, 0, 0, 0, 0, 1, 0),
      Vectors.dense(1, 0, 0, 0, 1, 1, 0, 0, 0, 0),
      Vectors.dense(0, 0, 1, 0, 1, 0, 0, 0, 1, 0),
      Vectors.dense(0, 1, 0, 0, 1, 1, 0, 0, 1, 0),
      Vectors.dense(1, 0, 0, 0, 1, 1, 0, 0, 0, 0)
      // 添加更多用户数据
    )

    val rows: RDD[Vector] = spark.sparkContext.parallelize(data)

    // 创建一个 RowMatrix
    val matrix = new RowMatrix(rows)

    // 执行SVD分解
    val k = 5 // 保留前5个奇异值
    val svd = matrix.computeSVD(k, computeU = true)

    // 获取左奇异向量（U）和奇异值（s）
    val U: RowMatrix = svd.U
    val s: Vector = svd.s

    // 计算余弦相似度
    val userVectors = U.rows.collect()
    val userFeatures = userVectors.zipWithIndex.map { case (vec, user_id) => (user_id, vec) }

    val itemVectors = data.zipWithIndex.map { case (vec, sku_id) => (sku_id, vec) }
    itemVectors.foreach(println)
    val similarity = userFeatures.flatMap { case (user_id, user_vec) =>
      itemVectors.map { case (sku_id, item_vec) =>
        val cosineSimilarity = user_vec.dot(item_vec) / (Vectors.norm(user_vec, 2) * Vectors.norm(item_vec, 2))
        (user_id, sku_id, cosineSimilarity)
      }
    }

    // 获取每个用户与未购买商品的Top5余弦相似度
    val topRecommendations = similarity.filter { case (user_id, sku_id, cosineSimilarity) => data(user_id)(sku_id) == 0 }
      .groupBy(_._1)
      .mapValues(_.toSeq.sortBy(-_._3).take(5).map { case (_, sku_id, cosineSimilarity) => (sku_id, cosineSimilarity) })

    topRecommendations.take(10).foreach(println)

    spark.stop()
  }
}
