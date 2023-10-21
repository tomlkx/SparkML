package SparkML_ExcavteActualCombat

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.sql.SparkSession

object MovieRecommendation {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .config("spark.sql.warehouse.dir", "hdfs://bigdata1:9000/user/hive/warehouse")
      .config("hive.metastore.uris", "thrift://bigdata1:9083")
      .config("spark.sql.pivotMaxValues", "15000")
      .enableHiveSupport()
      .appName("demo02")
      .master("local[*]")
      .getOrCreate()
    val sc = spark.sparkContext

    // 创建用户-电影评分矩阵，这是一个虚拟的矩阵，实际中应使用真实数据
    val ratings = Seq(
      Vectors.dense(5.0, 4.0, 0.0, 0.0, 0.0),
      Vectors.dense(4.0, 5.0, 0.0, 0.0, 0.0),
      Vectors.dense(0.0, 0.0, 5.0, 4.0, 0.0),
      Vectors.dense(0.0, 0.0, 4.0, 5.0, 0.0),
      Vectors.dense(0.0, 0.0, 0.0, 0.0, 5.0)
    )

    val rows = sc.parallelize(ratings)

    // 创建一个 RowMatrix
    val matrix = new RowMatrix(rows)

    // 执行SVD分解
    val k = 2 // 设置保留的奇异值数量
    val svd = matrix.computeSVD(k, computeU = true)

    // 获取U、S和V矩阵
    val U: RowMatrix = svd.U
    val S: Vector = svd.s
    val V = svd.V
    U.rows.foreach(println)
    println("----------")
    println(S)
    println("---------")
    println(V)
    println("----------")
    println(V.rowIter.size)
    val vector: Vector = V.rowIter.toList(3)
    val doubles: List[Double] = V.rowIter.toList.take(4).map(x => {
      x.dot(vector) / (Vectors.norm(x, 2) * Vectors.norm(vector, 2))
    })
    doubles.foreach(println)
    sc.stop()
  }
}
