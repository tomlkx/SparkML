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
      (1,Vectors.dense(5.0, 4.0, 0.0, 0.0, 0.0)),
      (2,Vectors.dense(4.0, 5.0, 0.0, 0.0, 0.0)),
      (5,Vectors.dense(0.0, 0.0, 5.0, 4.0, 0.0)),
      (3,Vectors.dense(0.0, 0.0, 4.0, 5.0, 0.0)),
      (4,Vectors.dense(0.0, 0.0, 0.0, 0.0, 5.0))
    )
    val rows = sc.parallelize(ratings)
    sc.stop()
  }
}
