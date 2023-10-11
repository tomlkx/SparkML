package SparkML_Learn

import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.sql.SparkSession

object LearnDemo34_SVD {
  def main(args: Array[String]): Unit = {
    // 创建一个 Spark 会话
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()
    // 示例数据：向量数组
    val data = Array(
      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    )
    // 将数据转换为向量的 RDD
    val rows = spark.sparkContext.parallelize(data)
    // 从 RDD 创建一个 RowMatrix
    val matrix = new RowMatrix(rows)
    // 计算 RowMatrix 的奇异值分解（SVD）
    val svd: SingularValueDecomposition[RowMatrix, Matrix] = matrix.computeSVD(5, computeU = true)
    // U 矩阵：左奇异向量（列表示转换后的特征）
    val U = svd.U
    // s 向量：奇异值（奇异值构成的对角矩阵）
    val s = svd.s
    // V 矩阵：右奇异向量（行表示原始特征）
    val V = svd.V
    // 打印 U 矩阵的行（左奇异向量）
    U.rows.foreach(println)

    // 打印奇异值
    println(s"奇异值为：$s")

    // 打印 V 矩阵（右奇异向量）
    println(s"V 矩阵为：$V")

    // 停止 Spark 会话
    spark.stop()
  }
}
