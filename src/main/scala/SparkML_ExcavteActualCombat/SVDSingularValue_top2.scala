package SparkML_ExcavteActualCombat

import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object SVDSingularValue_top2 {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()
    // 创建用户-商品数据，10个商品，0表示未购买，1表示已购买
    val data = Seq(
      (1, Vectors.dense(0, 1, 0, 1, 0, 0, 0, 0, 1, 0)),
      (2, Vectors.dense(1, 0, 0, 0, 1, 1, 0, 0, 0, 0)),
      (3, Vectors.dense(0, 0, 1, 0, 1, 0, 0, 0, 1, 0)),
      (4, Vectors.dense(0, 1, 0, 0, 1, 1, 0, 0, 1, 0)),
      (5, Vectors.dense(1, 0, 0, 0, 1, 1, 0, 0, 0, 0)),
      (6, Vectors.dense(1, 0, 0, 0, 1, 1, 0, 0, 0, 0))
      // 添加更多用户数据
    )

    val rows: RDD[(Int, Vector)] = spark.sparkContext.parallelize(data)

    // 创建一个 RowMatrix
    val matrix = new RowMatrix(rows.map(_._2))

    // 执行SVD分解
    val k = 3 // 保留前5个奇异值
    val svd = matrix.computeSVD(k, computeU = true)

    /**
     * U、V是两个正交矩阵，其中的每一行(每一列)分别被称为 左奇异向量(U) 和 右奇异向量(V)，
     * 他们和 ∑ 中对角线上的奇异值相对应，通常情况下我们只需要取一个较小的值 k，保留前k
     * 个奇异值向量和奇异值即可，其中:
     *
     * U 的维度是 m * k
     * V 的维度是 n * k
     * ∑ 是一个   k * k
     */

    /**
     * U矩阵是一个m×k的正交矩阵，其列向量被称为左奇异向量。
     * 左奇异向量与输入矩阵A的列向量（即数据样本）相关。
     * 在降维过程中,U矩阵的列向量表示输入数据的主要特征。
     * 降维后,我们可以使用U矩阵的列向量作为新的特征空间的基
     * 向量, 将原始数据投影到这个新的特征空间中。
     * 这样,我们就可以用k维数据表示原始数据的主要特征。
     *
     * V矩阵是一个n × k的正交矩阵，其列向量被称为右奇异向量。
     * 右奇异向量与输入矩阵A的行向量（即数据特征）相关。
     * 在降维过程中,V矩阵的列向量可以帮助我们重构输入数据的主要特征。
     * 虽然V矩阵本身不能直接用于降维后的数据分析和机器学习任务，但它可以在数据重构和特征提取方面发挥作用。
     */
    print(svd.V)
    import spark.implicits._
    val seq = svd.V.rowIter.toSeq
    spark.sparkContext.parallelize(seq).foreach(println)
    spark.stop()
  }
}
