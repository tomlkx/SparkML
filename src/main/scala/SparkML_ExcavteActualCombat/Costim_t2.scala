package SparkML_ExcavteActualCombat

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix, SingularValueDecomposition, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession


object Costim_t2 {
  def main(args: Array[String]): Unit = {
    def loadDataSet(): Array[Vector] = {
      Array(
        Vectors.dense(1, 0, 0, 1, 0),
        Vectors.dense(1, 0, 0, 0, 0),
        Vectors.dense(1, 0, 0, 1, 1),
        Vectors.dense(1, 1, 1, 0, 0),
        Vectors.dense(1, 1, 1, 0, 0),
        Vectors.dense(1, 1, 1, 0, 0),
        Vectors.dense(0, 1, 0, 0, 0)
      )
    }

    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("demo").getOrCreate()
    val sc: SparkContext = spark.sparkContext
    val value: RDD[Vector] = sc.parallelize(loadDataSet())
    val matrix = new RowMatrix(value)
    val vx: SingularValueDecomposition[RowMatrix, Matrix] = matrix.computeSVD(5, computeU = true)
    print("---------------重构原始矩阵dataMat------------")
    val s: Vector = vx.s
    val u: Array[Vector] = vx.U.rows.collect()
    val v: List[Vector] = vx.V.colIter.toList
    val doubles = loadDataSet().map(_.toArray).zipWithIndex.flatMap(line => line._1)
    //读取时 矩阵不进行转换
    val matrix1 = new DenseMatrix(7, 5, doubles, true)
    //输出矩阵
    matrix1.rowIter.foreach(println)
    //指定用户
    val user = 2
    //寻找未评分的物品
    val unratedItems: List[(Double, Int)] = matrix1.rowIter.toList(user).toArray.zipWithIndex.filter(_._1 == 0.0).toList
    println(unratedItems.toList)
    if (unratedItems.length > 0) {
      val itemScores = Seq[Double]()
      for (item <- unratedItems) {
        //获取矩阵列数
        val column: Int = matrix1.numCols
        var simTotal = 0.0
        var number = 0
        for (j <- 0 to column - 1) {
          //读取地user行 的第j列
          val userRating: Double = matrix1(user, j)
          if (userRating != 0.0 && j != item._2) {
            println(s"迭代列j:${j} 物品列item:${item._2}")
            val list: List[(Double, (Int, Int), Double, (Int, Int))] = (0 until matrix1.numRows).filter(i => {
              matrix1(i, j) > 0 && matrix1(i, item._2) > 0
            }).map(i => {
              (matrix1(i, j), (i, j), matrix1(i, item._2), (i, item._2))
            }).toBuffer.toList
            if (list.length != 0) {
              val v1: Vector = Vectors.dense(list.map(_._1).toArray)
              val v2: Vector = Vectors.dense(list.map(_._3).toArray)
              println(v1.toArray.toList)
              println(v2.toArray.toList)
              val x: Double = 0.5 * (v1.dot(v2) / (Vectors.norm(v1, 2) * Vectors.norm(v2, 2)))
              simTotal += x
              number += 1
            }
          }
        }
        println(s"平均相似度:${simTotal / number}")
      }
    }
    spark.stop()
  }
}
