package SparkML_ExcavteActualCombat

import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix, SingularValueDecomposition, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkContext}


object Costim_t1 {
  def main(args: Array[String]): Unit = {
    def loadDataSet(): Array[Vector] = {
      Array(
        Vectors.dense(1, 1, 0, 1, 1),
        Vectors.dense(1, 0, 0, 0, 1),
        Vectors.dense(0, 0, 0, 1, 1),
        Vectors.dense(1, 1, 1, 1, 0),
        Vectors.dense(0, 1, 1, 0, 0),
        Vectors.dense(1, 1, 1, 0, 0),
        Vectors.dense(1, 1, 0, 0, 0)
      )
    }

    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("demo").getOrCreate()
    val sc: SparkContext = spark.sparkContext
    import spark.implicits._
    import org.apache.spark.sql.functions._
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
        var matrix2: RowMatrix = vx.U.multiply(vx.V)
        matrix2.rows.foreach(println)
        val list: List[Vector] = matrix2.rows.collect().toList
        for (j <- 0 to column - 1) {
          //读取地user行 的第j列
          val userRating: Double = matrix1(user, j)
          if (userRating != 0.0 && j != item._2) {
            println(s"迭代列j:${j} 物品列item:${item._2}")
            val items: List[(Double, Double)] = (0 until list(0).size).map(i => {
              (list(item._2)(i), list(j)(i))
            }).toBuffer.toList
            if (list.length != 0) {
              //用户未购买列的不为0的数据
              val v1: Vector = Vectors.dense(items.map(_._1).toArray)
              val v2: Vector = Vectors.dense(items.map(_._2).toArray)
              println(v1.toArray.toList)
              println(v2.toArray.toList)
              val x: Double = v1.dot(v2) / (Vectors.norm(v1, 2) * Vectors.norm(v2, 2))
              println(s"相似度:${x}")
              simTotal += x
              number + 1
            }
          }
        }
        println(s"平均相似度:${simTotal / number}")
      }
    }
    spark.stop()
  }
}
