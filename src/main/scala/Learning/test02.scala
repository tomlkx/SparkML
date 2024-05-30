package Learning

import org.apache.spark.{SparkConf, mllib}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.mllib.linalg.{Matrices, Matrix, SingularValueDecomposition, Vectors}

import java.util.Properties
object test02 {
  def main(args: Array[String]): Unit = {

    val conf: SparkConf = new SparkConf()
      .setAppName("zbjs")
      .setMaster("local[*]")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.sql.storeAssignmentPolicy", "LEGACY")


    val spark: SparkSession = SparkSession.builder().config(conf).enableHiveSupport().getOrCreate()

    import spark.implicits._


    read_mysql(spark, "shtd_store_1", "order_info").createOrReplaceTempView("order_info")
    read_mysql(spark, "shtd_store_1", "order_detail").createOrReplaceTempView("order_detail")
    read_mysql(spark, "shtd_store_1", "sku_info").createOrReplaceTempView("sku_info")
    read_mysql(spark, "shtd_store_1", "user_info").createOrReplaceTempView("user_info")

    val sku_index: RDD[(Long, Long)] = read_mysql(spark, "shtd_store_1", "sku_info")
      .select("id")
      .distinct()
      .orderBy("id")
      .rdd
      .map {
        case Row(id: Long) => id
      }.zipWithIndex()
      .map(li => (li._1, li._2))


    spark.sql(
      """
        |
        |select
        |distinct i.user_id,d.sku_id
        |from
        |order_info i join order_detail d on d.order_id=i.id
        |join sku_info s on d.sku_id=s.id
        |join user_info u on i.user_id=u.id
        |
        |""".stripMargin).createOrReplaceTempView("t1")

    val dataframe: DataFrame = spark.sql(
      """
        |
        |select
        |dense_rank() over(order by user_id) user_id,
        |dense_rank() over(order by sku_id) sku_id
        |from
        |t1
        |order by user_id,sku_id
        |""".stripMargin)

    //创建用户-物品 矩阵 0 1 0 1 0 0 0 ---第一列为用户索引,其余列为是否购买
    val one_hot: Dataset[Row] = dataframe.groupBy("user_id").pivot("sku_id").count().na.fill(0).orderBy("user_id")
    one_hot.show()
    println("---------------第一行前5列结果展示为---------------")
    println(one_hot.select(one_hot.columns.take(5).map(col): _*).rdd.map(row => row.toSeq).take(1)(0).mkString(","))


    // 将每个物品的购买信息转换为向量,这里还是把特征列进行转换成向量
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(one_hot.columns.tail)
      .setOutputCol("features")

    val vetors: DataFrame = assembler.transform(one_hot).select(col("features").as("vector"))


    //矩阵转置, 把用户-物品 矩阵,转换成 物品-用户矩阵,用于计算物品相似度,转换为ml的Vectors
    val rowtranspose: Array[Array[Double]] = vetors.rdd.collect().map {
      case Row(features: linalg.Vector) => features.toArray
    }.transpose


    //转换成rdd的mllib的Vetors,(RowMatrix是mllib包下) 用于SVD分解,并保留5个奇异值
    val rows: Array[mllib.linalg.Vector] = rowtranspose.map(dta => Vectors.dense(dta))
    val rowRDD: RDD[mllib.linalg.Vector] = spark.sparkContext.parallelize(rows)



    //创建一个行矩阵
    val matrix = new RowMatrix(rowRDD)

    //进行SVD分解,并保留5个奇异值
    val svd: SingularValueDecomposition[RowMatrix, Matrix] = matrix.computeSVD(5, true)
    val u: RowMatrix = svd.U
    val s: mllib.linalg.Vector = svd.s
    val v: Matrix = svd.V

    //计算降维后新矩阵,并创建DF
    val sArr: Array[Double] = s.toArray
    val sNew: mllib.linalg.Vector = Vectors.dense(sArr)
    val Snew: Matrix = Matrices.diag(sNew)

    val matrixNew: RowMatrix = u.multiply(Snew).multiply(v.transpose)

    val commidty: RDD[(Long, mllib.linalg.Vector)] = matrixNew.rows.zipWithIndex()
      .map(li => (li._2, li._1))
      //连接用户
      .join(sku_index)
      .values
      .map(data => (data._2, data._1))

    val df1: Array[(Long, mllib.linalg.Vector)] = commidty.filter(_._1 <= 5).collect()
    val df2: RDD[(Long, mllib.linalg.Vector)] = commidty.filter(_._1 > 5)

    val result: Array[(Long, Double)] = df2.map(
      li => {
        var sum = 0.0
        var count = 0
        for (i <- df1) {
          //使用api计算余弦相似度,两个向量的点积/两个向量的L2范数的乘积,使用BLAS必须package org.apache.spark , BLAS.dot()

          val cosin: Double = i._2.dot(li._2) / (Vectors.norm(i._2, 2.0) * Vectors.norm(li._2, 2.0))
          sum += cosin
          count += 1

        }
        (li._1, sum / count)
      }
    )
      .sortBy(_._2, false)
      .take(5)


    var top=1

    println("------------------------推荐Top5结果如下------------------------")
    for (elem <- result) {
      println(s"相似度top$top(商品id: ${elem._1},平均相似度: ${elem._2})")
      top+=1
    }



    spark.close()
  }

  def read_mysql(spark:SparkSession,database:String,table:String) = {
    val pro = new Properties()
    pro.setProperty("user","root")
    pro.setProperty("password","123456")
    spark.read.jdbc(
      s"jdbc:mysql://LIUKAIXIN:4306/${database}",
      s"${table}",
      pro
    )
  }
}
