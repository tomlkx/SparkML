package Learning

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, Dataset, Row, SaveMode, SparkSession}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.functions.col
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Matrices, Matrix, SingularValueDecomposition, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD

import java.util
import java.util.Properties


object ds_demo02_ckdx {
  def main(args: Array[String]): Unit = {

    val conf: SparkConf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("1235")

    val spark: SparkSession = SparkSession.builder().config(conf).enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    m1(spark)
    m2(spark)


    spark.close()
  }

  def m1(spark:SparkSession)={
    dqmysql(spark,"TB_DS","order_info").cache().createOrReplaceTempView("order_info")
    dqmysql(spark,"TB_DS","order_detail").cache().createOrReplaceTempView("order_detail")
    dqmysql(spark,"TB_DS","sku_info").cache().createOrReplaceTempView("sku_info")
    dqmysql(spark,"TB_DS","user_info").cache().createOrReplaceTempView("user_info")
    Seq("order_detail", "order_info", "sku_info","user_info").foreach(spark.table(_).show())
    spark.sql(
      """
        |
        |select
        |distinct i.user_id,d.sku_id
        |from
        |order_info i join order_detail d on i.id=d.order_id
        |join sku_info s on d.sku_id=s.id
        |join user_info u on i.user_id=u.id
        |""".stripMargin).createOrReplaceTempView("t1")


    val frame: DataFrame = spark.sql(
      """
        |
        |select
        |concat(user_id,':',sku_id) aa
        |from(
        |select
        |dense_rank() over(order by user_id) -1 user_id,
        |dense_rank() over(order by sku_id) -1 sku_id
        |from
        |t1
        |order by user_id,sku_id
        |)a
        |limit 5
        |""".stripMargin)
//    val properties = new Properties()
//    properties.setProperty("user", "default")
//    properties.setProperty("password", "123456")
//    properties.setProperty("driver","ru.yandex.clickhouse.ClickHouseDriver")
//
//    frame.write.format("jdbc").mode(SaveMode.Append)
//      .jdbc("jdbc:clickhouse://10.1.167.140:8123/default","aaaa",properties)
    println("----user_id_mapping与sku_id_mapping数据前5条如下:-----")
    frame.rdd.map(data=>data.getString(0)).collect().foreach(println)

  }


  def m2(spark:SparkSession) ={
  import spark.implicits._
    dqmysql(spark, "train_store", "order_info").createOrReplaceTempView("order_info")
    dqmysql(spark, "train_store", "order_detail").createOrReplaceTempView("order_detail")
    dqmysql(spark, "train_store", "sku_info").createOrReplaceTempView("sku_info")
    dqmysql(spark, "train_store", "user_info").createOrReplaceTempView("user_info")

    val sku_id_RDD: RDD[(Long, Long)] = spark.sql("select * from sku_info ").select("id").distinct().orderBy("id").rdd.map {
      case Row(id: Long) => id
    }.zipWithIndex().map(data => (data._2, data._1))

    spark.sql(
      """
        |
        |select
        |distinct i.user_id,d.sku_id
        |from
        |order_info i join order_detail d on i.id=d.order_id
        |join sku_info s on d.sku_id=s.id
        |join user_info u on i.user_id=u.id
        |""".stripMargin).createOrReplaceTempView("t1")

    val frame: DataFrame = spark.sql(
      """
        |
        |select
        |dense_rank() over(order by user_id) - 1 user_id,
        |dense_rank() over(order by sku_id) - 1 sku_id
        |from
        |t1
        |order by user_id,sku_id
        |""".stripMargin)

    val onehot_DF: Dataset[Row] = frame.groupBy("user_id").pivot("sku_id").count().na.fill(0).orderBy("user_id")
onehot_DF.show()
    val row: Row = onehot_DF.take(1)(0)
    val str: String = row.toSeq.take(5).toList.map(data => data.toString.toDouble).mkString(",")
    println("-----------第一行前5列结果展示为--------------")
    println(str)


    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(onehot_DF.columns.tail)
      .setOutputCol("features")

    val vectors: DataFrame = assembler.transform(onehot_DF).select(col("features").as("vector"))

    val transpose: Array[Array[Double]] = vectors.rdd.collect().map {
      case Row(features: org.apache.spark.ml.linalg.Vector) => features.toArray
    }.transpose

    val rows: Array[linalg.Vector] = transpose.map(data => Vectors.dense(data))

    val rowsRDD: RDD[linalg.Vector] = spark.sparkContext.parallelize(rows)

    val matrix = new RowMatrix(rowsRDD)

    val svd: SingularValueDecomposition[RowMatrix, Matrix] = matrix.computeSVD(5, true)
    val u: RowMatrix = svd.U
    val s: linalg.Vector = svd.s
    val v: Matrix = svd.V

    val array: Array[Double] = s.toArray
    val sNew: linalg.Vector = Vectors.dense(array)
    val SNEW: Matrix = Matrices.diag(sNew)

    val matrix1: RowMatrix = u.multiply(SNEW).multiply(v.transpose)

    val reult_RDD: RDD[(Long, linalg.Vector)] = matrix1.rows
      .zipWithIndex()
      .map(data => (data._2, data._1))
      .join(sku_id_RDD)
      .values
      .map(data => (data._2, data._1))


    val df1: Array[(Long, linalg.Vector)] = reult_RDD.filter(_._1 <= 5).collect()
    val df2: RDD[(Long, linalg.Vector)] = reult_RDD.filter(_._1 > 5)

    df1.foreach(println)
    df2.foreach(println)
    println("-------------------------")
    val AAAAA: RDD[(Long, Double)] = df2.map(name1 => {
        var sum = 0.0
        var count = 0
        for (name2 <- df1) {

          val cosin: Double = name2._2.dot(name1._2) / (Vectors.norm(name1._2, 2.0) * Vectors.norm(name2._2, 2.0))
          sum += cosin
          count += 1
        }
        println(name1._1,sum,count)
        (name1._1, sum / count)
      }
    )

    AAAAA.foreach(println)

    val tuples: Array[(Long, Double)] = AAAAA.sortBy(-_._2).take(5)


    var rn =1
    println("-------------推荐top结果如下---------------")
    for (elem <- tuples) {
      println(s"相似度top$rn(商品id: ${elem._1},平均相似度: ${elem._2})")
      rn+=1
    }

  }
  def dqmysql(spark:SparkSession,database:String,table:String) = {
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
