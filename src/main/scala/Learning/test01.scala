package Learning

import org.apache.spark.SparkConf
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import java.util.Properties

object test01 {
  def main(args: Array[String]): Unit = {

    val conf: SparkConf = new SparkConf()
      .setAppName("zbjs")
      .setMaster("local[*]")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.sql.storeAssignmentPolicy", "LEGACY")

    val spark: SparkSession = SparkSession.builder().config(conf).enableHiveSupport().getOrCreate()


    read_mysql(spark,"shtd_store_1","order_info").createOrReplaceTempView("order_info")
    read_mysql(spark,"shtd_store_1","order_detail").createOrReplaceTempView("order_detail")
    read_mysql(spark,"shtd_store_1","sku_info").createOrReplaceTempView("sku_info")
    read_mysql(spark,"shtd_store_1","user_info").createOrReplaceTempView("user_info")


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


    val datafr: DataFrame = spark.sql(
      """
        |
        |select
        |concat(user_id,':',sku_id)
        |from(
        |select
        |dense_rank() over(order by user_id) -1 user_id,
        |dense_rank() over(order by sku_id) -1 sku_id
        |from t1
        |order by user_id,sku_id
        |)a
        |limit 5
        |
        |""".stripMargin)




    val strings: Array[String] = datafr.rdd.collect().map(_.getString(0))
    println("-------user_id_mapping与sku_id_mapping数据前5条如下：-------")
    strings.foreach(println)

//    val one_hot: Dataset[Row] = frame.groupBy("user_id").pivot("sku_id").count().na.fill(0).orderBy("user_id")


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
