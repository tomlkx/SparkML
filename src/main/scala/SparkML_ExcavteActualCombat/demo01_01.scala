package SparkML_ExcavteActualCombat

import org.apache.spark.ml.linalg
import org.apache.spark.ml.feature.{StandardScaler, StandardScalerModel, VectorAssembler}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.linalg.{Vectors => SVectors}

import java.util.Properties

object demo01_01 {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[8]").getOrCreate()
    val pro = new Properties()
    pro.setProperty("user", "root")
    pro.setProperty("password", "123456")
    import org.apache.spark.sql.functions._
    import spark.implicits._
    spark.read.jdbc("jdbc:mysql://bigdata1:3306/ds_db01", "order_master", pro).createTempView("order_master")
    spark.read.jdbc("jdbc:mysql://bigdata1:3306/ds_db01", "order_detail", pro).createTempView("order_detail")
    spark.read.jdbc("jdbc:mysql://bigdata1:3306/ds_db01", "product_info", pro).createTempView("product_info")
    spark.sql(
      """
        |select counts,customer_id
        |from (select count(1) as counts, customer_id
        |      from (select distinct b.customer_id, a.product_id
        |            from (select *
        |                  from order_detail
        |                  where product_id in (select distinct product_id
        |                                       from order_detail
        |                                       where order_sn in
        |                                             (select distinct order_master.order_sn
        |                                              from order_master
        |                                              where customer_id = 5811))) as a
        |                     inner join order_master b on a.order_sn = b.order_sn) c
        |      where customer_id != 5811
        |      group by customer_id
        |      order by counts desc
        |      limit 10) e
        |""".stripMargin)
      .createTempView("dataView1");
    spark.table("dataView1").select("customer_id").take(10).map(_(0)).mkString(",")
    spark.sql("select distinct b.product_id from (select * from order_master where customer_id in(5811)) a inner join order_detail as b on a.order_sn=b.order_sn").createTempView("dataView2")
    spark.sql(
      """
        |select distinct b.product_id from (select * from order_master where customer_id in(select customer_id from dataView1)) a inner join order_detail as b on a.order_sn=b.order_sn
        |where b.product_id not in(
        |select * from dataView2
        |)
        |""".stripMargin)
      .createTempView("dataView3")
    //特征组合
    val vs: VectorAssembler = new VectorAssembler()
      .setInputCols(Array("product_id", "brand_id", "price", "weight", "height", "length", "width"))
      .setOutputCol("features")
    val ss: StandardScaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("StanFeatures")
    val data2: DataFrame = vs.transform(spark.table("product_info"))
    data2.show()
    val model: StandardScalerModel = ss.fit(data2)
    //特征提取
    model.transform(data2).createTempView("dataView4")
    //获取与5811 购买相似商品的前是个用户所购买的商品列表
    val r1: Array[(Long, linalg.Vector)] = spark.table("dataView3").join(spark.table("dataView4"), Seq("product_id"))
      .select("product_id", "StanFeatures")
      .collect()
      .map {
        case Row(product_id: Long, stanFeatures: linalg.Vector) =>
          (product_id, stanFeatures)
      }
    //获取5811 购买的商品
    val r2: Array[(Long, linalg.Vector)] = spark.table("dataView2").join(spark.table("dataView4"), Seq("product_id"))
      .select("product_id", "StanFeatures")
      .collect()
      .map {
        case Row(product_id: Long, stanFeatures: linalg.Vector) =>
          (product_id, stanFeatures)
      }
    //存储结果
    var resultList: List[(Long, Double)] = List()
    //计算相似度
    for (x <- r1) {
      var count = 0
      var Similaritys: Double = 0
      for (i <- r2) {
        val Similarity: Double = cosineSimilarity(x._2, i._2)
        count = count + 1
        Similaritys = Similaritys + Similarity
      }
      resultList = resultList :+ (x._1, Similaritys / count)
    }
    resultList.sortBy(-_._2).foreach(println)
    spark.stop()
  }

  def cosineSimilarity(v1: linalg.Vector, v2: linalg.Vector): Double = {
    //点积运算
    val dotProduct = v1.dot(v2)
    //使用欧几里得范数标准化规范化  计算两个向量模的乘积
    val normProduct = SVectors.norm(v1, 2) * SVectors.norm(v2, 2)

    /**
     * 将点积除以两个向量的 L2 范数的乘积，得到夹角余弦相似度。夹角余弦相似度是一个介于 -1 和 1 之间的值，
     * 表示了两个向量之间的相似程度。值越接近 1，表示两个向量越相似；值越接近 -1，表示两个向量越相反；
     * 值接近 0，表示两个向量之间没有明显的相关性。
     */
    dotProduct / normProduct
  }
}