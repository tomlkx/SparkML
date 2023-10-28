package SparkML_ExcavteActualCombat

import org.apache.spark.ml.feature.{StandardScaler, StandardScalerModel, VectorAssembler}
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import java.util.Properties
import scala.collection.mutable

object demo01_02 {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[8]").getOrCreate()
    val pro = new Properties()
    pro.setProperty("user", "root")
    pro.setProperty("password", "123456")
    spark.sparkContext.setLogLevel("ERROR")
    import org.apache.spark.sql.functions._
    import spark.implicits._
    spark.read.jdbc("jdbc:mysql://127.0.0.1:3306/TestData", "sku_info", pro).createTempView("sku_info_xd")
    var sku_info_data = spark.table("sku_info_xd").select("id", "spu_id", "price", "weight", "tm_id", "category3_id").withColumn("price", 'price.cast("Double"))
    //组合向量
    val va: VectorAssembler = new VectorAssembler()
      .setInputCols(Array("price", "weight"))
      .setOutputCol("price_weight_feature")
    sku_info_data = va.transform(sku_info_data)
    //归一化
    val stan_price: StandardScaler = new StandardScaler()
      .setInputCol("price_weight_feature")
      .setOutputCol("stan_price_weight")
    //自定义UDF转换
    spark.udf.register("xxx", (x: DenseVector, i: Int) => {
      x.values(i)
    })
    val model1: StandardScalerModel = stan_price.fit(sku_info_data)
    sku_info_data = model1.transform(sku_info_data)
    //获取归一化后的值
    sku_info_data = sku_info_data.withColumn("price", expr("xxx(stan_price_weight,0)"))
    sku_info_data = sku_info_data.withColumn("weight", expr("xxx(stan_price_weight,1)"))
    //剔除无用列
    sku_info_data = sku_info_data.drop("price_weight_feature", "stan_price_weight")
    //创建虚拟变量（哑变量）
    val spuIdSpaces = sku_info_data.select("spu_id").distinct().sort("spu_id").rdd.map(_(0)).collect()
    val tmIdSpaces = sku_info_data.select("tm_id").distinct().sort("tm_id").rdd.map(_(0)).collect()
    val category3IdSpaces = sku_info_data.select("category3_id").distinct().sort("category3_id").rdd.map(_(0)).collect()
    val data = Seq((spuIdSpaces, "spu_id"), (tmIdSpaces, "tm_id"), (category3IdSpaces, "category3_id"))
    val buffer = List[String]().toBuffer
    buffer.append("id", "price", "weight")

    for (items <- data) {
      for (item <- items._1) {
        buffer.append(s"${items._2}#${item}")
        sku_info_data = sku_info_data.withColumn(s"${items._2}#${item}", when(col(s"${items._2}") === item, 1.0).otherwise(0.0))
      }
    }
    sku_info_data.show()
    //第一条数据的前10列
    println(sku_info_data.drop("tm_id", "category3_id", "spu_id").withColumn("id", 'id * 1.0).rdd.take(1)(0).toSeq.take(10).mkString(","))
    //组合向量
    val vs: VectorAssembler = new VectorAssembler()
      .setInputCols(buffer.toArray)
      .setOutputCol("features")
    sku_info_data = vs.transform(sku_info_data)
    //读取数据
    spark.read.jdbc("jdbc:mysql://127.0.0.1:3306/TestData", "order_info", pro).createTempView("order_info")
    spark.read.jdbc("jdbc:mysql://127.0.0.1:3306/TestData", "order_detail", pro).createTempView("order_detail")
    val doData = spark.table("order_info").join(spark.table("order_detail").drop("id"), 'order_id === 'id)
    //6708的所所有订单
    val order_6708 = spark.sql("select sku_id  from order_info f inner join order_detail z on z.order_id=f.id where f.user_id=6708")
    val top10_user_id = spark.sql(
      s"""
         |select user_id,count(1) count from(
         |  select user_id,sku_id from order_info a inner join order_detail b on b.order_id=a.id where b.sku_id in(
         |     ${order_6708.collect().map(_(0)).mkString(",")}
         |  ) group by user_id,sku_id
         |)t where user_id != 6708 group by user_id order by user_id limit 10
         |""".stripMargin)
    top10_user_id.show()
    // 找到这十个人的所有订单
    var top10_sku: DataFrame = doData
      //剔除6708的已经购买的商品
      .filter(!'sku_id.isin(order_6708.collect().map(_(0)).toSeq: _*))
      //查找前十个用户的订单
      .where('user_id.isin(top10_user_id.select("user_id").collect().map(_(0).asInstanceOf[Long]).toSeq: _*))
      //剔除前十个用户重复商品
      .groupBy('sku_id)
      .agg(count(lit(1)) as "count")
      .join(sku_info_data)
      .where('sku_id === 'id)
      .select("id", "features")
    // 初始化累加值 和 平均值
    top10_sku = top10_sku.withColumn("sums", lit(0.0)).withColumn("avgs", lit(0.0))
    //注册udf 夹角余弦相似度计算
    spark.udf.register("xxxx", (x1: Vector, x2: Vector) => {
      //点积运算
      val v1 = x1.dot(x2)
      //
      val v2 = Vectors.norm(x1, 2L) * Vectors.norm(x2, 2L)
      v1 / v2
    })
    //获取6708的商品列表 并剔除重复
    val rows: Array[Row] = sku_info_data.filter('id.isin(order_6708.collect().map(_(0)).toSeq: _*)).select("id", "features").distinct().collect()
    //记录
    val rowBuffer: mutable.Buffer[String] = List[String]().toBuffer
    for (row <- rows) {
      rowBuffer.append(s"sku_id#${row(0)}")
      top10_sku =
        top10_sku
          //插入6708的商品
          .withColumn(s"sku_id_${row(0)}", typedLit(row(1).asInstanceOf[Vector]))
          //计算相似度
          .withColumn(s"sku_id_${row(0)}", expr(s"xxxx(features,sku_id_${row(0)})"))
          //累加相似度
          .withColumn("sums", col(s"sku_id_${row(0)}").cast("double") + 'sums)
          //计算平均值
          .withColumn("avgs", ('sums / rowBuffer.length).cast("double"))
    }

    top10_sku.orderBy('avgs.desc).select('id, 'avgs).take(5).zipWithIndex.map(
      x => s"相似度top${x._2 + 1}(商品id：${x._1(0)}，平均相似度：${x._1(1)})"
    ).foreach(println)
    spark.stop()
  }
}
