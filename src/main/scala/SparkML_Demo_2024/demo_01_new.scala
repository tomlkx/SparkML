package SparkML_Demo_2024

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, VectorAssembler}
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.sql.SparkSession

import java.util.Properties

object demo_01_new {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .master("local[*]")
      .appName("demo")
      .config("spark.driver.memory","259522560")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    import org.apache.spark.sql.functions._
    import spark.implicits._
    /**
     * 0.准备工作
     * 设置IP、数据库配置
     * 表数据进行缓存到内存中 加快运算速度(避免反复去数据库读)
     *
     */
    val ip = "LIUKAIXIN"
    val port = 4306
    val database = "TB_DS"
    val pro = new Properties()
    pro.setProperty("user","root")
    pro.setProperty("password","123456")
    Array("order_detail","order_info","sku_info","user_info").foreach(table=>{
      // 读取表并创建Spark 表临时视图
      spark.read.jdbc(s"jdbc:mysql://${ip}:${port}/${database}",table,pro).cache().createOrReplaceTempView(table)
      // show 一下表对数据进行缓存
      spark.table(table).show()
    })
    /** *
     * 1、根据Hive的dwd库中相关表或MySQL中ds_db01中相关表，计算出与用户customer_id为5811
     * 的用户所购买相同商品种类最多的前10位用户（只考虑他俩购买过多少种相同的商品，不考虑相同的商品买了多少次），
     * 将10位用户customer_id进行输出，输出格式如下
     *
     * -------------------相同种类前10的id结果展示为：--------------------
     * 1,2,901,4,5,21,32,91,14,52
     */
    val user_id = 6708
    // 找到用户5811 购买的所有商品 并剔除重复
    val user_sku_items: Array[Int] = spark.sql(s"select a.sku_id from order_detail a inner join order_info b on a.order_id=b.id where b.user_id = ${user_id}").distinct().collect().map(_(0).toString.toInt)
    // 构建广播变量减少系统资源开销 如果 5811 没有数据自己模拟数据进行测试
    //val broad_user_5811_sku_items: Array[Int] = spark.sparkContext.broadcast(user_sku_items).value
    val broad_user_5811_sku_items: Array[Int] = spark.sparkContext.broadcast(Array(1,2,3,4,5,6,7)).value
    // 自定义UDF函数用于计算与5811用户购买商品的相似度
    spark.udf.register("jj",(x:Array[Int])=>{
      broad_user_5811_sku_items.intersect(x).length / (broad_user_5811_sku_items.length * 1.0)
    })
    // 构建查询
    spark.sql(
      """
        |select
        |     b.user_id,a.sku_id
        |from
        |     order_detail a
        |inner join
        |     order_info b
        |on a.order_id=b.id
        |""".stripMargin)
      .groupBy("user_id")
      .agg(collect_set("sku_id") as "user_arrays_sku")
      .filter('user_id =!= user_id)
      .withColumn("similarity",expr("jj(user_arrays_sku)"))
      .createOrReplaceTempView("task_01")
    // spark.table("task_01").limit(50).show(50,truncate = false)
    // 取出十位用户的sku_id 并剔除重复
    val top10_sku_Id: Array[Int] = spark.table("task_01")
      .orderBy('similarity.desc)
      .limit(10)
      .withColumn("sku_id", explode('user_arrays_sku))
      .select("sku_id")
      .distinct()
      .collect()
      .map(_(0).toString.toInt)
    // 打印第一题结果
    println("-------------------相同种类前10的id结果展示为：--------------------")
    println(spark.table("task_01").orderBy('similarity.desc).take(10).map(_(0).toString.toInt).mkString(","))
    spark.table("task_01").orderBy('similarity.desc).show(truncate = false)
    /** *
     * 2、根据Hive的dwd库中相关表或MySQL中ds_db01中相关商品表（sku_info），
     * 获取id、brand_id、price、weight、height、length、width、three_category_id
     * 字段并进行数据预处理，对数值类型进行规范化(StandardScaler)处理，对类别类型进行
     * one-hot编码处理（若该商品属于该品牌则置为1，否则置为0）,并按照id进行升序排序，
     * 在集群中输出第一条数据前10列（无需展示字段名）。
     *
     * --------------------第一条数据前10列结果展示为：---------------------
     * 1.0,0.892346,1.72568,0.0,0.0,0.0,0.0,1.0,0.0,0.0
     */
    val pwva: VectorAssembler = new VectorAssembler()
      .setInputCols(Array("price", "weight"))
      .setOutputCol("pwva")
    val scaler: StandardScaler = new StandardScaler()
      .setInputCol("pwva")
      .setOutputCol("scpwva")
      .setWithStd(true)
      .setWithMean(false)
    val hotEncoder: OneHotEncoder = new OneHotEncoder()
      .setInputCols(Array("spu_id", "tm_id", "category3_id"))
      .setOutputCols(Array("spu_id_oh","tm_id_oh","category_id_oh"))
    val vsfeature: VectorAssembler = new VectorAssembler()
      .setInputCols(Array("scpwva", "spu_id_oh", "tm_id_oh", "category_id_oh"))
      .setOutputCol("feature")
    val pipeline: Pipeline = new Pipeline().setStages(Array(pwva, scaler, hotEncoder,vsfeature))
    pipeline
      .fit(spark.table("sku_info"))
      .transform(spark.table("sku_info"))
      .cache()
      .createOrReplaceTempView("task_02")
    spark.table("task_02").show()
    println("--------------------第一条数据前10列结果展示为：---------------------")
    spark.table("task_02").orderBy('id).limit(1).map(line=>{
      f"${line(0).toString.toInt * 1.0},${line(11).asInstanceOf[Vector].toArray(0)},${line(11).asInstanceOf[Vector].toArray(1)},${line(12).asInstanceOf[Vector].toArray.take(7).mkString(",")}"
    }).foreach(println(_))
    /** *
     * 3、根据上述任务的结果，计算出与用户customer_id为5811的用户所购买相同商品种类最多的前10位用户id
     * （只考虑他俩购买过多少种相同的商品，不考虑相同的商品买了多少次），并根据Hive的dwd库中相关表或MySQL
     * 数据库shtd_store中相关表，获取到这10位用户已购买过的商品，并剔除用户5811已购买的商品，通过计算这
     * 10位用户已购买的商品（剔除用户5811已购买的商品）与用户5811已购买的商品数据集中商品的余弦相似度累
     * 加再求均值，输出均值前5商品id作为推荐使用。
     */
    spark.udf.register("jjyx",(v1:Vector,v2:Vector)=>{
      val x1: Double = v1.dot(v2)
      val x2: Double = Vectors.norm(v1,2L) * Vectors.norm(v2,2L)
      x1 / x2
    })
    println("------------------------推荐Top5结果如下------------------------")
    spark.table("task_02")
      .select('feature as "feature_user",'id as "sku_id_user")
      .filter('id.isin(broad_user_5811_sku_items: _*))
      .crossJoin(
        spark.table("task_02").filter(!'id.isin(broad_user_5811_sku_items: _*)).filter('id.isin(top10_sku_Id:_*)).distinct()
      )
      .withColumn("label",concat_ws("-",'sku_id_user,'id))
      .withColumn("相似度",expr("jjyx(feature_user,feature)"))
      .groupBy("label")
      .agg(avg("相似度") as "均值相似度")
      .withColumn("label",split('label,"-")(1))
      .groupBy('label)
      .agg(max('均值相似度) as "均值相似度")
      .orderBy('均值相似度.desc)
      .collect()
      .zipWithIndex
      .take(5)
      .foreach(x=>{
        println(s"相似度top${x._2 + 1}(商品id：${x._1(0)}，平均相似度：${x._1(1)})")
      })
    spark.stop()
  }
}
