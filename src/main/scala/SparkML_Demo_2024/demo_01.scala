package SparkML_Demo_2024

import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.collection.mutable

/***
 * 自己做的数据  模拟写的
 */
object demo_01 {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    val order_collection: DataFrame = spark.read.option("header", true).option("charset", "gbk").csv("C:\\Users\\LiuKaixin\\Documents\\GitHub\\MyProjectList\\SparkML基于用户用户和用户购买商品相似度进行推荐\\商品集合.csv")
    val user_collection: DataFrame = spark.read.option("header", true).option("charset", "gbk").csv("C:\\Users\\LiuKaixin\\Documents\\GitHub\\MyProjectList\\SparkML基于用户用户和用户购买商品相似度进行推荐\\用户购买集合.csv")
    import spark.implicits._
    import org.apache.spark.sql.functions._
    import org.apache.spark.sql.types._
    /** *
     * 1、根据Hive的dwd库中相关表或MySQL中ds_db01中相关表，计算出与用户customer_id为5811
     * 的用户所购买相同商品种类最多的前10位用户（只考虑他俩购买过多少种相同的商品，不考虑相同的商品买了多少次），
     * 将10位用户customer_id进行输出，输出格式如下
     *
     * -------------------相同种类前10的id结果展示为：--------------------
     * 1,2,901,4,5,21,32,91,14,52
     */
    val user_id = 2
    val user_coll: DataFrame = user_collection
      .groupBy("用户id")
      .agg(collect_set('商品id.cast(IntegerType)) as "order_id_coll")
    val to1coll: mutable.WrappedArray[Int] = user_coll.filter('用户id === user_id).select("order_id_coll").take(1)(0)(0).asInstanceOf[mutable.WrappedArray[Int]]
    println(s"打印用户${user_id}购买的商品列表:${to1coll.mkString(",")}")
    //构建广播集合 减少系统资源开销
    val bc_to1coll: mutable.WrappedArray[Int] = spark.sparkContext.broadcast(to1coll).value
    spark.udf.register("coll", (mu: mutable.WrappedArray[Int]) => {
      (bc_to1coll.toSet & mu.toSet).size / (bc_to1coll.length * 1.0)
    })
    val xs_date: Dataset[Row] = user_coll
      .filter('用户id =!= user_id)
      .withColumn("相似度", expr("coll(order_id_coll)"))
      .orderBy('相似度.desc)
    println("-------------------相同种类前10的id结果展示为：--------------------")
    println(xs_date.select("用户id").take(10).map(_(0).toString).mkString(","))

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
    //取出所有的品牌id并剔除重复排序
    val ppids: Array[Int] = order_collection.select('品牌id.cast(IntegerType)).distinct().orderBy('品牌id).collect().map(_(0).asInstanceOf[Int])
    println(s"所有品牌id集合:${ppids.mkString(",")}")
    val order_one_hou: DataFrame = order_collection
      .withColumn("品牌id", concat(lit("ppid"), '品牌id))
      .groupBy(
        '主键.cast(DoubleType) as "t1",
        '价格.cast(DoubleType) as "t2",
        '重量.cast(DoubleType) as "t3",
        '高度.cast(DoubleType) as "t4",
        '宽度.cast(DoubleType) as "t5",
        '长度.cast(DoubleType) as "t6"
      ).pivot('品牌id).agg(first(lit(1.0))).na.fill(0.0)
    val as: VectorAssembler = new VectorAssembler()
      .setInputCols(Array("t2", "t3"))
      .setOutputCol("vsDate")
    val as_order_one_hou = as.transform(order_one_hou);
    val sc: StandardScaler = new StandardScaler()
      .setInputCol("vsDate")
      .setOutputCol("vsDate_sc")
    spark.udf.register("xxx", (x: DenseVector, i: Int) => {
      x.values(i)
    })
    sc.fit(as_order_one_hou)
      .transform(as_order_one_hou)
      .withColumn("t2", expr("xxx(vsDate_sc,0)"))
      .withColumn("t3", expr("xxx(vsDate_sc,1)"))
      .orderBy('t1)
      .createTempView("table1")
    println("--------------------第一条数据前10列结果展示为：---------------------")
    println(
      spark.sql(
        f"""
           |select t1,t2,t3,${ppids.map(x => "ppid" + x).take(7).mkString(",")} from table1 limit 1
           |""".stripMargin).take(1)(0).mkString(",")
    )

    /** *
     * 3、根据上述任务的结果，计算出与用户customer_id为5811的用户所购买相同商品种类最多的前10位用户id
     * （只考虑他俩购买过多少种相同的商品，不考虑相同的商品买了多少次），并根据Hive的dwd库中相关表或MySQL
     * 数据库shtd_store中相关表，获取到这10位用户已购买过的商品，并剔除用户5811已购买的商品，通过计算这
     * 10位用户已购买的商品（剔除用户5811已购买的商品）与用户5811已购买的商品数据集中商品的余弦相似度累
     * 加再求均值，输出均值前5商品id作为推荐使用。
     */
    //组合向量
    val to_03_as: VectorAssembler = new VectorAssembler()
      .setInputCols(Array("t1", "t2", "t3").union(ppids.map(x => "ppid" + x)))
      .setOutputCol("feature")
    val fm: DataFrame = to_03_as.transform(spark.table("table1"))

    //计算差集
    spark.udf.register("no_coll", (mu: mutable.WrappedArray[Int]) => {
      mu.toSet &~ bc_to1coll.toSet
    })
    val top10OrderDate = xs_date
      .limit(10)
      .withColumn("no_order_id_coll", expr("no_coll(order_id_coll)"))
      .withColumn("t1", explode('no_order_id_coll))
      .withColumn("user_id", '用户id)
      .drop("order_id_coll", "相似度", "no_order_id_coll")
      .join(fm, Array("t1"))
      .select("t1", "t2", "t3", "feature", "user_id")
    //计算相似度 积使用笛卡尔(交叉连接)
    //获取 用户 1 的所有商品信息
    val frame: DataFrame = fm
      .where('t1.isin(to1coll.map(_.toDouble): _*))
      .select('feature as "CurrentUser1_feature", 't1 as "CurrentUser1_order_id")
    // 注册一个名为"Similarity"的UDF，接受两个向量参数
    // 向量类型为Vector，可以是Spark MLlib中的DenseVector或SparseVector
    spark.udf.register("Similarity", (x1: Vector, x2: Vector) => {
      // 使用内积(dot product)方法计算两个向量的点积
      val v1 = x1.dot(x2)

      // 计算两个向量的欧几里得范数（L2 norm）
      // 参数2L表示使用Long类型，确保结果不会溢出
      val v2 = Vectors.norm(x1, 2L) * Vectors.norm(x2, 2L)

      // 余弦相似度公式：两个向量的点积除以它们各自范数的乘积
      // 这个比率表示两个向量在高维空间中的夹角余弦值，范围在-1.0到1.0之间
      // 1.0表示完全相同，0.0表示正交，-1.0表示方向相反
      v1 / v2
    })
    println("------------------------推荐Top5结果如下------------------------")
    top10OrderDate
      .crossJoin(frame)
      .withColumn("labels", concat_ws(",",'user_id, 't1, 'user_id))
      .withColumn("相似度", expr("Similarity(feature,CurrentUser1_feature)"))
      .groupBy('labels)
      .agg(avg('相似度) as "avgSimilarity", first('t1) as "order_id")
      .select("avgSimilarity", "order_id")
      .distinct()
      .orderBy('avgSimilarity.desc)
      .take(5)
      .zipWithIndex
      .foreach(x => {
        println(s"相似度top${x._2 + 1}(商品id：${x._1(1)}，平均相似度：${x._1(0)})")
      })
    spark.stop()
  }
}
