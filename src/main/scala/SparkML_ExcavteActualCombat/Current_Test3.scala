package SparkML_ExcavteActualCombat

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.Window

import java.util.Properties
import scala.collection.mutable

object Current_Test3 {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .config("spark.sql.warehouse.dir", "hdfs://hadoop102:8020/user/hive/warehouse")
      .config("hive.metastore.uris", "thrift://hadoop102:9083")
      .enableHiveSupport()
      .appName("demo02")
      .master("local[*]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("ERROR")
    import spark.implicits._
    import org.apache.spark.sql.functions._
    val pro = new Properties()
    pro.setProperty("user", "root")
    pro.setProperty("password", "123456")
    //读取数据
    spark.read.jdbc("jdbc:mysql://hadoop102:3306/ds_db01", "order_detail", pro).createTempView("detail")
    spark.read.jdbc("jdbc:mysql://hadoop102:3306/ds_db01", "order_master", pro).createTempView("master")
    //两表join
    spark.table("detail").join(spark.table("master"), Seq("order_sn")).createTempView("master_detail")
    val t1_data = spark.table("master_detail").select("customer_id", "product_id")
      //对用户购买的商品剔除重复
      .distinct()
      //mapping
      .withColumn("mapping", dense_rank().over(Window.orderBy('customer_id)) - 1)
    t1_data.createTempView("t1_data")
    spark.udf.register("xxx", (numbers: Array[Long], list: Array[Long], mapping: Double) => {
      val doubles = mutable.ListBuffer[Double]()
      doubles.append(mapping)
      for (number <- numbers) {
        val d: Double = if (list.contains(number)) 1.0 else 0.0
        doubles.append(d)
      }
      doubles.toList
    })
    //透视函数
    spark.udf.register("xxxx", (list: List[Double]) => {
      Vectors.dense(list.toArray)
    })
    //升序排列
    spark.table("t1_data").sort('mapping, 'product_id).select('mapping, 'product_id).take(5).map(line => s"${line(0)},${line(1)}").foreach(println)
    //聚合用户购买的商品  构建成Array
    var t2_data = spark.table("t1_data").groupBy('customer_id, 'mapping).agg(collect_list('product_id) as "customer_product_list")
    //剔除所有重复商品并构建 商品Array
    val list: Array[Long] = spark.table("t1_data").select("product_id").distinct().sort('product_id).map(_(0).asInstanceOf[Long]).collect()
    t2_data = t2_data
      //插入所有商品
      .withColumn("product_list", lit(list))
      //构建特征
      .withColumn("feature_list", expr("xxx(product_list,customer_product_list,mapping)"))
      .withColumn("feature_list_vector", expr("xxxx(feature_list)"))
    t2_data.createTempView("t2_data")
    //对数据落盘
    t2_data
      .drop("product_list", "product_list", "customer_product_list", "feature_list")
      .write
      .mode(SaveMode.Overwrite)
      .saveAsTable("test")

    //SVD降维矩阵
    val value: RDD[Vector] = spark.table("test").select("feature_list_vector").rdd.map(_(0).asInstanceOf[Vector])
    val matrix = new RowMatrix(value)
    val vx: SingularValueDecomposition[RowMatrix, Matrix] = matrix.computeSVD(5, computeU = true)
    val v: Iterator[Vector] = vx.V.rowIter
    val u: RDD[Vector] = vx.U.rows
    val s: Vector = vx.s
    val vectors: Array[Vector] = value.collect()
    //指定用户
    val user = 1
    //找出用户没有购买的商品
    val tuples: Array[(Double, Int)] = vectors(user).toArray.zipWithIndex.filter(_._1 == 0).map(x => (x._1, x._2))
    for(item <- tuples){
      //获取长度
      val len: Int = vectors.length
      //初始化累加相似度
      var simTotal = 0.0
      //带有比例的相似度
      var ratSimTotal = 0.0
      //循环用户商品集合
      for(i <- vectors(user).toArray.zipWithIndex.map(x=>(x._1,x._2))){
        // 获取用户的地 item 件商品
        val userRating: Double = vectors(user)(item._2)
        if(userRating != 0){
          vectors(user)(item._2) > 0
          vectors(user)(i._2) > 0
        }
      }
    }
    spark.stop()
  }
}
