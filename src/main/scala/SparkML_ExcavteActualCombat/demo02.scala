package SparkML_ExcavteActualCombat

import org.apache.spark.ml.feature.{MaxAbsScaler, MinMaxScaler, PCA, PCAModel}
import org.apache.spark.{SparkContext, ml}
import org.apache.spark.ml.linalg._
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition, Vectors => mllibVectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.expressions.Window

import java.util.Properties
import scala.collection.mutable

object demo02 {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .config("spark.driver.extraJavaOptions", "-Xss4M")
      .config("spark.executor.extraJavaOptions", "-Xss4M")
      .appName("demo02")
      .master("local[*]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    import spark.implicits._
    import org.apache.spark.sql.functions._
    val pro = new Properties()
    pro.setProperty("user", "root")
    pro.setProperty("password", "123456")
    //读取数据
    spark.read.jdbc("jdbc:mysql://bigdata1:3306/ds_db01", "order_detail", pro).createTempView("detail")
    spark.read.jdbc("jdbc:mysql://bigdata1:3306/ds_db01", "order_master", pro).createTempView("master")
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
    //输出第一行前五列
    println(spark.table("t2_data").select("feature_list").take(1).map(_(0).asInstanceOf[mutable.WrappedArray[Double]]).array(0).take(5).mkString(","))

    /** *
     * 根据上述任务的结果，对其进行SVD分解，对数据进行降维保留至
     * 少保留前95%的信息量，根据该用户customer_id为5811的用户已
     * 购买的商品分别与未购买的商品计算余弦相似度再进行累加求均值，
     * 将均值最大的5件商品id进行输出作为推荐使用。将结果输出。
     * 结果格式如下：
     */
    //对特征进行缩放
    //    val scaler = new MaxAbsScaler()
    //      .setInputCol("feature_list_vector")
    //      .setOutputCol("scaledFeatures")
    //    scaler.fit(spark.table("t2_data")).transform(spark.table("t2_data")).createTempView("t2_data_feature")
    var frame = spark.table("t2_data")
      .where('customer_id =!= 5811)
      .withColumn("feature_avg", lit(0.0).cast("double"))
    spark.udf.register("xxxxx", (x1: Vector, x2: Vector) => {
      val v1 = x1.dot(x2)
      val v2 = Vectors.norm(x1, 2L) * Vectors.norm(x2, 2L)
      v1 / v2
    })
    //计算5811 购买的商品
    val tuples = spark.table("t2_data").where('customer_id === 5811).select("customer_id", "feature_list_vector").collect().map(line => (line(0), line(1).asInstanceOf[Vector]))
    for (item <- tuples) {
      frame =
      //插入 5811 的特征
        frame.withColumn(s"customer_feature", typedLit(item._2))
          //计算5811商品的特征与这个商品的特征 相似度
          .withColumn("feature_avg", expr(s"xxxxx(customer_feature,feature_list_vector)") + col("feature_avg"))
    }
    //计算均值
    frame = frame.withColumn("feature_avg", col("feature_avg") / lit(tuples.length))
    frame.select("customer_id", "feature_avg").sort('feature_avg.desc).take(5).map(line=>(line(0),line(1))).zipWithIndex.map(line=>{s"相似度top${line._2 + 1}(商品id：${line._1._1}，平均相似度：${line._1._2})"}).foreach(println)

    //val pca: PCAModel = new PCA()
    //      .setInputCol("scaledFeatures")
    //      .setOutputCol("pac_feature")
    //      .setK(14000)
    //      .fit(spark.table("t2_data_feature"))
    //    pca.transform(spark.table("t2_data_feature")).show()
    //val rdd = t2_data.select("feature_list").rdd.map(_(0).asInstanceOf[mutable.WrappedArray[Double]]).map(line => mllibVectors.dense(line.toArray))
    //缓存rdd
    //    rdd.cache()
    //    val matrix = new RowMatrix(rdd)
    //    val xdd: SingularValueDecomposition[RowMatrix, Matrix] = matrix.computeSVD(14500, computeU = true)
    //    xdd.V.rowIter.take(1).foreach(println)

    spark.stop()
  }
}
