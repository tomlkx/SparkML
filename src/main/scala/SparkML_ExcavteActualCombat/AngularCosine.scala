package SparkML_ExcavteActualCombat

import org.apache.spark.ml.feature.{MaxAbsScaler, MinMaxScaler, PCA, PCAModel, VectorAssembler}
import org.apache.spark.{SparkContext, ml}
import org.apache.spark.ml.linalg._
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition, Vectors => mllibVectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.expressions.Window

import java.util.Properties
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

object AngularCosine {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .config("spark.sql.warehouse.dir", "hdfs://bigdata1:9000/user/hive/warehouse")
      .config("hive.metastore.uris", "thrift://bigdata1:9083")
      .config("spark.sql.pivotMaxValues", "15000")
      .enableHiveSupport()
      .appName("demo02")
      .master("local[*]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("WARN")
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
    t1_data.cache()
    //透视函数
    val data: DataFrame = t1_data.groupBy("customer_id")
      .pivot("product_id")
      .agg(first(lit(1.0)))
      .na.fill(0.0)
    val buffer = scala.collection.mutable.ArrayBuffer[String]()
    for (i <- 1 to 14700) {
      buffer.append(i + "")
    }
    val frame: DataFrame = new VectorAssembler()
      .setInputCols(buffer.toArray)
      .setOutputCol("feature")
      .transform(data)
    val pca: PCAModel = new PCA()
      .setInputCol("feature_list_vector")
      .setOutputCol("pac_feature")
      .setK(5)
      .fit(frame)
    pca.transform(frame).show()
    spark.stop()
  }
}
