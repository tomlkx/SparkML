package SparkML_Demo_2024

import org.apache.spark.ml
import org.apache.spark.ml.feature
import org.apache.spark.ml.feature.{OneHotEncoder, VectorAssembler}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.linalg.{Matrices, Matrix, SingularValueDecomposition, Vectors, distributed}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.mllib.linalg
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.dense_rank

import java.util.Properties

object demo_02 {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("demo02").getOrCreate()
    spark.sparkContext.setLogLevel("FATAL")
    import spark.implicits._
    import org.apache.spark.sql.functions._
    val pro = new Properties()
    pro.setProperty("user", "root")
    pro.setProperty("password", "123456")
    val table_name: Seq[String] = Seq("order_detail", "order_info", "sku_info")
    table_name.foreach((x: String) => spark.read.jdbc("jdbc:mysql://LIUKAIXIN:4306/TB_DS", s"${x}", pro).cache().createTempView(x))
    table_name.foreach(line=>spark.table(line).show())
    spark.sql(
        """
          |select
          | a.user_id,
          | b.sku_id
          |from
          |order_info a inner join order_detail b on a.id = b.order_id
          |""".stripMargin)
      .distinct() // 剔除重复
      .withColumn("user_id_mapping", dense_rank().over(Window.orderBy('user_id)))
      .withColumn("sku_id_mapping", dense_rank().over(Window.orderBy('sku_id)))
      .withColumn("user_id_mapping", 'user_id_mapping  - lit(1))
      .withColumn("sku_id_mapping", 'sku_id_mapping - lit(1))
      .orderBy("user_id_mapping", "sku_id_mapping") // 排序
      .cache()
      .createTempView("table_task_1")


    println("------user_id_mapping与sku_id_mapping数据前五行如下：------")
    spark.table("table_task_1").take(5).map(x => s"${x(2)}:${x(3)}").foreach(println(_))


    spark.table("table_task_1")
      .withColumn("sku_id_mapping", concat(lit("sku_id_mapping"), 'sku_id_mapping.cast("int")))
      .groupBy("user_id_mapping")
      .pivot("sku_id_mapping")
      .agg(first(lit(1.0)))
      .na.fill(0.0)
      .createTempView("table_task_2")
    spark.table("table_task_2").show()
    println("------第一行前5列结果展示为------")
    println(
      spark.sql(
        s"""
           |select
           |  user_id_mapping,${spark.table("table_task_1").select('sku_id_mapping.cast("int")).distinct().orderBy("sku_id_mapping").collect().map(x => s"sku_id_mapping${x(0)}").mkString(",")}
           |from table_task_2 order by user_id_mapping
           |""".stripMargin).orderBy('user_id_mapping).take(1)(0).toSeq.take(5).mkString(",")
    )
    /**
     * 保证查询列的顺序
     */
    spark.sql(
        s"""
           |select
           | user_id_mapping,${spark.table("table_task_1").select('sku_id_mapping.cast("int")).distinct().orderBy("sku_id_mapping").collect().map(x => s"sku_id_mapping${x(0)}").mkString(",")}
           |from table_task_2 order by user_id_mapping
           |""".stripMargin)
      .createTempView("table_task_3")
    /**
     * 特征组合
     */
    val rdd: RDD[(String, linalg.Vector)] = new VectorAssembler()
      .setInputCols(spark.table("table_task_1").select('sku_id_mapping.cast("int")).distinct().orderBy("sku_id_mapping").collect().map(x => s"sku_id_mapping${x(0)}"))
      .setOutputCol("feature")
      .transform(spark.table("table_task_3"))
      .select("user_id_mapping", "feature")
      .rdd
      .map((line: Row) => {
        (line.get(0).toString, Vectors.dense(line(1).asInstanceOf[ml.linalg.Vector].toArray)
        )
      })
    /**
     * 降维
     */
    val matrix = new RowMatrix(rdd.map(_._2))
    val usv: SingularValueDecomposition[RowMatrix, Matrix] = matrix.computeSVD(k = 5, computeU = true)
    /**
     * V矩阵 跟 sku_id 关联
     */
    spark.createDataFrame(
        spark.table("table_task_1")
          .select('sku_id_mapping.cast("int"))
          .distinct()
          .orderBy("sku_id_mapping")
          .collect()
          .map(x => x(0).toString.toInt + 1)
          .zip(usv.V.rowIter.toSeq)
      ).toDF("sku_id", "feature")
      .createTempView("table_task_4")
    println(s"table_task1 sku_id 数量为: ${spark.table("table_task_1").select("sku_id").distinct().count()}")
    println(s"table_task4 sku_id 数量为: ${spark.table("table_task_4").select("sku_id").distinct().count()}")
    /**
     * 使用用户4购买 的商品  和 未购买的商品进行笛卡尔积
     */
    // 注册一个名为"Similarity"的UDF，接受两个向量参数
    // 向量类型为Vector，可以是Spark MLlib中的DenseVector或SparseVector
    spark.udf.register("Similarity", (x1: org.apache.spark.mllib.linalg.DenseVector, x2: org.apache.spark.mllib.linalg.DenseVector) => {
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
    val top_sku_id_4: Array[Int] = Array(1,2,3,4,5)
    //val top_sku_id_4: Array[Int] = spark.table("table_task_1").filter('user_id === lit(4)).select("sku_id").collect().map(_(0).toString.toInt)
    spark.table("table_task_4").filter('sku_id.isin(top_sku_id_4: _*))
      .withColumn("feature_label_4", 'feature)
      .withColumn("sku_id_label_4", 'sku_id)
      .select("feature_label_4","sku_id_label_4")
      .crossJoin(
        spark.table("table_task_4").filter(!'sku_id.isin(top_sku_id_4: _*))
      )
      .withColumn("相似度", expr("Similarity(feature_label_4,feature)"))
      .groupBy('sku_id)
      .agg(avg('相似度) as "平均相似度",count(lit(1)) as "count")
      .orderBy('平均相似度.desc)
      .take(5)
      .zipWithIndex
      .foreach(x=>{
        println(s"相似度top${x._2 + 1}(商品id：${x._1(0)}，平均相似度：${x._1(1)})")
      })
    spark.stop()
  }
}
