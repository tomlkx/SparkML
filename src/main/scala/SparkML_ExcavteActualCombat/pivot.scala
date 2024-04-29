package SparkML_ExcavteActualCombat

import org.apache.spark.SparkContext
import org.apache.spark.sql.{SparkSession}

object pivot {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("demo02")
      .master("local[*]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    import spark.implicits._
    import org.apache.spark.sql.functions._
    sc.setLogLevel("ERROR")
    //透视函数
    // 创建示例数据表
    val data = Seq(
      ("Alice", 2019, 100),
      ("Bob", 2019, 200),
      ("Alice", 2020, 150),
      ("Bob", 2020, 250),
      ("Bob", 2023, 250),
      ("Lkx", 0, 250)
    )
    val df = spark.createDataFrame(data).toDF("Name", "Year", "Value")

    // 使用数据透视来设置非重复列  按照Name分组 透视Year 如果有值插1 没有值0
    val pivotedDF = df.groupBy("Name").pivot("Year").agg(first(lit(1))).na.fill(0)
    pivotedDF.show()

    spark.stop()
  }
}
