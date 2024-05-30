package Learning

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}

import java.util.Properties

class ds_demo01_util {

  Logger.getLogger("org").setLevel(Level.ERROR)

  // 数据库链接与用户信息
  private val url = "jdbc:mysql://LIUKAIXIN:4306/G_DS?useUnicode=true&characterEncoding=utf-8&serverTimezone=Asia/Shanghai"
  private val prop = new Properties()
  prop.setProperty("user","root")
  prop.setProperty("password","123456")
  // 解决报错：Exception in thread "main" java.sql.SQLException: No suitable driver
  // 原因：找不到驱动

  // 获取sparkSession
  def getSparkSession:SparkSession={
    val conf = new SparkConf().setMaster("local").setAppName("国赛第一套")
    new SparkSession.Builder().config(conf)
      .config("hive.exec.dynamic.partition","true")
      .config("hive.exec.dynamic.partition.mode","nonstrict")
      // 解决hive查询中报错：Failed with exception java.io.IOException:org.apache.parquet.io.ParquetDecodingException: Can not read value at 0 in block -1 in file hdfs://bigdata01:9000/
      // 出现这个报错的根本原因是因为Hive和Spark中使用的不同的parquet约定引起的。
      .config("spark.sql.parquet.writeLegacyFormat","true")
      .enableHiveSupport().getOrCreate()
  }

  // 读取MySql表
  def readMysql(tableName:String):DataFrame={
    getSparkSession.read.jdbc(url,tableName,prop)
  }

  //写入MySql表
  def writeMySql(dataFrame: DataFrame,tableName:String,database:String):Unit={
    dataFrame.write.mode("overwrite").jdbc(s"jdbc:mysql://192.168.10.1/$database?useUnicode=true&characterEncoding=utf-8&serverTimezone=Asia/Shanghai",tableName,prop)
  }


}
