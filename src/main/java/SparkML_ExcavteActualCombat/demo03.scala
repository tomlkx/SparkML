package SparkML_ExcavteActualCombat

import org.apache.spark.sql.SparkSession
import org.dom4j.{Document, DocumentHelper}
import org.dom4j.io.SAXReader

import java.io.FileInputStream
import java.net.URLDecoder
import java.util.Properties
import scala.collection.mutable
import scala.jdk.CollectionConverters.asScalaBufferConverter

object demo03 {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()
    import org.apache.spark.sql.functions._
    import spark.implicits._
    val pro = new Properties()
    pro.load(new FileInputStream(URLDecoder.decode(demo03.getClass.getResource("/jdbc.properties").getPath, "utf-8")))
    spark.read.jdbc("jdbc:mysql://192.168.45.5:20228/shtd_industry", "MachineData", pro)
    val read = new SAXReader()
    val document: Document = DocumentHelper.parseText(
      """
        |<xxd>
        |<col ColName="设备IP">192.168.2.230</col>
        |        <col ColName="进给速度">null</col>
        |        <col ColName="急停状态">null</col>
        |        <col ColName="加工状态">null</col>
        |        <col ColName="刀片相关信息">null</col>
        |        <col ColName="切削时间">null</col>
        |        <col ColName="未使用内存">null</col>
        |        <col ColName="循环时间">null</col>
        |        <col ColName="报警信息">null</col>
        |        <col ColName="主轴转速">null</col>
        |        <col ColName="上电时间">null</col>
        |        <col ColName="总加工个数">null</col>
        |        <col ColName="班次信息">早班</col>
        |        <col ColName="运行时间">null</col>
        |        <col ColName="正在运行刀具信息">null</col>
        |        <col ColName="有效轴数">0</col>
        |        <col ColName="主轴负载">null</col>
        |        <col ColName="PMC程序号">null</col>
        |        <col ColName="进给倍率">0</col>
        |        <col ColName="主轴倍率">null</col>
        |        <col ColName="已使用内存">null</col>
        |        <col ColName="可用程序量">null</col>
        |        <col ColName="刀补值">null</col>
        |        <col ColName="工作模式">null</col>
        |        <col ColName="机器状态">离线</col>
        |        <col ColName="连接状态">faild</col>
        |        <col ColName="加工个数">null</col>
        |        <col ColName="机床位置">null</col>
        |        <col ColName="注册程序量">null</col>
        |</xxd>
        |""".stripMargin)
    val hash = new mutable.HashMap[String, String]()
    document.getRootElement.elements().asScala.toList.foreach(item=>{
      hash.put(item.attributeValue("ColName"),item.getText)
    })
    val pojo: fact_machine_learning_data = fact_machine_learning_data(
      1,
      1,
      isStatus(hash.getOrElse("机器状态", 0.0).toString),
      h_get(hash, "主轴倍率"),
      h_get(hash, "主轴负载"),
      h_get(hash, "进给倍率"),
      h_get(hash, "进给速度"),
      h_get(hash, "PMC程序号"),
      h_get(hash, "循环时间"),
      h_get(hash, "运行时间"),
      h_get(hash, "有效轴数"),
      h_get(hash, "总加工个数"),
      h_get(hash, "已使用内存"),
      h_get(hash, "未使用内存"),
      h_get(hash, "可用程序量"),
      h_get(hash, "注册程序量"),
      "", "", "", "", ""
    )
    println(pojo)
    spark.stop()
  }

  def h_get(h:mutable.HashMap[String, String],key:String):Double={
    isNUll(h.getOrElse(key,0.0).toString)
  }

  def isNUll(item: String): Double = {
    item.trim match {
      case null | "null" | "" => 0.0
      case _ => item.toDouble
    }
  }

  def isStatus(status:String): Double = {
    status match {
      case "离线" => 1.0
      case _ => 0.0
    }
  }

  case class fact_machine_learning_data(

                                         machine_id: Double,
                                         machine_record_state: Double,
                                         machine_record_mainshaft_speed: Double,
                                         machine_record_mainshaft_multiplerate: Double,
                                         machine_record_mainshaft_load: Double,
                                         machine_record_feed_speed: Double,
                                         machine_record_feed_multiplerate: Double,
                                         machine_record_pmc_code: Double,
                                         machine_record_circle_time: Double,
                                         machine_record_run_time: Double,
                                         machine_record_effective_shaft: Double,
                                         machine_record_amount_process: Double,
                                         machine_record_use_memory: Double,
                                         machine_record_free_memory: Double,
                                         machine_record_amount_use_code: Double,
                                         machine_record_amount_free_code: Double,
                                         machine_record_date: String,
                                         dwd_insert_user: String,
                                         dwd_insert_time: String,
                                         dwd_modify_user: String,
                                         dwd_modify_time: String
                                       )
}
