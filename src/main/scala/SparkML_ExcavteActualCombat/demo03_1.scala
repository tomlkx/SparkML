package SparkML_ExcavteActualCombat

import org.apache.spark.sql.{Row, SaveMode, SparkSession}
import org.dom4j.{Document, DocumentHelper}

import java.io.FileInputStream
import java.net.URLDecoder
import java.util.Properties
import scala.collection.mutable
import scala.jdk.CollectionConverters.asScalaBufferConverter

object demo03_1 {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()
    import org.apache.spark.sql.functions._
    import spark.implicits._
    val pro = new Properties()
    pro.load(new FileInputStream(URLDecoder.decode(demo03_1.getClass.getResource("/jdbc.properties").getPath, "utf-8")))
    spark.read.jdbc("jdbc:mysql://localhost:3306/shtd_industry", "MachineData", pro).show()
    val data = spark.read.jdbc("jdbc:mysql://localhost:3306/shtd_industry", "MachineData", pro)
    // TODO 1.过滤为null
      .filter(item => {
        item.get(3) != null
      })
    // TODO 2.转换
      .map {
        case Row(machineRecordID: Int, machineID: Int, machineRecordState: String, machineRecordData: String, machineRecordDate: java.sql.Timestamp) =>
          (machineRecordID, machineID, machineRecordState, s"<xdd>${machineRecordData}</xdd>", machineRecordDate)
      }
      .map(item => {
        val document: Document = DocumentHelper.parseText(item._4)
        val hash = new mutable.HashMap[String, String]()
        document.getRootElement.elements().asScala.toList.foreach(item => {
          hash.put(item.attributeValue("ColName"), item.getText)
        })
        fact_machine_learning_data(
          item._1,
          item._2,
          isStatus(item._3),
          h_get(hash, "主轴转速"),
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
          item._5.toString,
          null,
          "2022-07-30 11:11:11",
          null,
          "2022-07-30 11:11:11"
        )
      }
      ).toDF()
      .write
      .mode(SaveMode.Overwrite)
      .jdbc("jdbc:mysql://localhost:3306/shtd_industry", "xds", pro)
    spark.stop()
  }


  def h_get(h: mutable.HashMap[String, String], key: String): Double = {
    isNUll(h.getOrElse(key, 0.0).toString)
  }

  def isNUll(item: String): Double = {
    item.trim match {
      case null | "null" | "" => 0.0
      case _ => item.toDouble
    }
  }

  def isStatus(status: String): Double = {
    status match {
      case "报警" => 1.0
      case _ => 0.0
    }
  }

  case class fact_machine_learning_data(
                                         machine_record_id:Int,
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
