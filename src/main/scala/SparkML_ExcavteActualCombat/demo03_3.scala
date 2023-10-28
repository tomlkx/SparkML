//package SparkML_ExcavteActualCombat
//
//import com.fasterxml.jackson.databind.ObjectMapper
//import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
//import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
//import org.apache.spark.ml.feature.VectorAssembler
//import org.apache.spark.sql.{DataFrame, SparkSession}
//
//import java.sql.Timestamp
//import java.text.SimpleDateFormat
//import java.util.{Date, Properties}
//import scala.collection.{immutable, mutable}
//import scala.jdk.CollectionConverters.mapAsJavaMapConverter
//import scala.util.Try
//import scala.xml.{Elem, XML}
//
//object demo03_02 {
//  def main(args: Array[String]): Unit = {
//    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()
//    import org.apache.spark.sql.functions._
//    import spark.implicits._
//    val properties = new Properties()
//    properties.setProperty("user", "root")
//    properties.setProperty("password", "123456")
//    spark.read.jdbc("jdbc:mysql://bigdata1:3306/shtd_industry", "MachineData", properties).createTempView("MachineData")
//    spark.table("MachineData")
//      .map(line=>{
//        var test: Test = Test(line(0).asInstanceOf[Int])
//        if(line(2).toString == "预警")
//          test.machine_record_state=1.0
//        //设置时间
//        test.machine_record_date
//        //解析ｄｏｍ
//        val elem: Elem = XML.loadString(s"<x>${line(5).toString}</x>")
//        (elem \ "col").map(item=>{
//          //
//          val name: Object = item.attribute("colName").getOrElse("")
//          val d: Double = Try(item.text.toDouble).getOrElse(0.0)
//        })
//        test
//      })
//     //添加四个字段
//　　　//.withColumn()
//     //写入Ｈive 表
//
//    //随机森林
//    //读取Ｈｉｖｅ　表
//    //提取特征
//    //剔除无用特征
//    //构建随机森林
//    //网格调整参数
//    //通过管道串联
//    //使用分类评估器验证模型准确率
//    spark.stop()
//  }
//
//  case class Test(
//                   machine_record_id: Int,
//                   var machine_id: Double=0.0,
//                   var machine_record_state: Double=0.0,
//                   var machine_record_mainshaft_speed: Double=0.0,
//                   var machine_record_mainshaft_multiplerate: Double=0.0,
//                   var machine_record_mainshaft_load: Double=0.0,
//                   var machine_record_feed_speed: Double=0.0,
//                   var machine_record_feed_multiplerate: Double=0.0,
//                   var machine_record_pmc_code: Double=0.0,
//                   var machine_record_circle_time: Double=0.0,
//                   var machine_record_run_time: Double=0.0,
//                   var machine_record_effective_shaft: Double=0.0,
//                   var machine_record_amount_process: Double=0.0,
//                   var machine_record_use_memory: Double=0.0,
//                   var machine_record_free_memory: Double=0.0,
//                   var machine_record_amount_use_code: Double=0.0,
//                   var machine_record_amount_free_code: Double=0.0,
//                   var machine_record_date: Timestamp=null
//                 )
//}
