package SparkML_ExcavteActualCombat

import com.fasterxml.jackson.databind.ObjectMapper
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}

import java.{lang, util}
import java.sql.Timestamp
import java.text.SimpleDateFormat
import java.util.{Date, Properties}
import scala.collection.{immutable, mutable}
import scala.collection.JavaConverters._
import scala.util.Try
import scala.xml.{Elem, XML}

object demo03 {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()
    import spark.implicits._
    import org.apache.spark.sql.functions._
    val properties = new Properties()
    properties.setProperty("user", "root")
    properties.setProperty("password", "123456")
    spark.read.jdbc("jdbc:mysql://bigdata1:3306/shtd_industry", "MachineData", properties).cache().createTempView("MachineData")
    //打印一条数据 可以起到对数据缓存的作用
    spark.table("MachineData").show()
    println(spark.table("MachineData").select("MachineRecordData").limit(1).collect()(0)(0))
    /***
     * 自定义UDF　解析 xml
     */
    val items: Seq[(String, String)] = Seq(
      ("主轴转速", "machine_record_mainshaft_speed"), ("主轴倍率", "machine_record_mainshaft_multiplerate"),
      ("主轴负载", "machine_record_mainshaft_load"), ("进给倍率", "machine_record_feed_speed"),
      ("进给速度", "machine_record_feed_multiplerate"), ("PMC程序号", "machine_record_pmc_code"),
      ("循环时间", "machine_record_circle_time"), ("运行时间", "machine_record_run_time"),
      ("有效轴数", "machine_record_effective_shaft"), ("总加工个数", "machine_record_amount_process"),
      ("已使用内存", "machine_record_use_memory"), ("未使用内存", "machine_record_free_memory"),
      ("可用程序量", "machine_record_amount_use_code"), ("注册程序量", "machine_record_amount_free_code")
    )
    // 注册广播变量节省系统资源
    val itemsBro: Seq[(String, String)] = spark.sparkContext.broadcast(items).value
    // 注冊自定义UDF
    spark.udf.register("xxx", (dataString: String, MachineRecordID: Int, MachineID: Int, MachineRecordState: String, MachineRecordDate: Timestamp) => {
      //解析 XML
      val elem: Elem = XML.loadString(s"<x>${dataString}</x>")
      //提取 ColName
      val tuples: immutable.Map[String, Double] = (elem \ "col").map(item => {
        val name: String = item.attribute("ColName").getOrElse("").toString
        val value: Double = Try(item.text.toDouble).getOrElse(0.0)
        (name, value)
      }).toMap
      //补全未收集齐的列
      val buffer: mutable.HashMap[String, Any] = mutable.HashMap[String, Any]()
      itemsBro.foreach(item => {
        //如果找不到补为0.0
        val value: Double = tuples.getOrElse(item._1, 0.0)
        buffer.put(item._2, value)
      })
      //时间转换字符串
      val dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
      val state = if (MachineRecordState == "报警") 1.0 else 0.0
      val list: Seq[(String, Any)] = Seq(
        ("machine_record_id", MachineRecordID), ("machine_id", MachineID.toDouble),
        ("machine_record_state", state), ("dwd_insert_user", null),
        ("dwd_insert_time", dateFormat.format(new Date())), ("dwd_modify_user", null),
        ("machine_record_date", dateFormat.format(MachineRecordDate.getTime)), ("dwd_modify_time",dateFormat.format(new Date()))
      )
      list.foreach(line => {
        buffer.put(line._1, line._2)
      })
      //转换为Json
      val mapper = new ObjectMapper()
      //转换为Java 类型HashMap
      val javaHashMap: util.Map[String, Any] = buffer.asJava
      mapper.writeValueAsString(javaHashMap)
    })
    val array = spark.table("MachineData")
      //提取所有ColName
      .withColumn(
        "test",
        expr("xxx(MachineRecordData,MachineRecordID,MachineID,MachineRecordState,MachineRecordDate)")
      )
      .select("test")
      //对数据进行转换
      .rdd
      .map(_(0).toString)
      .toDS()
      .cache()
    //打印一条数据 可以起到对数据缓存的作用
    array.show(truncate = false)
    spark.read.json(array)
      .select(
        'machine_record_id, 'machine_id,
        'machine_record_state, 'machine_record_mainshaft_speed,
        'machine_record_mainshaft_multiplerate, 'machine_record_mainshaft_load,
        'machine_record_feed_speed, 'machine_record_feed_multiplerate,
        'machine_record_pmc_code,
        'machine_record_circle_time, 'machine_record_run_time,
        'machine_record_effective_shaft, 'machine_record_amount_process, 'machine_record_use_memory,
        'machine_record_free_memory, 'machine_record_amount_use_code,
        'machine_record_amount_free_code, 'machine_record_date, 'dwd_insert_user, 'dwd_insert_time,
        'dwd_modify_user, 'dwd_modify_time
      )
      .withColumn("etldate", lit("20220703"))
      .cache()
      .createTempView("test")
    //打印一条数据 可以起到对数据缓存的作用
    spark.table("test").show()
    //升序查询第一条
    spark.table("test").sort('machine_record_id).limit(1).show()
    //组合特征
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array(
        "machine_id",
        "machine_record_state", "machine_record_mainshaft_speed",
        "machine_record_mainshaft_multiplerate", "machine_record_mainshaft_load",
        "machine_record_feed_speed", "machine_record_feed_multiplerate",
        "machine_record_pmc_code",
        "machine_record_circle_time", "machine_record_run_time",
        "machine_record_effective_shaft", "machine_record_amount_process", "machine_record_use_memory",
        "machine_record_free_memory", "machine_record_amount_use_code",
        "machine_record_amount_free_code"
      ))
      .setOutputCol("feature")
    val features: DataFrame = assembler.transform(spark.table("test")).cache()
    features.show()
    //划分数据集合
    val Array(t1, t2) = features.randomSplit(Array(0.7, 0.3), 1)
    //训练模型
    val model: RandomForestClassificationModel = new RandomForestClassifier()
      .setNumTrees(100)
      .setLabelCol("machine_record_state")
      .setFeaturesCol("feature")
      .fit(t1)
    //预测
    val frame: DataFrame = model.transform(t2).select("machine_record_state", "prediction")
    //评估模型
    val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("machine_record_state")
    val d: Double = evaluator.evaluate(frame)
    print(d)
    spark.stop()
  }

  case class Test(
                   machine_record_id: Int,
                   machine_id: Double=0.0,
                   machine_record_state: Double=0.0,
                   machine_record_mainshaft_speed: Double=0.0,
                   machine_record_mainshaft_multiplerate: Double=0.0,
                   machine_record_mainshaft_load: Double=0.0,
                   machine_record_feed_speed: Double=0.0,
                   machine_record_feed_multiplerate: Double=0.0,
                   machine_record_pmc_code: Double=0.0,
                   machine_record_circle_time: Double=0.0,
                   machine_record_run_time: Double=0.0,
                   machine_record_effective_shaft: Double=0.0,
                   machine_record_amount_process: Double=0.0,
                   machine_record_use_memory: Double=0.0,
                   machine_record_free_memory: Double=0.0,
                   machine_record_amount_use_code: Double=0.0,
                   machine_record_amount_free_code: Double=0.0,
                   machine_record_date: Timestamp=null
                 )
}
