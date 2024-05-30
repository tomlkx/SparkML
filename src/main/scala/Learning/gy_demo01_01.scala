package Learning

import java.text.SimpleDateFormat
import java.util.{Date, Properties}

import org.apache.spark.SparkConf
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.dom4j.Element
import org.dom4j.io.SAXReader

import scala.collection.mutable.{ListBuffer, Map}
import org.apache.spark.sql.functions._
//用于创建表的样例类,装每一行的数据
case class learning_data(machine_record_id:Int,
                         machine_id:Int,
                         machine_record_state:Double,
                         machine_record_mainshaft_speed:Double,
                         machine_record_mainshaft_multiplerate:Double,
                         machine_record_mainshaft_load :Double,
                         machine_record_feed_speed:Double,
                         machine_record_feed_multiplerate:Double,
                         machine_record_pmc_code:Double,
                         machine_record_circle_time:Double,
                         machine_record_run_time :Double,
                         machine_record_effective_shaft:Double,
                         machine_record_amount_process:Double,
                         machine_record_use_memory :Double,
                         machine_record_free_memory:Double,
                         machine_record_amount_use_code:Double,
                         machine_record_amount_free_code:Double,
                         machine_record_date:String
                        )
object Test_1 {
  def main(args: Array[String]): Unit = {
    //根据 dwd 库中 fact_machine_data 表（或 MySQL 的 shtd_industry 库中
    //MachineData 表），根据以下要求转换：获取最大分区（MySQL 不用考虑）
    //的 数 据 后 ， 首 先 解 析 列 machine_record_data （ MySQL 中 为
    //MachineRecordData）的数据（数据格式为 xml，采用 dom4j 解析，解析 demo
    //在客户端/home/ubuntu/Documents 目录下），并获取每条数据的主轴转速，
    //主轴倍率，主轴负载，进给倍率，进给速度，PMC 程序号，循环时间，运行
    //时间，有效轴数，总加工个数，已使用内存，未使用内存，可用程序量，注
    //册程序量等相关的值（若该条数据没有相关值，则按下表设置默认值），同
    //时转换 machine_record_state 字段的值，若值为报警，则填写 1，否则填写
    //0，以下为表结构，将数据保存在 dwd.fact_machine_learning_data，分区
    //字段为 etldate，类型为 String，且值为当前比赛日的前一天日期（分区字
    //段格式为 yyyyMMdd）。使用 hive cli 按照 machine_record_id 升序排序

    val conf = new SparkConf().setAppName("工业数据挖掘1-1")
      .setMaster("local[*]")
    val spark = SparkSession.builder().enableHiveSupport().config(conf).getOrCreate()
    import spark.implicits._
    val properties = new Properties()
    properties.put("user","root")
    properties.put("password","123456")
    val format = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
    //题目要求的字段
    val list=List("主轴转速","主轴倍率","主轴负载","进给倍率","进给速度","PMC 程序号","循环时间","运行时间","有效轴数","总加工个数","已使用内存","未使用内存","可用程序量","注册程序量")
    //这里我们读取machinedata这张表
    spark.read.jdbc("jdbc:mysql://localhost/shtd_industry?characterEncoding=UTF-8","machinedata",properties)
      //把表转换为rdd 就是一个装row对象的rdd,一个row对象就是一行数据
      .rdd
      //所以这个map在循环的是这张表的所有数据
      .map(li=>{
      //存放表数据,用于后续创建表.这个table_list是存放数据的,可以想象成一个新的row对象装新数据和字段,只是存值和取值不一样

      val table_list=new ListBuffer[Any]
        //这里先把表里面有的字段放进去
      table_list.append(li.getInt(0)) //machine_record_id
      table_list.append(li.getInt(1)) //machine_id
        //这里设备状态,表中是文字报警或者其他的,我们要转换为double报警转换为1.0其他转换为0.0并添加进table_list
      if (li.getString(2).equals("报警")){
        table_list.append(1.0) //machine_record_state
      }else{
        table_list.append(0.0) //machine_record_state
      }
      //这是machine_record_date这个字段,
      table_list.append(li.getTimestamp(4).toString)
        //上面四个字段表里面都有所有直接拿进去放好,0 machine_record_id ,1  machine_id , 2 machine_record_state ,3 machine_record_date

        //现在我们解析MachineRecordData字段
      val xmlString="<cols>"+li.getString(3)+"</cols>"
      // 创建SAXReader对象
      val reader = new SAXReader()
      // 将XML字符串转换为Document对象
      val document = reader.read(new java.io.StringReader(xmlString))
      // 获取根元素
      val root = document.getRootElement()
        //上面固定解析步骤,不要问为什么,这个是别人写好的工具类,我们直管学习使用
      // 遍历所有名为"col"的子元素并提取属性值和文本内容,这里提取col所有的内容形成一个迭代器
      val elements  = root.elements("col").iterator()
      //存放xml解析出数据,这个map放置我们解析出col里面的所有 ColName-‘值’ 这样,有些我们不需要的字段我们还是放里面
      val map =  Map[String,String]()
        //循环迭代器
      while (elements .hasNext){
        //attributeName取的是ColName是什么ColName="报警信息", attributeName = 报警信息
        val element = elements.next().asInstanceOf[Element]
        val attributeName = element.attributeValue("ColName")
        //就是对应的值<col ColName="报警信息">null</col>，就等于null
        val text = element.getText()
        //我们这里迭代一次就把全部存进去
        map.put(attributeName,text)
      }
        //这里我们再拿这个map里面的东西,循环我们每一个必须要的字段,在这个map里面找,
      val MachineRecordData = list.map(li => {
        //模式匹配,能找到并且不为null我们就拿值并转换为toDouble,拿不到或是拿出来为null我们就设为0.0,map.get(li)出来的东西就是Some(59828240),some包起来的
        println(map.get(li))
        map.get(li) match {
          case Some(name) => if (name.equals("null")) 0.0 else name.toDouble
          case None => 0.0
        }

      })
        //这里就实例化表对象,数据放在table_list和MachineRecordData里面这里我们对应取就行了
      learning_data(table_list(0).asInstanceOf[Int],table_list(1).asInstanceOf[Int],table_list(2).asInstanceOf[Double],MachineRecordData(0),MachineRecordData(1),MachineRecordData(2),MachineRecordData(3),MachineRecordData(4),MachineRecordData(5),MachineRecordData(6),MachineRecordData(7),MachineRecordData(8),MachineRecordData(9),MachineRecordData(10),MachineRecordData(11),MachineRecordData(12),MachineRecordData(13),table_list(3).asInstanceOf[String])
    })
      //样例类可以直接todf建表
      .toDF()
      //正常添加四个字段
      .withColumn("dwd_insert_user",lit(""))
      .withColumn("dwd_insert_time",lit(format.format(new Date)))
      .withColumn("dwd_modify_user",lit(""))
      .withColumn("dwd_modify_time",lit(format.format(new Date)))
      .show()
    //最后写进目标表
//      .write
//      .mode(SaveMode.Overwrite)
//      .jdbc("jdbc:mysql://bigdata1/shtd_industry?characterEncoding=UTF-8","fact_machine_learning_data",properties)
      //.show()

  }
}
