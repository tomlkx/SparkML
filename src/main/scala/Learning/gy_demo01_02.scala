package Learning

import java.util.Properties

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object gy_demo01_02 {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[*]").setAppName("工业数据挖掘1-2")
    val spark = SparkSession.builder().enableHiveSupport().config(conf).getOrCreate()
    val properties = new Properties()
    properties.put("user","root")
    properties.put("password","123456")
    // 从 Hive 中读取训练数据集,这里读取刚刚写入的数据,处理好的数据
    val data = spark.read.jdbc("jdbc:mysql://localhost/shtd_industry?characterEncoding=UTF-8","fact_machine_learning_data",properties)
    //这一步是划分测试数据和训练数据,什么是训练就是塞给计算机找规律的,测试数据就是对他进行考试的数据
    val splits = data.randomSplit(Array(0.7, 0.3))
    //trainData训练,testData测试. trainData训练占全部数据0.7,estData占0.3
    val (trainData, testData) = (splits(0), splits(1))
    //到这一步什么是特征什么是向量,特征就是每一形容字段,比如长度宽度,大小之类的,向量是每一个模型接收特征的数据格式, 比如 我 18 100斤 臂长 170  特征向量就会是 18,100,170 这样
    // 将特征列转换为向量,就是把所以形容的特征值塞到一个字段里面
    //VectorAssembler是一个创建向量的类,接收需要整合到一起的字段集合trainData.columns.slice(3,17)这些字段就是特征字段
    val assembler = new VectorAssembler()
      .setInputCols(trainData.columns.slice(3,17))
      //这里是输出的字段叫什么
      .setOutputCol("features")
    //这里是执行转换
    val trainDataVector = assembler.transform(trainData)
    //trainDataVector.show()
    //有了特征向量字段直接就塞给模型就行了
    // 构建随机森林模型
    val rfc = new RandomForestClassifier()
      //设置哪个是结果machine_record_state字段就是是否报警,我们模型就为了预测这条记录是否报警,
      .setLabelCol("machine_record_state")
      //这里是传入特征向量是哪一列
      .setFeaturesCol("features")
      //参赛调整具体看比赛怎么给
      .setNumTrees(100)
      .setMaxDepth(10)

    // 开始训练模型 ,trainDataVector训练数据集
    val model = rfc.fit(trainDataVector)
    //model就是训练好的模型
    // 将测试特征列转换为向量
    val testDataVector = assembler.transform(testData)
    // 使用model进行预测并输出结果到 MySQL 数据库model.transform(testDataVector)就是预测 testDataVector是测试数据集
    val result = model.transform(testDataVector)
      //这里查询我们需要的字段prediction就是预测结果,这里数据很少肯定预测不对不用关心结果
      .select(col("machine_record_id"), col("prediction"),col("machine_record_state"))
    result.show(50)
  }
}
