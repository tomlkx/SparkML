package Learning

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import java.util.Properties

object ds_demo01 {

  def test01(util: ds_demo01_util):String={

    util.readMysql("order_detail").createOrReplaceTempView("detail_table")
    util.readMysql("order_info").createOrReplaceTempView("info_table")
    util.readMysql("sku_info").createOrReplaceTempView("sku_table")
    util.readMysql("user_info").createOrReplaceTempView("user_table")

    val spark = util.getSparkSession

    //用户id 商品id
    spark.sql(
      """
        |select
        |distinct
        |i.user_id,
        |d.sku_id
        |from
        |info_table  i
        |join detail_table  d on i.id = d.order_id
        |join sku_table s on d.sku_id = s.id
        |join user_table u on  i.user_id = u.id
        |""".stripMargin)
      .createTempView("t1")
    //6708购买的商品
    spark.sql(
      """
        |select
        |sku_id
        |from t1
        |where user_id = 6708
        |""".stripMargin)
      .createTempView("t2")
    val sku_rdd = spark.sql(
      """
        |select
        |t1.user_id,
        |count(*) rk
        |from
        |t1
        |join t2 on t1.sku_id = t2.sku_id
        |where t1.user_id != 6708
        |group by t1.user_id
        |order by rk desc
        |limit 10
        |""".stripMargin)
    val str = sku_rdd.rdd.collect().map(_.getLong(0)).mkString(",")
    println(str)

    val row = util.getSparkSession.sql("""select * from t2""").collect().map(_ (0).toString()).mkString(",")
    println(row)
    row
  }

  def cosineSimilarity(vec1: Vector, vec2: Vector): Double = {
    val dotProduct = vec1.dot(vec2)
    val norm1 = Vectors.norm(vec1,2)
    val norm2 = Vectors.norm(vec2,2)
    dotProduct / (norm1 * norm2)
  }

  // 计算两组用户特征的平均余弦相似度，并输出结果
  def calculateAndPrintTop5(
                             userFeatures1: Array[(Long, Vector)],
                             userFeatures2: Array[(Long, Vector)]
                           ): Unit = {
    val similarities = for {
      (userId1, vector1) <- userFeatures1
      (userId2, vector2) <- userFeatures2
    } yield (userId1, userId2, cosineSimilarity(vector1, vector2))

    val top5Similarities = similarities
      .groupBy(_._1)
      .mapValues(_.sortBy(-_._3).take(5))
      .toList
      .sortBy(_._2.head._3)(Ordering[Double].reverse)
      .take(5)

    println("------------------------推荐Top5结果如下------------------------")
    top5Similarities.zipWithIndex.foreach {
      case ((userId1, userId2Similarities), index) =>
        val avgSimilarity = userId2Similarities.map(_._3).sum / userId2Similarities.length.toDouble
        val topSimilarity = userId2Similarities.head
        println(
          s"相似度top${index + 1}(商品id：${topSimilarity._2}，平均相似度：$avgSimilarity)"
        )
    }
  }

  def test02(util: ds_demo01_util):DataFrame={

    util.readMysql("sku_info").createOrReplaceTempView("sku_info")
    val frame = util.getSparkSession.sql("""select distinct id,spu_id,price,weight,tm_id,category3_id from sku_info order by id""")

    val columns = Array("spu_id","tm_id","category3_id")
    println(
      """进行---------------------------------------------------------------------
        |StringIndexer 编码：
        |   columns                         是一个包含列名的集合
        |   map                             对columns中的每个列名执行相同的操作
        |   setInputCol                     设置输入列，也就是要编码的列名，‘colName’是当前循环的列名
        |   setOutputCol                    设置输出列，也就是存储编码结果的列名，这里是在输入列名的基础上加上‘_indexed’后缀
        |   setHandleInvalid("keep")        设置处理无效值的策略为”keep“,这表示如果遇到未知的类别值，讲保留原始值而不引发错误
        |   indexers                        是一个包含了创建的‘StringIndexer’对象的集合，每个对象对应一个列的处理
        |   例如，如果你有一个包含 "red"、"blue" 和 "green" 的颜色列，经过此处理后，它们将被编码为整数，如 0、1 和 2，以便输入到机器学习算法中。
        |""".stripMargin)
    val indexers = columns.map { colName =>
      new StringIndexer().setInputCol(colName).setOutputCol(colName + "_indexed").setHandleInvalid("keep")
    }

    println(
      """
        |进行--------------------------------------------------------------------
        |one-hot  编码
        |   setInputCols                  设置输入列，用来进行独热编码
        |   setOutputCols                 设置输出列
        |编码后的样子：(13,[2],[1.0])
        |   13                            一共13个变量
        |   2                             非0元素的索引值
        |   1.0                           只有一个非0元素的值，为1.0
        |""".stripMargin)
    // onehot处理
    val onehoter = new OneHotEncoder()
      .setInputCols(columns.map(_ + "_indexed"))
      .setOutputCols(columns.map(_ + "_onehot"))

    println(
      """
        |进行----------------------------------------------------------------------
        |特征向量组装器(VectorAssembler)
        |   featureCol 是一个包含特征列名称的数组，其中包括 "price" 和 "weight" 列。这些列包含了你希望合并为一个特征向量的特征。
        |   VectorAssembler 用于将多个特征列合并成一个特征向量列
        |   setInputCols(featureCol) 设置了输入列，告诉 VectorAssembler 需要合并哪些列。在这里，它会合并 "price" 和 "weight" 列。
        |   setOutputCol("scaledFeatures") 设置了输出列的名称，这是合并后的特征向量列的名称。在这里，合并后的特征向量将被存储在一个名为 "scaledFeatures" 的新列中。
        |例如：
        |   +-----+-------+----------------+
        |   |price|weight | scaledFeatures |
        |   +-----+-------+----------------+
        |   | 10.0|  2.1  | [10.0, 2.1]   |
        |   | 15.0|  3.3  | [15.0, 3.3]   |
        |   | 20.0|  2.7  | [20.0, 2.7]   |
        |   | 12.0|  2.9  | [12.0, 2.9]   |
        |   +-----+-------+----------------+
        |""".stripMargin)
    // StandardScaler处理
    var featureCol = Array("price","weight")
    val assemble = new VectorAssembler()
      .setInputCols(featureCol)
      .setOutputCol("scaledFeatures")

    println(
      """
        |进行---------------------------------------------------------------
        |标准化(StandardScaler)
        |   setInputCol                                       设置输入列
        |   setOutputCol                                      设置输出列
        |   setWithStd                                        设置标准差（标准差为true），表示要在标准化过程中考虑特征的标准差。标准差是一个衡量数据分散程度的指标，标准化会将特征缩放到具有单位标准差。
        |   setWithMean                                       设置均值（均值标志为false），表示在标准化过程中不考虑特征的均值，如果设置为true，则会将特征缩放到具有零均值
        |这段代码的目的是使用 StandardScaler 对 "scaledFeatures" 列进行标准化处理，使其具有单位标准差，同时不进行均值的调整。标准化是一种数据预处理技术，有助于确保不同尺度的特征对机器学习模型的影响是一致的，从而提高模型的性能。标准化后的结果将存储在新的列 "scaledFeatures_result" 中。
        |处理结果展示：
        |   scaledFeatures|scaledFeatures_result
        |   [2220.0,0.24] |[0.759105832811737,0.15528316608735387]
        |   [3321.0,15.24]|[1.135581293138639,9.86048104654697]
        |   [3100.0,15.24]|[1.060012649421795,9.86048104654697]
        |""".stripMargin)
    val scaler = new StandardScaler()
      .setInputCol("scaledFeatures")
      .setOutputCol("scaledFeatures_result")
      .setWithStd(true)
      .setWithMean(false)

    println(
      """
        |进行------------------------------------------------------------------------
        |VectorAssembler（组合列）
        |结果：(42,[0,1,2,5,18,36],[1.0,0.759105832811737,0.15528316608735387,1.0,1.0,1.0])
        |   42                                                              这是整个稀疏向量的长度，表示有42个位置（或特征）。
        |   [0, 1, 2, 5, 18, 36]                                            这是一个包含非零值的位置索引数组。它告诉我们在稀疏向量中的哪些位置有非零值。在这个例子中，分别有非零值的位置是 0、1、2、5、18 和 36。
        |   [1.0, 0.759105832811737, 0.15528316608735387, 1.0, 1.0, 1.0]    这是与上述位置索引数组中相应位置对应的值数组。它告诉我们每个非零位置的值。例如，位置0的值是1.0，位置1的值是0.759105832811737，以此类推。
        |我们可以通过下标来取出这42个值
        |""".stripMargin)
    // 输出到一列
    featureCol = Array("id","scaledFeatures_result")++columns.map(x => x + "_onehot")
    val assemble1 = new VectorAssembler()
      .setInputCols(featureCol)
      .setOutputCol("features")

    val pipeline_frame = new Pipeline().setStages(indexers++Array(onehoter,assemble,scaler,assemble1)).fit(frame).transform(frame)

    val spark = util.getSparkSession
    println("""导入隐式转换""")
    import spark.implicits._

    println(
      """
        |进行-----------------------------------------------------------------------------------------------
        |输出一行十列
        |   asInstanceOf                                    强制类型转换为向量
        |   map1                                            遍历每一行
        |   x                                               由于是row类型的,所以强制转换为向量
        |   toArray                                         转换为数组，数组包含了一行的元素
        |   map2                                            遍历每一行
        |   take(10)                                        取出每一行前十列
        |   mkString                                        将每一行前十列拼接成一个字符串
        |   rdd                                             转成一个rdd
        |   collect()(0)                                    将数据返回到客户端，拿出第一行
        |   println                                         打印
        |""".stripMargin)
    println(pipeline_frame.select("features").map(x => x(0).asInstanceOf[Vector].toArray).map(_.take(10).mkString(",")).rdd.collect()(0))

    pipeline_frame

  }

  def test03(gs_1_util: ds_demo01_util, frame:DataFrame, string: String): Unit = {

    val spark = gs_1_util.getSparkSession
    import spark.implicits._

    frame.select("id", "features").createOrReplaceTempView("t1")

    //由于6708用户购买商品很少，所以模拟7条数据
    val string1 = string + "1,2,3,4,5,6,7"

    val user6708Features = gs_1_util.getSparkSession.sql(s"""select * from t1 where id in (${string1})""").map {
      case Row(id: Long, vector: Vector) => (id, vector)
    }.collect()

    val otherUserFeatures = gs_1_util.getSparkSession.sql(s"""select * from t1 where id not in (${string1})""").map {
      case Row(id: Long, vector: Vector) => (id, vector)
    }.collect()
    otherUserFeatures.foreach(line=>{
      println(line._1)
      println(line._2)
    })
    calculateAndPrintTop5(user6708Features, otherUserFeatures)

  }
  def main(args: Array[String]): Unit = {

    val gs_1_util = new ds_demo01_util

//    test01(gs_1_util)

//    test02(gs_1_util)

    test03(gs_1_util,test02(gs_1_util),test01(gs_1_util))

  }

}
