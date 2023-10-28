package SparkML_Learn

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object LearnDemo38_JieKaDe {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[8]").getOrCreate()
    //使用杰卡特计算两个 集合物品相似度
    spark.udf.register("xxx",(v1:Vector,v2:Vector)=>{
      // 转换第一个向量 v1 到一个 Double 数组
      val array1: Array[Double] = v1.toArray
      // 转换第二个向量 v2 到一个 Double 数组
      val array2: Array[Double] = v2.toArray
      // 计算两个数组的交集大小，即共同元素的数量
      val size1: Int = array1.intersect(array2).size
      // 计算两个数组的并集大小，即所有不重复元素的数量
      val size2: Int = array1.union(array2).size
      // 计算 Jaccard 相似度，即交集大小除以并集大小
      size1.toDouble / size2.toDouble
    })
    import org.apache.spark.sql.functions._
    import spark.implicits._
    val value: RDD[Vector] = spark.sparkContext.parallelize(Seq(
      Vectors.dense(1, 2, 3, 8, 9, 10),
      Vectors.dense(4, 7, 8, 9, 3),
      Vectors.dense(11, 7, 8, 9, 13)
    ))
    value.map(line=>(line,Vectors.dense(1, 2, 3, 4, 5, 6)))
      .toDF("v1","v2")
      .withColumn("xsd",expr("xxx(v1,v2)"))
      .show()
    spark.stop()
  }
}
