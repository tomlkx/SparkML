package SparkMLLearn

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.{ALS, ALSModel}

import scala.Console.println

object LearnDemo31_ALS {
  def main(args: Array[String]): Unit = {
    // 创建 SparkSession
    val spark: SparkSession = SparkSession.builder().appName("MovieRecommendation").master("local[*]").getOrCreate()
    import spark.implicits._

    // 定义函数用于解析数据
    def parseRating(str: String): Rating = {
      val fields: Array[String] = str.split("::")
      assert(fields.size == 4)
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
    }

    // 读取数据并解析为 Dataset[Rating]
    val data: Dataset[String] = spark.read.textFile("data/mllib/als/sample_movielens_ratings.txt")
    val ratings: DataFrame = data.map(parseRating).toDF()

    // 划分训练集和测试集
    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

    // 创建 ALS 模型并设置参数
    val als: ALS = new ALS()
      .setMaxIter(5) // 设置最大迭代次数
      .setRegParam(0.01) // 设置正则化参数
      .setUserCol("userId") // 设置用户列名
      .setItemCol("movieId") // 设置物品列名
      .setRatingCol("rating") // 设置评分列名

    // 在训练集上拟合 ALS 模型
    val model: ALSModel = als.fit(training)
    model.setColdStartStrategy("drop") // 设置冷启动策略为"drop"，即对新用户或新物品进行预测时，将返回NaN

    // 在测试集上进行预测并评估模型的性能
    val prediction: DataFrame = model.transform(test)
    val evaluator: RegressionEvaluator = new RegressionEvaluator()
      .setLabelCol("rating") // 设置标签列名
      .setPredictionCol("prediction") // 设置预测列名
      .setMetricName("rmse") // 设置评估指标为 RMSE（均方根误差）
    val rmse: Double = evaluator.evaluate(prediction)
    println("Root Mean Squared Error (RMSE): " + rmse)

    // 为所有用户生成推荐结果
    val userRecs: DataFrame = model.recommendForAllUsers(10)

    // 为所有电影生成推荐结果
    val movieRecs: DataFrame = model.recommendForAllItems(10)
    // 从评分数据中选择不重复的用户列，并限制为3行，用于生成用户子集的推荐结果
    val users: Dataset[Row] = ratings.select(als.getUserCol).distinct().limit(3)

    // 对用户子集进行推荐，生成每个用户的前10个推荐结果
    val userSubsetRecs = model.recommendForUserSubset(users, 10)

    // 从评分数据中选择不重复的物品列，并限制为3行，用于生成物品子集的推荐结果
    val movies = ratings.select(als.getItemCol).distinct().limit(3)

    // 对物品子集进行推荐，生成每个物品的前10个推荐结果
    val movieSubSetRecs = model.recommendForItemSubset(movies, 10)

    spark.stop()
  }

  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
}
