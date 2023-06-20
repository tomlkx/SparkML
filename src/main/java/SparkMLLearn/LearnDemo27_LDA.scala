package SparkMLLearn

import org.apache.spark.ml.clustering.{LDA, LDAModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object LearnDemo27_LDA {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("demo").master("local[*]").getOrCreate()
    // 加载数据集，格式为libsvm
    val data: DataFrame = spark.read.format("libsvm").load("data/mllib/sample_lda_libsvm_data.txt")

    // 创建LDA对象并设置参数
    val lda: LDA = new LDA()
      .setK(10) // 设置主题数为10
      .setMaxIter(10) // 设置最大迭代次数为10
    // 使用数据集训练LDA模型
    val model: LDAModel = lda.fit(data)

    // 计算模型对数据集的对数似然度和对数困惑度
    /**
     * 对数似然度是用于衡量LDA模型对给定数据的拟合程度。
     * 它基于每个文档中的词语出现概率和主题分布的组合来计算。
     * 对数似然度越高，表示模型拟合数据越好。
     * 在LDA中，通常使用对数似然度来评估训练好的模型在训练数据上的拟合程度。
     */
    val ll: Double = model.logLikelihood(data)
    /**
     * 对数困惑度是用于评估LDA模型在新数据上的预测性能。
     * 它衡量了模型对未见过的数据的预测能力和泛化能力。
     * 对数困惑度越低，表示模型在新数据上的预测性能越好。
     *  在LDA中，通常使用对数困惑度来评估训练好的模型在新数据上的表现。
     */
    val lp: Double = model.logPerplexity(data)
    println(s"Log Likelihood: $ll")
    println(s"Log Perplexity: $lp")

    // 描述每个主题的前3个关键词
    val topic: DataFrame = model.describeTopics(3)
    topic.show()

    // 对数据集进行主题转换
    val transformed: DataFrame = model.transform(data)
    transformed.show()
    spark.stop()
  }
}
