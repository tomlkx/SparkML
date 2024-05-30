package SparkML_Demo_2024

import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.sql.SparkSession

object SVDExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("SVDExample")
      .master("local[*]")
      .getOrCreate()

    val data = Seq(
      (1, 0, 1, 0, 1, 0,  1),
      (2, 0, 0, 1, 0, 0, 0),
      (3, 1, 0, 1, 0, 0, 0),
      (4, 0, 0, 1, 0, 1, 0),
      (5, 0, 1, 0, 0, 0, 0),
      (6, 1, 0, 0, 0, 0, 0),
      (7, 1, 0, 1, 0, 0, 0)
    )

    import spark.implicits._
    val df = data.toDF("user_id", "sku_id1", "sku_id2", "sku_id3", "sku_id4", "sku_id5", "sku_id6")

    // Convert DataFrame to RDD of vectors
    val vectors = df.drop("user_id").rdd.map { row =>
      Vectors.dense(row.toSeq.toArray.map(_.toString.toDouble))
    }

    // Create a RowMatrix
    val mat = new RowMatrix(vectors)

    // Perform SVD
    val k = 5 // Number of singular values to keep
    val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(k, computeU = true)

    // Extract U, Σ, and V^T matrices
    val U: RowMatrix = svd.U
    val s = svd.s // Singular values
    val V: Matrix = svd.V

    print(V)

    // Compute cosine similarity
    val user1 = 0 // Assuming user 1
    val user1Features = U.rows.collect()(user1)
    val user1Purchased = Seq("sku_id1", "sku_id3", "sku_id5").map(df.columns.indexOf).filter(_ != -1)
    val user1NotPurchased = Seq("sku_id2", "sku_id4", "sku_id6").map(df.columns.indexOf).filter(_ != -1)
    println(user1Purchased)
    val user1PurchasedFeatures = V.toArray.grouped(V.numCols).map { item =>
      user1Purchased.map(item).toArray
    }.toArray
    val user1NotPurchasedFeatures = V.toArray.grouped(V.numCols).map { item =>
      user1NotPurchased.map(item(_)).toArray
    }.toArray

    // Calculate cosine similarity between purchased and not purchased items
    val cosineSimilarity = user1PurchasedFeatures.map { purchased =>
      user1NotPurchasedFeatures.map { notPurchased =>
        val dotProduct = purchased.zip(notPurchased).map { case (a, b) => a * b }.sum
        val purchasedNorm = math.sqrt(purchased.map(a => a * a).sum)
        val notPurchasedNorm = math.sqrt(notPurchased.map(a => a * a).sum)
        dotProduct / (purchasedNorm * notPurchasedNorm)
      }.sum / user1NotPurchasedFeatures.length
    }.sum / user1PurchasedFeatures.length

    println(s"Average cosine similarity between purchased and not purchased items for user 1: $cosineSimilarity")

    spark.stop()
  }
}