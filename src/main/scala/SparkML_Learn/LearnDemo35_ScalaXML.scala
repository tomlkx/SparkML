package SparkML_Learn

import scala.util.Try
import scala.xml.XML

object LearnDemo35_ScalaXML {
  def main(args: Array[String]): Unit = {
    val xmlString =
      """
    <bookstore>
      <book>
        <title>Scala for Beginners</title>
        <author>John Doe</author>
        <price>19.99</price>
      </book>
      <book>
        <title>Advanced Scala Programming</title>
        <author>Jane Smith</author>
        <price></price>
      </book>
    </bookstore>
    """
    val xml = XML.loadString(xmlString)
    // 使用模式匹配提取数据
    val books = (xml \ "book").map { book =>
      val title = (book \ "title").text
      val author = (book \ "author").text
      /**
       * Try 是 Scala 中的一个封装类型，用于处理可能抛出异常的操作。
       * 它有两个子类：Success 和 Failure，分别表示操作成功和操作失败的情况。
       * Try 使得在进行可能抛出异常的操作时，能够更加优雅地处理异常情况，而不是使用传统的异常处理机制。
       */
      val price = Try((book \ "price").text.toDouble).getOrElse()
      //这种情况就会报错
      //val price = (book \ "price").text.toDouble
      (title, author, price)
    }
    // 打印提取的数据
    for ((title, author, price) <- books) {
      println(s"Title: $title, Author: $author, Price: $price")
    }
  }
}
