/* Solution.scala */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.SparkConf

import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification._

import org.sqlite.JDBC

import java.sql.DriverManager

object Solution {
  val uri = "jdbc:sqlite:database.sqlite"

  val seriousCount = 30100
  val sarcasmCount = 30100
  val numFeatures = 5000

  /** Fixes the database schema
    *
    * The schema of the provided database.sqlite file lacks type signatures
    * on some columns, which causes sqlite to report them as typeless, which
    * Sqlite JDBC treats as "type 0" (implicit conversion), which JDBC
    * interprets as NULL types, which causes spark sql to throw an exception.
    * Because user-provided signatures are not available with spark sql's
    * DefaultSource, manually overriding the types in sqlite seems to be the
    * least invasive option to make spark understand the database.
    *
    * This function forcibly changes the schema of the 'May2015' table,
    * annotating each previously-unannotated column explicitly as BLOB.
    * Note that this keeps the affinity, so no data loss occurs, but this
    * causes the 'table_info' PRAGMA to return the correct types.
    */
  def unsafeFixSchema() {
    val conn = DriverManager.getConnection(uri)
    val stmt = conn.createStatement()
    stmt.execute("PRAGMA writable_schema = true")
    stmt.execute("UPDATE sqlite_master SET sql = '" +
      "CREATE TABLE May2015(" +
        "created_utc INTEGER," +
        "ups INTEGER," + 
        "subreddit_id BLOB," +
        "link_id BLOB," +
        "name BLOB," +
        "score_hidden BLOB," +
        "author_flair_css_class BLOB," +
        "author_flair_text BLOB," +
        "subreddit BLOB," +
        "id BLOB," +
        "removal_reason BLOB," +
        "gilded int," +
        "downs int," +
        "archived BLOB," +
        "author BLOB," +
        "score int," +
        "retrieved_on int," +
        "body TEXT," +
        "distinguished BLOB," +
        "edited BLOB," +
        "controversiality int," +
        "parent_id BLOB)' " +
      "WHERE name='May2015'")
    stmt.execute("PRAGMA writable_schema = false")
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Solution")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    
    Class.forName("org.sqlite.JDBC")

    unsafeFixSchema()

    val df = sqlContext.read.format("jdbc")
      .options(Map(
          "url" -> uri,
          "dbtable" -> "May2015"
        ))
      .load()

    df.printSchema()

    val sarcasm = df
      .filter(df("body").endsWith(" /s"))
      .withColumn("label", lit(1.0))
      .limit(sarcasmCount)

    val serious = df
      .filter(!df("body").contains("/s"))
      .withColumn("label", lit(0.0))
      .limit(seriousCount)

    val together = sarcasm.unionAll(serious)

    val Array(trainData, holdoutData) = together.randomSplit(Array(0.67, 0.33))

    val tokenizer = new Tokenizer()
      .setInputCol("body")
      .setOutputCol("words")

    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filteredWords")
      .setStopWords(Array("/s"))

    val hashingTF = new HashingTF()
      .setInputCol("filteredWords")
      .setOutputCol("rawFeatures")
      .setNumFeatures(numFeatures)

    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setMaxIter(100)
      .setRegParam(1/1.25)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, hashingTF, idf, lr))

    val model = pipeline.fit(trainData)

    val predictions = model.transform(holdoutData)

    println(predictions.agg(sum(abs(col("prediction") - col("label"))), count(col("label"))).show())

    predictions
      .filter(expr("prediction = 1.0"))
      .filter(expr("probability > 0.8"))
      .show()
  }
}
