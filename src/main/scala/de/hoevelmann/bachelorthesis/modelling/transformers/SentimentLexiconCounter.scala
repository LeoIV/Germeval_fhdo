package de.hoevelmann.bachelorthesis.modelling.transformers

import org.apache.spark.ml.linalg.SQLDataTypes._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Transformer, linalg}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.mutable
import scala.io.Source

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 09.06.2017.
  */
class SentimentLexiconCounter(override val uid: String) extends Transformer {

  def this() = this(Identifiable.randomUID("SentimentLexiconCounter"))

  val inputCol: Param[String] = new Param[String](this, "inputCol", "the input col", (x: String) => x.nonEmpty)

  def setInputCol(value: String): this.type = set(inputCol, value)

  def getInputCol: String = $(inputCol)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def getOutputCol: String = $(outputCol)

  def featuresLength =  2

  val outputCol: Param[String] = new Param[String](this, "outputCol", "the output col", (x: String) => x.nonEmpty)

  private val positive: List[(String, Int)] = Source.fromFile("germeval/corpus/sentiws_positive.txt").getLines.toList.map(line => line.split("\t")(0)).zipWithIndex

  private val negative: List[(String, Int)] = Source.fromFile("germeval/corpus/sentiws_negative.txt").getLines.toList.map(line => line.split("\t")(0)).zipWithIndex

  override def transform(dataset: Dataset[_]): DataFrame = {

    println("sentiment lexicon")

    def createSentiVec(words: Seq[String]): linalg.Vector = {
      val hash = words.hashCode()
      if (SentimentLexicon.cache.contains(hash))
        SentimentLexicon.cache(hash)
      else {
        val positiveCount: Double = positive.filter(positiveWord => words.contains(positiveWord._1.toLowerCase)).map(_ => 1.0).sum
        val negativeCount: Double = negative.filter(negativeWord => words.contains(negativeWord._1.toLowerCase)).map(_ => 1.0).sum
        val assembled: Array[Double] = Array(positiveCount, negativeCount)
        val result = Vectors.dense(assembled)
        SentimentLexicon.cache.put(hash, result)
        result
      }
    }

    val sentiVecUdf = udf(createSentiVec _)

    val df = dataset.toDF.cache

    df.select(df("*"), sentiVecUdf(df($(inputCol))).as($(outputCol)))

  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema.add($(outputCol), VectorType)
}

private object SentimentLexiconCounter {
  val cache: mutable.HashMap[Int, linalg.Vector] = scala.collection.mutable.HashMap[Int, linalg.Vector]()
}