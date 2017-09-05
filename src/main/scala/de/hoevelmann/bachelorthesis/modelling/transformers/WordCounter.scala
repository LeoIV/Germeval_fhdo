package de.hoevelmann.bachelorthesis.modelling.transformers

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, _}

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 16.06.2017.
  */
class WordCounter(override val uid: String) extends Transformer with DefaultParamsWritable {

  val inputCol: Param[String] = new Param[String](this, "inputCol", "the input column", (x: String) => x.nonEmpty)

  def setInputCol(value: String): this.type = set(inputCol, value)

  def getInputCol: String = $(inputCol)

  val outputCol: Param[String] = new Param[String](this, "outputCol", "the output column", (x: String) => x.nonEmpty)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def getOutputCol: String = $(outputCol)

  override def transform(dataset: Dataset[_]): DataFrame = {

    val df = dataset.toDF
    val wordCountRdd = df.select($(inputCol)).rdd.map(row => row.getAs[Seq[String]]($(inputCol)).length.toDouble)
    val numRows = wordCountRdd.count().toDouble
    val averageReviewLength = wordCountRdd.reduce((a, b) => a + b) / numRows

    def computeReviewLength(words: Seq[String]): Double = words.length.toDouble / averageReviewLength

    val computeReviewLengthUdf = functions.udf(computeReviewLength _)

    df.select(df("*"), computeReviewLengthUdf(df($(inputCol))).as($(outputCol)))

  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema.add($(outputCol), DoubleType)

  def this() = this(Identifiable.randomUID("wordCounter"))
}


