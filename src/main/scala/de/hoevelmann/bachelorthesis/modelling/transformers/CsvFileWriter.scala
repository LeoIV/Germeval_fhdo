package de.hoevelmann.bachelorthesis.modelling.transformers

import java.io.FileWriter

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

import scala.util.Random

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 08.08.2017.
  */
class CsvFileWriter(val uid: String) extends Transformer with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("CsvFileWriter"))

  val featuresColumn: Param[String] = new Param[String](this, "featuresColumn", "the featuresColumn", (x: String) => x.nonEmpty)

  val labelColumn: Param[String] = new Param[String](this, "labelColumn", "the labelColumn")

  val outputFilepath: Param[String] = new Param[String](this, "outputFilepath", "the output filepath")

  def setFeaturesColumn(value: String): this.type = set(featuresColumn, value)

  def getFeaturesColumn: String = $(featuresColumn)

  def setLabelColumn(value: String): this.type = set(labelColumn, value)

  def getLabelColumn: String = $(labelColumn)

  def setOutputFilepath(value: String): this.type = set(outputFilepath, value)

  def getOutputFilepath: String = $(outputFilepath)

  setDefault(labelColumn, "")

  override def transform(dataset: Dataset[_]): DataFrame = {
    println("writing")
    val fw = new FileWriter(if ($(outputFilepath).isEmpty) "C:/Users/hoevelmann/Downloads/MATRIX" + Random.nextInt + ".csv" else $(outputFilepath))
    if ($(labelColumn).nonEmpty) {
      val tuples = dataset.toDF().rdd.collect().map(row => (row.getAs[org.apache.spark.ml.linalg.Vector]($(featuresColumn)), row.getAs[Int]($(labelColumn))))
      tuples.foreach(tuple => {
        fw.write(tuple._2 + "," + tuple._1.toArray.mkString(",") + "\n")
      })
    }
    else {
      val features = dataset.toDF().rdd.collect().map(row => row.getAs[org.apache.spark.ml.linalg.Vector]($(featuresColumn)))
      features.foreach(features => {
        fw.write(features.toArray.mkString(",") + "\n")
      })
    }
    fw.close()
    dataset.toDF
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema
}
