package de.hoevelmann.bachelorthesis.modelling.transformers

import java.io.{File, FileWriter}

import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types._

import scala.sys.process._
import scala.util.{Random, Try}

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 16.06.2017.
  */
class FastTextClassifier(override val uid: String) extends Predictor[Seq[String], FastTextClassifier, FastTextClassificationModel] with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("fastTextClassifier"))

  override protected def train(dataset: Dataset[_]): FastTextClassificationModel = {

    val reviewsWithLabels: Array[(String, Double)] = dataset.toDF().rdd.collect.map(row => {
      val review = row.getAs[Seq[String]]($(featuresCol)).mkString(" ")
      val label = row.getAs[Double]($(labelCol))
      (review, label)
    })

    val modelFile = java.util.UUID.randomUUID.toString


    val ftFw = new FileWriter("test.txt")
    reviewsWithLabels.foreach(reviewWithLabel =>
      ftFw.write(reviewWithLabel._1 + " __label__" + reviewWithLabel._2 + "\n")
    )
    ftFw.close()
//-pretrainedVectors germeval/corpus/wholeModel.vec
    val systemCallString = "fasttext supervised -input test.txt -output " + modelFile
    systemCallString.lineStream_!
    new File("test.txt").delete()


    val res = new FastTextClassificationModel()
    res.setFastTextModelPath(modelFile)
    res


  }

  override def copy(extra: ParamMap): FastTextClassifier = defaultCopy(extra)

  override def featuresDataType: DataType = new ArrayType(org.apache.spark.sql.types.StringType, false)

}

class FastTextClassificationModel(override val uid: String) extends PredictionModel[Seq[String], FastTextClassificationModel] {

  def this() = this(Identifiable.randomUID("fastTextClassifier"))

  val fastTextModelPath = new Param[String](this, "fastTextModelPath", "the model path for fasttext")

  def setFastTextModelPath(value: String) = set(fastTextModelPath, value)

  def getFastTextModelPath: String = $(fastTextModelPath)

  override protected def predict(features: Seq[String]): Double = {
    FastTextClassificationModel.predict(List(features), $(fastTextModelPath)).head
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF()
    val spark = SparkSession.builder().getOrCreate()
    val rowSchema = new StructType()
      .add("index", LongType)
      .add($(predictionCol), DoubleType)
    val reviews: Array[Seq[String]] = df.collect().map(row => row.getAs[Seq[String]]($(featuresCol)))
    val predictions = FastTextClassificationModel.predict(reviews, $(fastTextModelPath))
      .zipWithIndex
      .map(prediction => new GenericRowWithSchema(Array(prediction._2.toLong, prediction._1), rowSchema).asInstanceOf[Row])
    val predictionsDf = spark.createDataFrame(spark.sparkContext.parallelize(predictions), rowSchema).as("predictionsDf")

    val schema = df.schema.add("index", LongType)

    val indexedDf = spark.createDataFrame(df.rdd
      .zipWithIndex()
      .map(ri => new GenericRowWithSchema(ri._1.toSeq.toArray :+ ri._2, schema).asInstanceOf[Row]), schema).as("indexedDf")

    indexedDf.join(predictionsDf, col("indexedDf.index") === col("predictionsDf.index"), "inner").drop("index")


    // TODO add the systemCallResult column to the dataset
  }

  override def copy(extra: ParamMap): FastTextClassificationModel = defaultCopy(extra)

  override def featuresDataType: DataType = new ArrayType(org.apache.spark.sql.types.StringType, false)

}

private object FastTextClassifier {

  private val monitor = "monitor"

}

private object FastTextClassificationModel {

  def predict(features: Seq[Seq[String]], fastTextModelPath: String): Seq[Double] = {
    synchronized {
      val predictionFw = new FileWriter("test.txt")
      features.foreach(feature => {
        predictionFw.write(feature.mkString(" ") + "\n")
      })
      predictionFw.close()

      val systemCallString = "fasttext predict " + fastTextModelPath + ".bin test.txt"
      val systemCallStream = systemCallString.lineStream_!
      val result: Seq[Double] = systemCallStream.toList.map(_.replace("__label__", "").toDouble)
      new File("test.txt").delete()
      return result
    }
  }
}