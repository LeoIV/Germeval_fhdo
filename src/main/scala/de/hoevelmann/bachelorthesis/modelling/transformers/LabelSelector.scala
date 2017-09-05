package de.hoevelmann.bachelorthesis.modelling.transformers

import org.apache.spark.ml.{Transformer, _}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}


/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 20.06.2017.
  *
  * Slices the data in one column in a given number of partitions and adds an column for each partition.
  */
class LabelSelector(override val uid: String)
  extends Transformer with DefaultParamsWritable {

  val predictionColumnPrefix: Param[String] = new Param[String](this, "predictionColumnPrefix", "the prefix for the prediction columns", (x: String) => x.nonEmpty)

  def setPredictionColumnPrefix(value: String): this.type = set(predictionColumnPrefix, value)

  def getPredictionColumnPrefix: String = $(predictionColumnPrefix)

  val predictionCol: Param[String] = new Param[String](this, "outputCol", "the name of the generated prediction", (x: String) => x.nonEmpty)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def getPredictionCol: String = $(predictionCol)

  override def transform(dataset: Dataset[_]): DataFrame = {

    val df = dataset.toDF

    def mergePredictions(a0: linalg.Vector, a1: linalg.Vector, a2: linalg.Vector, a3: linalg.Vector, a4: linalg.Vector,
                         a5: linalg.Vector, a6: linalg.Vector, a7: linalg.Vector, a8: linalg.Vector, a9: linalg.Vector): Double = {
      val reducedVector: linalg.Vector = Seq[linalg.Vector](a0, a1, a2, a3, a4, a5, a6, a7, a8, a9)
        .reduce((a, b) => Vectors.dense(a.toArray.zip(b.toArray).map(x => x._1 + x._2)))
      reducedVector.toArray.zipWithIndex.maxBy(_._1)._2.toDouble
    }

    val udfMergePredictions = udf(mergePredictions _)

    val resultDf = df.select(df("*"), udfMergePredictions(df($(predictionColumnPrefix) + "_" + 0), df($(predictionColumnPrefix) + "_" + 1), df($(predictionColumnPrefix) + "_" + 2), df($(predictionColumnPrefix) + "_" + 3),
      df($(predictionColumnPrefix) + "_" + 4), df($(predictionColumnPrefix) + "_" + 5), df($(predictionColumnPrefix) + "_" + 6), df($(predictionColumnPrefix) + "_" + 7), df($(predictionColumnPrefix) + "_" + 8),
      df($(predictionColumnPrefix) + "_" + 9)).alias($(predictionCol)))
    resultDf

  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema.add($(predictionCol), DoubleType, nullable = false)

  def this() = this(Identifiable.randomUID("labelSelector"))

}