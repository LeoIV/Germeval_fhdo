package de.hoevelmann.bachelorthesis.modelling.transformers

import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.ml.linalg.{SQLDataTypes, Vectors}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Transformer, linalg}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, functions}

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 10.08.2017.
  */
class KMeansTransformer(override val uid: String) extends Transformer {

  def this() = this(Identifiable.randomUID("KMeansTransformer"))

  val inputCol: Param[String] = new Param[String](this, "inputCol", "the input col", (x: String) => x.nonEmpty)

  def setInputCol(value: String): this.type = set(inputCol, value)

  def getInputCol: String = $(inputCol)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def getOutputCol: String = $(outputCol)

  val outputCol: Param[String] = new Param[String](this, "outputCol", "the output col", (x: String) => x.nonEmpty)


  override def transform(dataset: Dataset[_]): DataFrame = {

    val kMeansModel: KMeansModel = KMeansModel.load("kMeansModel").setPredictionCol("indexPrediction").setFeaturesCol($(inputCol))
    val vectorSize = kMeansModel.getK
    val dfWithIdx = kMeansModel.transform(dataset)

    def idxToOneHot(idx: Int): linalg.Vector = Vectors.sparse(vectorSize, Seq((idx, 1.0)))

    val idxToOneHotUdf = functions.udf(idxToOneHot _)

    dfWithIdx.select(dfWithIdx("*"), idxToOneHotUdf(dfWithIdx("indexPrediction")).as($(outputCol))).drop("indexPrediction")

  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema.add($(outputCol), SQLDataTypes.VectorType)
}

