package de.fhdo.germeval.modelling.transformers

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{DataType, DoubleType}

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 11.08.2017.
  */
class FloatColumnToDoubleColumn(override val uid: String) extends UnaryTransformer[Float, Double, FloatColumnToDoubleColumn] {

  def this() = this(Identifiable.randomUID("FloatColumnToDoubleColumn"))
  override protected def createTransformFunc: (Float) => Double = _.toDouble

  override protected def outputDataType: DataType = DoubleType
}
