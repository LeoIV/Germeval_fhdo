package de.fhdo.germeval.modelling.transformers

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 19.06.2017.
  */
class LowerCaser(override val uid: String) extends UnaryTransformer[String, String, LowerCaser] {


  def this() = this(Identifiable.randomUID("lowerCaser"))


  override protected def createTransformFunc: (String) => String = {
    _.toLowerCase
  }

  override protected def outputDataType: DataType = StringType
}
