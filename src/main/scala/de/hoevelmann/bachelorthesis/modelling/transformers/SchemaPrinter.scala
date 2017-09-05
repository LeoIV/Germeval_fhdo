package de.hoevelmann.bachelorthesis.modelling.transformers

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 19.06.2017.
  */
class SchemaPrinter(val uid: String) extends Transformer with DefaultParamsWritable {
  override def transform(dataset: Dataset[_]): DataFrame = {
      dataset.toDF().show(10)
    dataset.toDF()
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema

  def this() = this(Identifiable.randomUID("CacherTransformer"))
}