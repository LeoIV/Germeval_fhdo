package de.fhdo.germeval.modelling.transformers

import org.apache.spark.api.java.function.FilterFunction
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 20.06.2017.
  */
class RowPruner(override val uid: String, retainCondition: FilterFunction[Row])
  extends Transformer {

  private def retCond = retainCondition

  override def transform(dataset: Dataset[_]): DataFrame = dataset.toDF().filter(retainCondition)

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema

  def this(retainCondition: FilterFunction[Row]) = this(Identifiable.randomUID("rowPruner"), retainCondition)

}