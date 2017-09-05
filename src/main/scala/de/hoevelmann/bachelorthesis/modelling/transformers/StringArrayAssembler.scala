package de.hoevelmann.bachelorthesis.modelling.transformers

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql._
import org.apache.spark.sql.types.{ArrayType, StringType, StructType}

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 19.06.2017.
  */
class StringArrayAssembler(val uid: String) extends Transformer with DefaultParamsWritable {

  val inputCols: Param[Array[String]] = new Param[Array[String]](this, "inputCols", "the two input columns", (x: Array[String]) => (2 to 5).contains(x.length))

  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  def getInputCols: Array[String] = $(inputCols)

  val outputCol: Param[String] = new Param[String](this, "outputCol", "the output column", (x: String) => x.nonEmpty)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def getOutputCol: String = $(outputCol)


  override def transform(dataset: Dataset[_]): DataFrame = {

    //TODO this function currently supports only two to five columns.
    val df = dataset.toDF

    def merge2Arrays(array1: Seq[String], array2: Seq[String]): Seq[String] = array1 ++ array2

    def merge3Arrays(array1: Seq[String], array2: Seq[String], array3: Seq[String]): Seq[String] = array1 ++ array2 ++ array3

    def merge4Arrays(array1: Seq[String], array2: Seq[String], array3: Seq[String], array4: Seq[String]): Seq[String] = array1 ++ array2 ++ array3 ++ array4

    def merge5Arrays(array1: Seq[String], array2: Seq[String], array3: Seq[String], array4: Seq[String], array5: Seq[String]): Seq[String] = array1 ++ array2 ++ array3 ++ array4 ++ array5

    val mergeArrays2Udf = functions.udf(merge2Arrays _)
    val mergeArrays3Udf = functions.udf(merge3Arrays _)
    val mergeArrays4Udf = functions.udf(merge4Arrays _)
    val mergeArrays5Udf = functions.udf(merge5Arrays _)

    val transformedDf: DataFrame =
      if ($(inputCols).length == 2) df.select(df("*"), mergeArrays2Udf(df($(inputCols)(0)), df($(inputCols)(1))).as($(outputCol)))
      else if ($(inputCols).length == 3) df.select(df("*"), mergeArrays3Udf(df($(inputCols)(0)), df($(inputCols)(1)), df($(inputCols)(2))).as($(outputCol)))
      else if ($(inputCols).length == 4) df.select(df("*"), mergeArrays4Udf(df($(inputCols)(0)), df($(inputCols)(1)), df($(inputCols)(2)), df($(inputCols)(3))).as($(outputCol)))
      else df.select(df("*"), mergeArrays5Udf(df($(inputCols)(0)), df($(inputCols)(1)), df($(inputCols)(2)), df($(inputCols)(3)), df($(inputCols)(4))).as($(outputCol)))


    transformedDf

  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    schema.add($(outputCol), new ArrayType(StringType, false))
  }

  def this() = this(Identifiable.randomUID("CacherTransformer"))
}