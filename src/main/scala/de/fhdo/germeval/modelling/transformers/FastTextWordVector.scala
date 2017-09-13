package de.fhdo.germeval.modelling.transformers

import de.fhdo.germeval.modelling.fasttext.FastText
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.SQLDataTypes._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 16.06.2017.
  */
class FastTextWordVector(override val uid: String) extends Transformer with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("fasttextwordvectors"))

  val fastTextModelPath: Param[String] = new Param[String](this, "fastTextModelPath", "the path to the fasttext model", (x: String) => x.nonEmpty)

  val inputCol = new Param[String](this, "inputCol", "The input column")
  val outputCol = new Param[String](this, "outputCol", "The output column")

  def setInputCol(value: String): this.type = set(inputCol, value)

  def getInputCol: String = $(inputCol)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def getOutputCol: String = $(outputCol)

  def setFastTextModelPath(value: String): this.type = set(fastTextModelPath, value)

  def getFastTextModelPath: String = $(fastTextModelPath)

  override def transform(dataset: Dataset[_]): DataFrame = {

    val sparkSession = SparkSession.builder().getOrCreate()

    var wordCollector: List[String] = List[String]()
    var vectorLengthCollector: List[Int] = List[Int]()

    dataset.toDF().rdd.collect.foreach(row => {
      val wordsInRow = row.getAs[Seq[String]]($(inputCol))
      wordCollector = wordCollector ++ wordsInRow
      vectorLengthCollector = vectorLengthCollector :+ wordsInRow.length
    })
    var fastTextVectors: Seq[Array[Float]] = FastText.wordToVectorBatch(wordCollector, $(fastTextModelPath))
    val vectorSequenceSchema = StructType(
      Array(
        StructField($(outputCol), new ArrayType(VectorType, false), nullable = false),
        StructField("index", IntegerType, nullable = false)
      )
    )
    var vectorSequencesDs = sparkSession.createDataFrame(sparkSession.sparkContext.emptyRDD[Row], vectorSequenceSchema)
    vectorSequencesDs.persist()
    var i = 0
    for (currentNumberOfWords <- vectorLengthCollector) {
      val (currentSequence, remainingVectors) = fastTextVectors.splitAt(currentNumberOfWords)
      fastTextVectors = remainingVectors


      val rowRdd: RDD[Row] = {
        val row = new GenericRowWithSchema(Array(Array(currentSequence.map(x => Vectors.dense(x.map(_.toDouble)))), i), vectorSequenceSchema)
        sparkSession.sparkContext.parallelize(Array(row))
      }
      i += 1
      vectorSequencesDs = vectorSequencesDs.union(sparkSession.createDataFrame(rowRdd, vectorSequenceSchema))
      println(vectorSequencesDs.count())
    }
    println("DATASET LENGTH: " + dataset.toDF().rdd.collect().length)

    val schema = StructType(dataset.toDF().schema.fields ++ Array(StructField("index", IntegerType, nullable = false)))
    val df: DataFrame = sparkSession.createDataFrame(dataset.toDF().rdd.zipWithIndex().map(x => {
      val rowData = x._1.toSeq.toArray ++ Array(x._2.toInt)
      new GenericRowWithSchema(rowData, schema).asInstanceOf[Row]
    }
    ), schema)
    val res = df.join(vectorSequencesDs).where(df("index") === vectorSequencesDs("index")).drop("index")
    res.printSchema()
    res
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema.add(StructField($(outputCol), new ArrayType(VectorType, false), nullable = false))
}