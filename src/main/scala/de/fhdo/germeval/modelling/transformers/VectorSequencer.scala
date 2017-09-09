package de.fhdo.germeval.modelling.transformers

import org.apache.spark.ml.linalg.{SQLDataTypes, Vectors}
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{UnaryTransformer, linalg}
import org.apache.spark.sql.types.{ArrayType, DataType, DoubleType, FloatType}

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 28.06.2017.
  */
class VectorSequencer(override val uid: String) extends UnaryTransformer[Seq[linalg.Vector], linalg.Vector, VectorSequencer] {

  def this() = this(Identifiable.randomUID("vectorsequencer"))

  val sequenceLength: IntParam = new IntParam(this, "sequenceLength", "the sequence length", (x: Int) => x > 0)

  val embeddingSize: IntParam = new IntParam(this, "embeddingSize", "the embedding size", (x: Int) => x > 0)

  def setSequenceLength(value: Int): this.type = set(sequenceLength, value)

  def getSequenceLength: Int = $(sequenceLength)

  def setEmbeddingSize(value: Int): this.type = set(embeddingSize, value)

  def getEmbeddingSize: Int = $(embeddingSize)

  override protected def createTransformFunc: (Seq[linalg.Vector]) => linalg.Vector = { vectors =>
    println("sequencing")
    val numCols = vectors.length
    val filledArrays: Seq[Array[Float]] = vectors.take($(sequenceLength)).map(_.toArray.map(_.toFloat))
    if ($(sequenceLength) > numCols)
      Vectors.sparse($(sequenceLength) * $(embeddingSize), filledArrays.flatten.map(_.toDouble).zipWithIndex.map(_.swap))
    else Vectors.dense(filledArrays.flatten.map(_.toDouble).toArray)
  }

  override protected def outputDataType: DataType = SQLDataTypes.VectorType
}
