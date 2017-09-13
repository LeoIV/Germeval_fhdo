package de.fhdo.germeval.modelling.transformers

import de.fhdo.germeval.modelling.fasttext.FastText
import org.apache.spark.ml.linalg.SQLDataTypes._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{UnaryTransformer, linalg}
import org.apache.spark.sql.types.DataType

import scala.collection.immutable.HashMap
import scala.io.Source

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 16.06.2017.
  */
class FastTextSentenceVector(override val uid: String) extends UnaryTransformer[Seq[String], linalg.Vector, FastTextSentenceVector] {

  def this() = this(Identifiable.randomUID("fasttextsentencevectors"))

  override protected def createTransformFunc: Seq[String] => linalg.Vector = { sentence => FastTextSentenceVector.sentenceToVector(sentence).head }

  override protected def outputDataType: DataType = VectorType
}

private object FastTextSentenceVector {

  //TODO make this dynamic

  private val wordVectors = HashMap[String, Array[Double]](Source.fromFile("germeval/corpus/wholeModel.vec")
    .getLines()
    .toSeq
    .map(_.split(" "))
    .map(f => (f.head, f.tail.map(_.toDouble))): _*)


  private def sentenceToVector(sentence: Seq[String]) = FastText.sentenceToVectorFromHashMap(Seq(sentence), wordVectors)
}
