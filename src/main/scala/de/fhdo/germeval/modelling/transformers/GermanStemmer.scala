package de.fhdo.germeval.modelling.transformers

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{ArrayType, DataType, FloatType, StringType}

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 09.06.2017.
  */
class GermanStemmer(override val uid: String) extends UnaryTransformer[Seq[String], Seq[String], GermanStemmer] {

  def this() = this(Identifiable.randomUID("germanStemmer"))

  override protected def createTransformFunc: (Seq[String]) => Seq[String] = {
    _.iterator.map(GermanStemmer.stem).toSeq
  }

  override protected def outputDataType: DataType = new ArrayType(StringType, false)
}

object GermanStemmer {
  val germanStemmer = new snowball.GermanStemmer()

  private def stem(word: String): String = {
    this.synchronized {
      germanStemmer.setCurrent(word)
      germanStemmer.stem()
      germanStemmer.getCurrent
    }
  }
}