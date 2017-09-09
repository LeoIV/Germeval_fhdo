package de.fhdo.germeval.modelling.transformers

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 15.08.2017.
  */
class EnglishStemmer(override val uid: String) extends UnaryTransformer[Seq[String], Seq[String], EnglishStemmer] {

  def this() = this(Identifiable.randomUID("EnglishStemmer"))

  override protected def createTransformFunc: (Seq[String]) => Seq[String] = {
    _.iterator.map(EnglishStemmer.stem).toSeq
  }

  override protected def outputDataType: DataType = new ArrayType(StringType, false)
}

object EnglishStemmer {
  val englishStemmer = new snowball.EnglishStemmer()

  private def stem(word: String): String = {
    this.synchronized {
      englishStemmer.setCurrent(word)
      englishStemmer.stem()
      englishStemmer.getCurrent
    }
  }
}