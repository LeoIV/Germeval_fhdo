package de.fhdo.germeval.modelling.transformers

import de.fhdo.germeval.modelling.entities.{FastTextVPTree, FastTextVector}
import de.hoevelmann.bachelorthesis.modelling.entities.{FastTextVPTree, FastTextVector}
import org.apache.spark.ml.feature.Word2VecModel
import org.apache.spark.ml.linalg.SQLDataTypes._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{UnaryTransformer, linalg}
import org.apache.spark.sql.types.{ArrayType, DataType}

import scala.collection.immutable.HashMap
import scala.util.Try

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 28.06.2017.
  */
class NonAveragedWord2Vec(override val uid: String) extends UnaryTransformer[Seq[String], Seq[linalg.Vector], NonAveragedWord2Vec] {

  def this() = this(Identifiable.randomUID("nonaveragedword2vec"))

  val word2VecModel: Param[FastTextVPTree] = new Param[FastTextVPTree](this, "word2VecModel", "a fitted word2vec model")

  def setWord2VecModel(value: Word2VecModel): this.type = {
    val map: Seq[(String, FastTextVector)] = value.getVectors.rdd.map(f => Seq[(String, FastTextVector)]((f.getAs[String]("word"), new FastTextVector(f.getAs[String]("word"), f.getAs[linalg.Vector]("vector").toArray)))).reduce((a, b) => a ++ b)
    set(word2VecModel, new FastTextVPTree(HashMap[String, FastTextVector](map: _*)))
  }


  override protected def createTransformFunc: (Seq[String]) => Seq[linalg.Vector] = { x =>
    println("creating word2vecs")
    x.par.map(word => Try($(word2VecModel).findByWord(word).get)).filter(_.isSuccess).map(_.get).map(v => Vectors.dense(v.vector)).seq
  }

  override protected def outputDataType: DataType = new ArrayType(VectorType, false)

}
