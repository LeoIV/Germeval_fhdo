package de.hoevelmann.bachelorthesis.modelling.transformers

import de.hoevelmann.bachelorthesis.modelling.entities.{FastTextVPTree, FastTextVector}
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.{DoubleParam, Param}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}

import scala.collection.immutable.HashMap
import scala.collection.mutable
import scala.util.Try

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 09.06.2017.
  */
class WordSubstitutor(override val uid: String) extends UnaryTransformer[Seq[String], Seq[String], WordSubstitutor] with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("wordsubstitute"))

  val treeBuildingVectors: Param[Seq[FastTextVector]] = new Param(this, "tree building vectors", "the word vectors to build the tree from", (x: Seq[FastTextVector]) => x.nonEmpty)

  val wordVectorVocabulary: Param[HashMap[String, FastTextVector]] = new Param(this, "word vector vocabulary", "the vocabulary of word vectors", (x: HashMap[String, FastTextVector]) => x.nonEmpty)

  val maxDistance: DoubleParam = new DoubleParam(this, "maxDistance", "the maximum allowed distance", (x: Double) => x > 0.0)

  val wordSubstituteMap: Param[mutable.HashMap[String, Option[String]]] = new Param(this, "word substitute map", "the word substitute map")

  private lazy val fastTextVPTree = new FastTextVPTree(HashMap[String, FastTextVector]($(treeBuildingVectors).map(x => (x.word, x)): _*))

  setDefault(wordSubstituteMap, new mutable.HashMap[String, Option[String]]())

  def setWordSubstituteMap(value: mutable.HashMap[String, Option[String]]): this.type = set(wordSubstituteMap, value)

  def getWordSubstituteMap: mutable.HashMap[String, Option[String]] = $(wordSubstituteMap)

  def setWordVectorVocabulary(value: HashMap[String, FastTextVector]): this.type = set(wordVectorVocabulary, value)

  def getWordVectorVocabulary: HashMap[String, FastTextVector] = $(wordVectorVocabulary)

  def setMaxDistance(value: Double): this.type = set(maxDistance, value)

  def getMaxDistance: Double = $(maxDistance)

  def setTreeBuildingVectors(value: Seq[FastTextVector]): this.type = set(treeBuildingVectors, value)

  def getTreeBuildingVectors: Seq[FastTextVector] = $(treeBuildingVectors)

  override protected def createTransformFunc: (Seq[String]) => Seq[String] = _.iterator.toSeq.par.map(word => WordSubstitutor.wordSubstitute(word, $(wordVectorVocabulary), $(maxDistance), fastTextVPTree, $(wordSubstituteMap))).seq

  override protected def outputDataType: DataType = new ArrayType(StringType, false)
}

private object WordSubstitutor {


  private def wordSubstitute(word: String, wordVectorVocabulary: HashMap[String, FastTextVector], maxDistance: Double, fastTextVPTree: FastTextVPTree, wordSubstituteMap: mutable.HashMap[String, Option[String]]): String = {

    this.synchronized {
      val entryFromMap = Try(wordSubstituteMap(word))
      if (entryFromMap.isFailure) {
        val vectorForSearchTerm = Try(wordVectorVocabulary(word))
        if (vectorForSearchTerm.isFailure) {
          wordSubstituteMap.put(word, None)
        }
        else {
          val nearestNeighbor = fastTextVPTree.findByVector(wordVectorVocabulary(word), 1).head
          if (nearestNeighbor.distance(vectorForSearchTerm.get) < maxDistance) {
            print("!")
            wordSubstituteMap.put(word, Some(nearestNeighbor.word))
          }
          else wordSubstituteMap.put(word, None)
        }
      }
      wordSubstituteMap(word).getOrElse(word)
    }
  }

}