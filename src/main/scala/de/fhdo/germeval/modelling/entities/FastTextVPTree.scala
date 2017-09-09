package de.fhdo.germeval.modelling.entities

import com.eatthepath.jvptree.{DistanceFunction, VPTree}

import scala.collection.JavaConverters._
import scala.collection.immutable.HashMap
import scala.util.Try

/**
  * Created by hoevelmann on 07.06.2017.
  */
class FastTextVPTree(vectors: HashMap[String, FastTextVector]) {

  /**
    * Find the vector for a word in our fastText model
    *
    * @param word the word we search for
    * @return vector option, may be empty if the word doesn't exist in our fasttext model
    */
  def findByWord(word: String): Option[FastTextVector] = {
    // returning "None" if not existent
    Try(vectors(word)).toOption
  }

  /**
    * Find the nearest neighbours for a given vector
    *
    * @param vector                 the vector, for which we search the nearest neighbors
    * @param numberNearestNeighbors the desired number of nearest neighbors
    * @return sequence containing the nearest neighbors
    */
  def findByVector(vector: FastTextVector, numberNearestNeighbors: Int = 1): Seq[FastTextVector] = {
    tree.getNearestNeighbors(vector, numberNearestNeighbors).asScala
  }

  /**
    * Search for vectors in the fastTextModel, that are within a certain distance and that occur in the sourceWordSet. Takes very long time.
    *
    * @param vectorOption    option for the vector we start the search with. if the option is empty, this method returns 'None'
    * @param searchCriterion the criterion, that limits the search
    * @param maxDistance     the max distance, a vector may be in
    * @return an option for the closes vector. may be empty, if there is no partner for this vector
    */
  def findClosestByCriterion(vectorOption: Option[FastTextVector], searchCriterion: FastTextVector => Boolean, maxDistance: Double): Option[FastTextVector] = vectorOption match {
    case Some(searchVector) =>
      val vectorsWithinDistance = tree.getAllWithinDistance(searchVector, maxDistance).asScala.filter(searchCriterion)
      if (vectorsWithinDistance.isEmpty) None
      else Some(vectorsWithinDistance.minBy(distanceFunction.getDistance(searchVector, _)))
    case None => None
  }

  /*
   * Private attributes
   */

  private val distanceFunction = new FastTextDistanceFunction()


  private val tree = new VPTree[FastTextVector](new FastTextDistanceFunction(), vectors.values.asJavaCollection)

  /*
   * Auxiliary classes
   */

  class FastTextDistanceFunction extends DistanceFunction[FastTextVector] {
    override def getDistance(firstPoint: FastTextVector, secondPoint: FastTextVector): Double = math.sqrt(firstPoint.vector.par.zip(secondPoint.vector.par).map(x => math.pow(x._1 - x._2, 2)).sum)
  }

}
