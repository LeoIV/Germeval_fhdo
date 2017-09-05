package de.hoevelmann.bachelorthesis.modelling.entities

import com.eatthepath.jvptree.DistanceFunction


/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 12.06.2017.
  */
class FastTextDistanceFunction extends DistanceFunction[FastTextVector] {
  override def getDistance(firstPoint: FastTextVector, secondPoint: FastTextVector): Double = math.sqrt(firstPoint.vector.par.zip(secondPoint.vector.par).map(x => math.pow(x._1 - x._2, 2.0)).sum)
}