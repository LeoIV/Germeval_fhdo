package de.hoevelmann.bachelorthesis.modelling.entities

/**
  * Created by hoevelmann on 07.06.2017.
  */
@SerialVersionUID(154353442378L)
class FastTextVector(val word: String, val vector: Array[Double]) extends Serializable {

  def +(that: FastTextVector) = new FastTextVector("UNDEFINED", vector.zip(that.vector).map(x => x._1 + x._2))

  def -(that: FastTextVector) = new FastTextVector("UNDEFINED", vector.zip(that.vector).map(x => x._1 - x._2))

  def distance(that: FastTextVector): Double = math.sqrt(this.vector.par.zip(that.vector.par).map(x => math.pow(x._1 - x._2, 2)).sum)

}
