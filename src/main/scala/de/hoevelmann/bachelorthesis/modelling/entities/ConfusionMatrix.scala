package de.hoevelmann.bachelorthesis.modelling.entities

import java.io.Serializable

/**
  * Created by hoevelmann on 06.06.2017.
  *
  * @param numClasses number of classes
  * @param values     (actualClass, prediction)
  *
  */
@SerialVersionUID(1937548L)
class ConfusionMatrix(val numClasses: Int, val values: Seq[(Int, Int)]) extends Serializable {


  override def toString: String = {
    val builder = StringBuilder.newBuilder
    builder.append("\t\t")
    for (actualClass <- 0 until numClasses) {
      builder.append(actualClass.toString + "\t")
    }
    builder.append("\n")
    for (i <- 0 to (numClasses + 2) * 8)
      builder.append("-")
    builder.append("\n")
    for (actualClass <- 0 until numClasses) {
      builder.append(actualClass.toString + "\t|\t")
      for (predictedClass <- 0 until numClasses) {
        builder.append(values.count(x => x._1 == actualClass && x._2 == predictedClass).toString + "\t")

      }
      builder.append("\n")
    }
    val matrix = builder.mkString

    "left: actual class, top: predicted\n" + matrix
  }
}