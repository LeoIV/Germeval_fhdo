package de.hoevelmann.bachelorthesis.modelling.transformers

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types._

/**
  * Expects a list of double as features. The list should be the labels.
  *
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 10.08.2017.
  */
class VotingClassifier(override val uid: String) extends Predictor[org.apache.spark.ml.linalg.Vector, VotingClassifier, VotingClassificationModel] with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("VotingClassifier"))

  override protected def train(dataset: Dataset[_]): VotingClassificationModel = new VotingClassificationModel()

  override def copy(extra: ParamMap): VotingClassifier = defaultCopy(extra)

  override def featuresDataType: DataType = org.apache.spark.ml.linalg.SQLDataTypes.VectorType

}

class VotingClassificationModel(override val uid: String) extends PredictionModel[org.apache.spark.ml.linalg.Vector, VotingClassificationModel] {

  def this() = this(Identifiable.randomUID("VotingClassificationModel"))

  override protected def predict(features: org.apache.spark.ml.linalg.Vector): Double = features.toArray.map(math.round(_).toInt).groupBy(g => g).map(g => (g._1, g._2.length)).maxBy(_._2)._1.toDouble

  override def copy(extra: ParamMap): VotingClassificationModel = defaultCopy(extra)

  override def featuresDataType: DataType = org.apache.spark.ml.linalg.SQLDataTypes.VectorType

}
