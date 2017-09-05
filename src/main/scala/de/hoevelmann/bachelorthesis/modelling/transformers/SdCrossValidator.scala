package de.hoevelmann.bachelorthesis.modelling.transformers

import com.github.fommil.netlib.F2jBLAS
import org.apache.spark.ml.Model
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.ml.util.{Identifiable, Instrumentation}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 07.08.2017.
  */
class SdCrossValidator extends CrossValidator {

  private val f2jBLAS = new F2jBLAS

  def fitSd(dataset: Dataset[_]): SdCrossValidatorModel = {
    val schema = dataset.schema
    transformSchema(schema, logging = true)
    val sparkSession = dataset.sparkSession
    val est = $(estimator)
    val eval = $(evaluator)
    val epm = $(estimatorParamMaps)
    val numModels = epm.length
    val metrics = new Array[Double](epm.length)
    val allMetrics = new Array[Double]($(numFolds))

    val splits = MLUtils.kFold(dataset.toDF.rdd, $(numFolds), $(seed))
    splits.zipWithIndex.foreach { case ((training, validation), splitIndex) =>
      val trainingDataset = sparkSession.createDataFrame(training, schema).cache()
      val validationDataset = sparkSession.createDataFrame(validation, schema).cache()
      // multi-model training
      logDebug(s"Train split $splitIndex with multiple sets of parameters.")
      val models = est.fit(trainingDataset, epm).asInstanceOf[Seq[Model[_]]]
      trainingDataset.unpersist()
      var i = 0
      while (i < numModels) {
        // TODO: duplicate evaluator to take extra params from input
        val metric = eval.evaluate(models(i).transform(validationDataset, epm(i)))
        logDebug(s"Got metric $metric for model trained with ${epm(i)}.")
        metrics(i) += metric
        allMetrics(splitIndex) = metric
        i += 1
      }
      validationDataset.unpersist()
    }
    f2jBLAS.dscal(numModels, 1.0 / $(numFolds), metrics, 1)
    copyValues(new SdCrossValidatorModel(allMetrics.sum / allMetrics.length.toDouble, allMetrics))
  }

  def fitSdWAugmentation(trainingDataset: Dataset[_], augmentationDataset: Dataset[_]): SdCrossValidatorModel = {
    val schema = trainingDataset.schema
    transformSchema(schema, logging = true)
    val sparkSession = trainingDataset.sparkSession
    val est = $(estimator)
    val eval = $(evaluator)
    val epm = $(estimatorParamMaps)
    val numModels = epm.length
    val metrics = new Array[Double](epm.length)
    val allMetrics = new Array[Double]($(numFolds))

    val splits = MLUtils.kFold(trainingDataset.toDF.rdd, $(numFolds), $(seed))

    splits.zipWithIndex.foreach { case ((training, validation), splitIndex) =>
      val trainingDataset = sparkSession.createDataFrame(training, schema).union(augmentationDataset.toDF()).cache()
      val validationDataset = sparkSession.createDataFrame(validation, schema).cache()
      // multi-model training
      logDebug(s"Train split $splitIndex with multiple sets of parameters.")
      val models = est.fit(trainingDataset, epm).asInstanceOf[Seq[Model[_]]]
      trainingDataset.unpersist()
      var i = 0
      while (i < numModels) {
        // TODO: duplicate evaluator to take extra params from input
        val metric = eval.evaluate(models(i).transform(validationDataset, epm(i)))
        logDebug(s"Got metric $metric for model trained with ${epm(i)}.")
        metrics(i) += metric
        allMetrics(splitIndex) = metric
        i += 1
      }
      validationDataset.unpersist()
    }
    f2jBLAS.dscal(numModels, 1.0 / $(numFolds), metrics, 1)
    copyValues(new SdCrossValidatorModel(allMetrics.sum / allMetrics.length.toDouble, allMetrics))
  }


  def fitSd(trainingDataset: Dataset[_], validationDataset: Dataset[_]): SdCrossValidatorModel = {
    val schema = trainingDataset.schema
    transformSchema(schema, logging = true)
    val sparkSession = trainingDataset.sparkSession
    val est = $(estimator)
    val eval = $(evaluator)
    val epm = $(estimatorParamMaps)
    val numModels = epm.length
    val metrics = new Array[Double](epm.length)
    val allMetrics = new Array[Double]($(numFolds))

    val trainingSplits = MLUtils.kFold(trainingDataset.toDF.rdd, $(numFolds), $(seed))
    val validationSplits = MLUtils.kFold(validationDataset.toDF.rdd, $(numFolds), $(seed))

    def splits = trainingSplits.zip(validationSplits).map(trainAndVal => (trainAndVal._1._1, trainAndVal._2._2))

    splits.zipWithIndex.foreach { case ((training, validation), splitIndex) =>
      val trainingDataset = sparkSession.createDataFrame(training, schema).cache()
      val validationDataset = sparkSession.createDataFrame(validation, schema).cache()
      // multi-model training
      logDebug(s"Train split $splitIndex with multiple sets of parameters.")
      val models = est.fit(trainingDataset, epm).asInstanceOf[Seq[Model[_]]]
      trainingDataset.unpersist()
      var i = 0
      while (i < numModels) {
        // TODO: duplicate evaluator to take extra params from input
        val metric = eval.evaluate(models(i).transform(validationDataset, epm(i)))
        logDebug(s"Got metric $metric for model trained with ${epm(i)}.")
        metrics(i) += metric
        allMetrics(splitIndex) = metric
        i += 1
      }
      validationDataset.unpersist()
    }
    f2jBLAS.dscal(numModels, 1.0 / $(numFolds), metrics, 1)
    copyValues(new SdCrossValidatorModel(allMetrics.sum / allMetrics.length.toDouble, allMetrics))
  }

}

class SdCrossValidatorModel(val average: Double, val allMetrics: Array[Double], override val uid: String) extends Model[SdCrossValidatorModel] {

  def this(ave: Double, all: Array[Double]) = this(ave, all, Identifiable.randomUID("SdCrossValidatorModel"))

  override def toString() = "average: " + average + "\n\tallmetrics: " + allMetrics.mkString("[", " | ", "]")

  override def copy(extra: ParamMap): SdCrossValidatorModel = defaultCopy(extra)

  override def transform(dataset: Dataset[_]): DataFrame = dataset.toDF()

  override def transformSchema(schema: StructType): StructType = schema
}