package de.fhdo.germeval.modelling.classifiers

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{Adagrad, Optimizer, Top1Accuracy, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.ml.linalg
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 27.06.2017.
  */
class LSTMClassifier(val vectorDimensionality: Int, val sequenceLength: Int, val numClasses: Int) extends Serializable {


  private val spark: SparkSession = SparkSession.builder().getOrCreate()


  def train(trainingData: Seq[(Double, linalg.Vector)], validationData: Seq[(Double, linalg.Vector)]): LSTMClassifierModel = {


    val model = Sequential[Float]()
      .add(Recurrent[Float]()
        .add(LSTM(vectorDimensionality, 128)))
      .add(Select(2, -1))
      .add(Linear(128, 100))
      .add(Linear(100, numClasses))
      .add(LogSoftMax())

    println("jabadabadu")

    val trainingRdd: RDD[Sample[Float]] = spark.sparkContext.parallelize(trainingData).map(x => {
      val smp = Sample(
        featureTensor = Tensor(x._2.toArray.map(_.toFloat), Array(sequenceLength, vectorDimensionality)).contiguous(),
        labelTensor = Tensor(Array(x._1.toFloat), Array(1))
      )
      smp
    })

    val validationRdd: RDD[Sample[Float]] = spark.sparkContext.parallelize(validationData).map(x => {
      val smp = Sample(
        featureTensor = Tensor(x._2.toArray.map(_.toFloat), Array(sequenceLength, vectorDimensionality)).contiguous(),
        labelTensor = Tensor(Array(x._1.toFloat), Array(1))
      )
      smp
    })

    //val criterion = TimeDistributedCriterion(ClassNLLCriterion[Float]())
    println("jojo")

    trainingRdd.foreach(sample => "[" + sample.feature().storage().toString() + "]\t " + sample.label().storage().toString())
    validationRdd.foreach(sample => "[" + sample.feature().storage().toString() + "]\t " + sample.label().storage().toString())

    val optimizer = Optimizer(
      model = model,
      sampleRDD = trainingRdd,
      criterion = ClassNLLCriterion[Float](),
      batchSize = 36)
    println("tralaal")
    val module: Module[Float] = optimizer
      .setOptimMethod(new Adagrad())
      .setEndWhen(Trigger.maxEpoch(20))
      .setValidation(Trigger.everyEpoch, validationRdd, Array(new Top1Accuracy[Float]), 36)
      .optimize()
    println("jipi")
    new LSTMClassifierModel(numClasses, sequenceLength, vectorDimensionality, module)
  }
}


