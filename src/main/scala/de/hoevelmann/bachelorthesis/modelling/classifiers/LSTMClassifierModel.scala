package de.hoevelmann.bachelorthesis.modelling.classifiers

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg
import org.apache.spark.rdd.RDD

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 28.06.2017.
  */
class LSTMClassifierModel(
                           val numClasses: Int,
                           val sequenceLength: Int,
                           val vectorDimensionality: Int,
                           val module: Module[Float]) {


  def predictClass(features: Seq[linalg.Vector]): Seq[Int] = {
    println("predicting classes")
    val samples = for (feature: linalg.Vector <- features) yield Sample(
      featureTensor = Tensor(feature.toArray.map(_.toFloat), Array(sequenceLength, vectorDimensionality)).contiguous(),
      labelTensor = Tensor(Array(-1.0f), Array(1))
    )
    val samplesRdd: RDD[Sample[Float]] = SparkContext.getOrCreate().parallelize(samples)


    module.predictClass(samplesRdd).collect()
  }
}