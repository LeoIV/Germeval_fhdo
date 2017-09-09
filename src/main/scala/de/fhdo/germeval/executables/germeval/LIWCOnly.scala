package de.fhdo.germeval.executables.germeval

import de.fhdo.germeval.modelling.entities.ConfusionMatrix
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.evaluation.MultilabelMetrics
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.io.Source
import scala.util.Random

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 18.07.2017.
  */
object LIWCOnly {
  def execute(spark: SparkSession = SparkSession.builder().master("local[*]").getOrCreate()) {
    val trainingReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_train.csv")
    val testReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_dev.csv")

    import spark.implicits._
    spark.sparkContext.setLogLevel("ERROR")

    println("creating dataframes")

    val sentimentTrainingDataFrame: DataFrame = (for (review <- Random.shuffle(trainingReviews)) yield {
      (review.sentiClass, Vectors.dense(review.liwcFeatures))
    }).toDF("sentiment", "liwcFeatures")
    val sentimentTestDataFrame: DataFrame = (for (review <- Random.shuffle(testReviews)) yield {
      (review.sentiClass, Vectors.dense(review.liwcFeatures))
    }).toDF("sentiment", "liwcFeatures")
    val relevanceTrainingDataFrame: DataFrame = (for (review <- Random.shuffle(trainingReviews)) yield {
      (if (review.relevant) 1 else 0, Vectors.dense(review.liwcFeatures))
    }).toDF("relevanceLabel", "liwcFeatures")
    val relevanceTestDataFrame: DataFrame = (for (review <- Random.shuffle(testReviews)) yield {
      (if (review.relevant) 1 else 0, Vectors.dense(review.liwcFeatures))
    }).toDF("relevanceLabel", "liwcFeatures")

    println("creating preprocessors")

    /*
     * define the pipeline steps
     */


    def sentimentLayers: Array[Int] = Array(93) ++ Array(50) ++ Array(3)

    def relevanceLayers: Array[Int] = Array(93) ++ Array(50) ++ Array(2)

    val sentimentMLPTrainer = new MultilayerPerceptronClassifier().setLabelCol("sentiment").setFeaturesCol("liwcFeatures").setLayers(sentimentLayers).setBlockSize(128).setSeed((Math.random() * 100000).toLong).setMaxIter(100)

    val relevanceMLPTrainer = new MultilayerPerceptronClassifier().setLabelCol("relevanceLabel").setFeaturesCol("liwcFeatures").setLayers(relevanceLayers).setBlockSize(128).setSeed((Math.random() * 100000).toLong).setMaxIter(100)

    println("fitting models")

    val sentimentPipelineMLP = new Pipeline().setStages(Array(sentimentMLPTrainer))
    val relevancePipelineMLP = new Pipeline().setStages(Array(relevanceMLPTrainer))

    val sentimentModel = sentimentPipelineMLP.fit(sentimentTrainingDataFrame)
    println("fitted sentiment models")
    val relevanceModel = relevancePipelineMLP.fit(relevanceTrainingDataFrame)
    println("fitted relevance models")

    val sentimentValidationResult = sentimentModel.transform(sentimentTestDataFrame)
    val relevanceValidationResult = relevanceModel.transform(relevanceTestDataFrame)

    val sentimentEvaluator = new MultilabelMetrics(sentimentValidationResult.toDF().rdd.map(row => (Array(row.getAs[Double]("prediction")), Array(row.getAs[Int]("sentiment").toDouble))))
    val relevanceEvaluator = new MultilabelMetrics(relevanceValidationResult.toDF().rdd.map(row => (Array(row.getAs[Double]("prediction")), Array(row.getAs[Int]("relevanceLabel").toDouble))))

    println("(sentiment) f1 multilayer perceptron: " + sentimentEvaluator.microF1Measure)
    println("(relevance) f1 multilayer perceptron: " + relevanceEvaluator.microF1Measure)

    val cfsMtrxSentiment = new ConfusionMatrix(3, sentimentValidationResult.rdd.collect().map(row => Tuple2(Math.round(row.getAs[Double]("prediction")).toInt, row.getAs[Int]("sentiment"))))
    val cfsMtrxRelevance = new ConfusionMatrix(2, relevanceValidationResult.rdd.collect().map(row => Tuple2(Math.round(row.getAs[Double]("prediction")).toInt, row.getAs[Int]("relevanceLabel"))))

    print("confusion matrix sentiment: \n\n" + cfsMtrxSentiment.toString + "\n")
    print("confusion matrix relevance: \n\n" + cfsMtrxRelevance.toString + "\n")

  }

  /*
   * ============= Auxiliary stuff ============
   */

  /*
   * Auxiliary objects
   */


  /*
   * Auxiliary methods
   */


  def loadReviews(filename: String): Seq[Review] = {

    val result = Source.fromFile(filename, "utf-8")
      .getLines().toSeq.tail.par
      .map(_.split("\t"))
      .map(x =>
        new Review(
          relevant = x(2) == "true",
          sentiClass = if (x(3) == "negative") 0 else if (x(3) == "neutral") 1 else 2,
          liwcFeatures = x.slice(5, x.length).map(_.replace(",", ".")).map(_.toDouble)))
      .filter(_.liwcFeatures.length == 93)
      .seq
    result
  }

  /*
   * Auxiliary classes
   */

  class Review(val relevant: Boolean, val sentiClass: Int, val liwcFeatures: Array[Double])

  class TermWithClass(val term: String, val sentimentClass: Int, val occurrences: Int)

}
