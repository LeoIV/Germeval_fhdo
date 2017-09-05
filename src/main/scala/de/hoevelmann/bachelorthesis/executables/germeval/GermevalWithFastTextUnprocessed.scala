package de.hoevelmann.bachelorthesis.executables.germeval

import de.hoevelmann.bachelorthesis.modelling.transformers.{FastTextClassifier, GermanStemmer, SdCrossValidator}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.io.Source
import scala.util.Random

object GermevalWithFastTextUnprocessed {

  /**
    * Execute the Germeval task
    *
    * @param spark              the SparkSession
    * @param ngramSize          size of the n-grams (default: 1)
    * @param numFeatures        number of features, that will be selected (default: 1000), set to zero if you want to disable the feature selection
    * @param intermediateLayers the hidden layers of the multilayer perceptron (default: Array(300))
    * @param useLIWCFeatures    should we use the liwc-features (default: true)
    * @param stem               should we stem the words (default: true)
    * @param numIterations      number of iterations (default: 100)
    */
  def execute(spark: SparkSession = SparkSession.builder().master("local[*]").getOrCreate(),
              ngramSize: Int = 1,
              numFeatures: Int = 1000,
              intermediateLayers: Array[Int] = Array(300),
              useLIWCFeatures: Boolean = true,
              stem: Boolean = true,
              numIterations: Int = 100,
              maxDistance: Double = 1.0): Unit = {
    import spark.implicits._
    val trainingReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_train.csv")
    val testReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_dev.csv")

    spark.sparkContext.setLogLevel("ERROR")

    println("creating dataframes")

    val sentimentTrainingDataFrame: DataFrame = (for (review <- Random.shuffle(trainingReviews ++ testReviews)) yield {
      (review.sentiClass, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("sentiment", "input", "liwcFeatures")
    // val sentimentTestDataFrame: DataFrame = (for (review <- Random.shuffle(testReviews)) yield {
    //   (review.sentiClass, review.terms, Vectors.dense(review.liwcFeatures))
    // }).toDF("sentiment", "input", "liwcFeatures")
    val relevanceTrainingDataFrame: DataFrame = (for (review <- Random.shuffle(trainingReviews ++ testReviews)) yield {
      (if (review.relevant) 1 else 0, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("relevanceLabel", "input", "liwcFeatures")
    //val relevanceTestDataFrame: DataFrame = (for (review <- Random.shuffle(testReviews)) yield {
    //  (if (review.relevant) 1 else 0, review.terms, Vectors.dense(review.liwcFeatures))
    //}).toDF("relevanceLabel", "input", "liwcFeatures")

    println("creating preprocessors")

    /*
     * define the pipeline steps
     */

    val germanStemmer: GermanStemmer = new GermanStemmer().setInputCol("input").setOutputCol("stems")

    /*
     * mlp model definition
     */

    val sentimentFastTextTrainer = new FastTextClassifier().setLabelCol("sentiment").setFeaturesCol("stems")
    val relevanceFastTextTrainer = new FastTextClassifier().setLabelCol("relevanceLabel").setFeaturesCol("stems")
    //val relevanceMLPTrainer = new MultilayerPerceptronClassifier().setLabelCol("relevanceLabel").setFeaturesCol("topFeatures").setLayers(relevanceLayers).setSeed(42L).setMaxIter(90)

    println("fitting models")

    val sentimentPipeline = new Pipeline().setStages(Array(germanStemmer, sentimentFastTextTrainer))
    val relevancePipeline = new Pipeline().setStages(Array(germanStemmer, relevanceFastTextTrainer))

    val paramGrid = new ParamGridBuilder()
      .build()


    val sentimentCV = new SdCrossValidator()
      .setEstimator(sentimentPipeline)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("sentiment").setMetricName("f1"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val relevanceCV = new SdCrossValidator()
      .setEstimator(relevancePipeline)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("relevanceLabel").setMetricName("f1"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)


    val sentimentModel = sentimentCV.fitSd(sentimentTrainingDataFrame)
    println("fitted sentiment models")
    val relevanceModel = relevanceCV.fitSd(relevanceTrainingDataFrame)
    println("fitted relevance models")

    println("sentiment fasttext: " + sentimentModel.allMetrics.toList)
    println("\taverage: " + sentimentModel.average)
    println("relevance fasttext : " + relevanceModel.allMetrics.toList)
    println("\taverage: " + relevanceModel.average)

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
          terms = x(1).toLowerCase().split(" "),
          relevant = x(2) == "true",
          sentiClass = if (x(3) == "negative") 0 else if (x(3) == "neutral") 1 else 2,
          liwcFeatures = x.slice(5, x.length).map(_.replace(",", ".")).map(_.toDouble)))
      .filter(_.liwcFeatures.length == 93)
      .seq
    result
  }

  private def uuid() = java.util.UUID.randomUUID.toString


  /*
   * Auxiliary classes
   */

  class Review(val terms: Seq[String], val relevant: Boolean, val sentiClass: Int, val liwcFeatures: Array[Double])

  class TermWithClass(val term: String, val sentimentClass: Int, val occurrences: Int)

}