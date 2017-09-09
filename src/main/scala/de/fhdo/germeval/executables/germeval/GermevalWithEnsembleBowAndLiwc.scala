package de.hoevelmann.bachelorthesis.executables.germeval

/*
To run the ensemble models, a "probability col" has to be added to the Spark MLP. This will come in future releases of Spark.
A custom version of Spark has been built to run these models, that isn't part of this repository. Since this class produces
build errors, it is commented out. It should run with a newer version of Spark.
 */

import de.hoevelmann.bachelorthesis.modelling.transformers.GermanStemmer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.evaluation.MultilabelMetrics
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.{Logger, LoggerFactory}

import scala.io.Source
import scala.util.Random
/*
object GermevalWithEnsembleBowAndLiwc {

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
    val trainingReviews: Seq[Review] = loadReviews("liwc_train.csv")
    val testReviews: Seq[Review] = loadReviews("liwc_dev.csv")

    println("creating dataframes")

    val sentimentTrainingDataFrame: DataFrame = (for (review <- Random.shuffle(trainingReviews)) yield {
      (review.sentiClass, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("sentiment", "input", "liwcFeatures")
    val sentimentTestDataFrame: DataFrame = (for (review <- Random.shuffle(testReviews)) yield {
      (review.sentiClass, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("sentiment", "input", "liwcFeatures")
    val relevanceTrainingDataFrame: DataFrame = (for (review <- Random.shuffle(trainingReviews)) yield {
      (if (review.relevant) 1 else 0, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("relevanceLabel", "input", "liwcFeatures")
    val relevanceTestDataFrame: DataFrame = (for (review <- Random.shuffle(testReviews)) yield {
      (if (review.relevant) 1 else 0, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("relevanceLabel", "input", "liwcFeatures")

    println("creating preprocessors")

    /*
     * define the pipeline steps
     */

    val languageTool: LanguageToolTransformer = new LanguageToolTransformer().setInputCol("input").setOutputCol("languageToolFeatures")
    val germanStemmer: GermanStemmer = new GermanStemmer().setInputCol("input").setOutputCol("stems")
    //val sentimentLexiconStep: Option[SentimentLexicon] = if (sentimentLexicon.nonEmpty) Some(new SentimentLexicon().setSentimentLexicon(sentimentLexicon).setInputCol("input").setOutputCol("output")) else None

    val hashingTermFrequencies: HashingTF = new HashingTF().setInputCol("stems").setOutputCol("tfs").setNumFeatures(16384)
    val idf: IDF = new IDF().setInputCol("tfs").setOutputCol("idfs")

    val sentimentChiSqSelector: ChiSqSelector = new ChiSqSelector().setFeaturesCol("idfs").setLabelCol("sentiment").setOutputCol("topFeatures").setNumTopFeatures(1000)
    val relevanceChiSqSelector: ChiSqSelector = new ChiSqSelector().setFeaturesCol("idfs").setLabelCol("relevanceLabel").setOutputCol("topFeatures").setNumTopFeatures(1000)


    /*
     * mlp model definition
     */

    def sentimentBowLayers: Array[Int] = Array(1000) ++ intermediateLayers ++ Array(3)

    def sentimentLiwcFeatures: Array[Int] = Array(84, 50, 3)

    def relevanceBowLayers: Array[Int] = Array(1000) ++ intermediateLayers ++ Array(2)

    def relevanceLiwcFeatures: Array[Int] = Array(84, 50, 2)

    val sentimentMLPBowTrainer = new MultilayerPerceptronClassifier().setPredictionCol("afjdslk").setProbabilityCol("t54").setLabelCol("sentiment").setFeaturesCol("topFeatures")
      .setLayers(sentimentBowLayers).setSeed(Random.nextLong).setMaxIter(numIterations).setRawPredictionCol("bowClassifierPrediction")
    val sentimentLiwcTrainer = new MultilayerPerceptronClassifier().setPredictionCol("fjsdfaoi").setProbabilityCol("544").setLabelCol("sentiment").setFeaturesCol("liwcFeatures")
      .setLayers(sentimentLiwcFeatures).setSeed(Random.nextLong).setMaxIter(numIterations).setRawPredictionCol("word2VecClassifierPrediction")

    val relevanceMLPBowTrainer = new MultilayerPerceptronClassifier().setPredictionCol("fksdhfldsk").setProbabilityCol("123").setLabelCol("relevanceLabel").setFeaturesCol("topFeatures")
      .setLayers(relevanceBowLayers).setBlockSize(128).setSeed((Math.random() * 100000).toLong).setMaxIter(numIterations).setRawPredictionCol("bowClassifierPrediction")
    val relevanceLiwcTrainer = new MultilayerPerceptronClassifier().setPredictionCol("fsdfjsjojlk").setPredictionCol("345").setLabelCol("relevanceLabel").setFeaturesCol("liwcFeatures")
      .setLayers(relevanceLiwcFeatures).setBlockSize(128).setSeed((Math.random() * 100000).toLong).setMaxIter(numIterations).setRawPredictionCol("word2VecClassifierPrediction")

    val ensembleAssembler: VectorAssembler = new VectorAssembler().setInputCols(Array("bowClassifierPrediction", "word2VecClassifierPrediction")).setOutputCol("ensembleFeatures")
    val ensembleSentimentMLPTrainer = new MultilayerPerceptronClassifier().setLabelCol("sentiment").setFeaturesCol("ensembleFeatures").setLayers(Array(6, 3))
      .setSeed((Math.random() * 100000).toLong).setMaxIter(numIterations).setProbabilityCol("54343").setRawPredictionCol("4749")
    val ensembleRelevanceMLPTrainer = new MultilayerPerceptronClassifier().setLabelCol("relevanceLabel").setFeaturesCol("ensembleFeatures").setLayers(Array(4, 2))
      .setSeed((Math.random() * 100000).toLong).setMaxIter(numIterations).setProbabilityCol("r4w87").setRawPredictionCol("5490")

    println("fitting models")

    val sentimentPipelineMLP = new Pipeline().setStages(Array(languageTool, germanStemmer, hashingTermFrequencies, idf, sentimentChiSqSelector,
      sentimentMLPBowTrainer, sentimentLiwcTrainer, ensembleAssembler, ensembleSentimentMLPTrainer))
    val relevancePipelineMLP = new Pipeline().setStages(Array(languageTool, germanStemmer, hashingTermFrequencies, idf, relevanceChiSqSelector,
      relevanceMLPBowTrainer, relevanceLiwcTrainer, ensembleAssembler, ensembleRelevanceMLPTrainer))

    val sentimentModelMLP = sentimentPipelineMLP.fit(sentimentTrainingDataFrame)
    println("fitted sentiment models")
    val relevanceModelMLP = relevancePipelineMLP.fit(relevanceTrainingDataFrame)
    println("fitted relevance models")

    val sentimentValidationResult = sentimentModelMLP.transform(sentimentTestDataFrame)
    val relevanceValidationResult = relevanceModelMLP.transform(relevanceTestDataFrame)

    val sentimentEvaluator = new MultilabelMetrics(sentimentValidationResult.toDF().rdd.map(row => (Array(row.getAs[Double]("prediction")), Array(row.getAs[Int]("sentiment").toDouble))))
    val relevanceEvaluator = new MultilabelMetrics(relevanceValidationResult.toDF().rdd.map(row => (Array(row.getAs[Double]("prediction")), Array(row.getAs[Int]("relevanceLabel").toDouble))))

    println("(sentiment) f1 multilayer perceptron: " + sentimentEvaluator.microF1Measure)
    println("(relevance) f1 multilayer perceptron: " + relevanceEvaluator.microF1Measure)
  }

  /*
   * ============= Auxiliary stuff ============
   */

  /*
   * Auxiliary objects
   */

  private val logger: Logger = LoggerFactory.getLogger(GermevalWithEnsembleBowAndLiwc.getClass)

  /*
   * Auxiliary methods
   */


  def loadReviews(filename: String): Seq[Review] = {

    val result = Source.fromFile(filename, "utf-8")
      .getLines().toSeq.tail.par
      .map(_.split("\t"))
      .map(x =>
        new Review(
          terms = GermevalWordSequenceProcessor.processWordSequence(x(1)),
          relevant = x(2) == "true",
          sentiClass = if (x(3) == "negative") 0 else if (x(3) == "neutral") 1 else 2,
          liwcFeatures = x.slice(5, x.length).map(_.replace(",", ".")).map(_.toDouble)))
      .seq
    result
  }

  /*
   * Auxiliary classes
   */

  class Review(val terms: Seq[String], val relevant: Boolean, val sentiClass: Int, val liwcFeatures: Array[Double])

  class TermWithClass(val term: String, val sentimentClass: Int, val occurrences: Int)

}
*/