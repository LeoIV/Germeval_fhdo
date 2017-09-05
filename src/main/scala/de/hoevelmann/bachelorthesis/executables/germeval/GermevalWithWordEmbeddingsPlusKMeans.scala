package de.hoevelmann.bachelorthesis.executables.germeval

import de.hoevelmann.bachelorthesis.modelling.transformers._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.io.Source
import scala.util.Random

object GermevalWithWordEmbeddingsPlusKMeans {

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
  def execute(spark: SparkSession = SparkSession.builder().master("local[*]").getOrCreate(), ngramSize: Int = 1, numFeatures: Int = 1000, intermediateLayers: Array[Int] = Array(300), useLIWCFeatures: Boolean = true, stem: Boolean = true, numIterations: Int = 100): Unit = {
    import spark.implicits._

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    Logger.getLogger("TaskSetManager").setLevel(Level.ERROR)
    val trainingReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_train.csv")
    val testReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_dev.csv")

    println("creating dataframes")

    val sentimentTrainingDataFrame: DataFrame = (for (review <- Random.shuffle(trainingReviews ++ testReviews)) yield {
      (review.sentiClass, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("sentiment", "input", "liwcFeatures")

    println("creating preprocessors")

    val word2vecTransformer = new FastTextSentenceVector().setInputCol("input").setOutputCol("word2vecs")
    val sentimentLexicon = new SentimentLexicon().setInputCol("input").setOutputCol("senLexFeatures")
    val kMeans = new KMeansTransformer().setInputCol("word2vecs").setOutputCol("kMeansFeatures")
    val languageTool: LanguageToolTransformer = new LanguageToolTransformer().setInputCol("input").setOutputCol("languageToolFeatures")
    val assemblerSenLexKMeansLT = new VectorAssembler().setInputCols(Array("word2vecs", "liwcFeatures", "senLexFeatures", "kMeansFeatures", "languageToolFeatures")).setOutputCol("assembledFeatures")
    val assemblerSenLexKMeans = new VectorAssembler().setInputCols(Array("word2vecs", "liwcFeatures", "senLexFeatures", "kMeansFeatures")).setOutputCol("assembledFeatures")
    val assemblerSenLex = new VectorAssembler().setInputCols(Array("word2vecs", "liwcFeatures", "senLexFeatures")).setOutputCol("assembledFeatures")
    val assemblerLiwc = new VectorAssembler().setInputCols(Array("word2vecs", "liwcFeatures")).setOutputCol("assembledFeatures")
    val assembler = new VectorAssembler().setInputCols(Array("word2vecs")).setOutputCol("assembledFeatures")

    val sentimentGBTDepth5 = new OneVsRest().setClassifier(new GBTClassifier()).setLabelCol("sentiment").setFeaturesCol("assembledFeatures")
    val sentimentGBTDepth8 = new OneVsRest().setClassifier(new GBTClassifier().setMaxDepth(8)).setLabelCol("sentiment").setFeaturesCol("assembledFeatures")

    println("fitting models")

    val sentimentPipelineGBTSenLexKMeansLT = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, kMeans, languageTool, assemblerSenLexKMeansLT, sentimentGBTDepth5))
    val sentimentPipelineGBTSenLexKMeansLTDepth8 = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, kMeans, languageTool, assemblerSenLexKMeansLT, sentimentGBTDepth8))

    val sentimentPipelineGBTSenLexKMeans = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, kMeans, assemblerSenLexKMeans, sentimentGBTDepth5))
    val sentimentPipelineGBTSenLexKMeansDepth8 = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, kMeans, assemblerSenLexKMeans, sentimentGBTDepth8))

    val sentimentPipelineGBTSenLex = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, assemblerSenLex, sentimentGBTDepth5))
    val sentimentPipelineGBTSenLexDepth8 = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, assemblerSenLex, sentimentGBTDepth8))

    val sentimentPipelineGBT = new Pipeline().setStages(Array(word2vecTransformer,sentimentLexicon,assemblerSenLex,new LogisticRegression().setLabelCol("sentiment").setFeaturesCol("assembledFeatures")))
    val sentimentPipelineGBTDepth8 = new Pipeline().setStages(Array(word2vecTransformer,sentimentLexicon,assemblerSenLex,new OneVsRest().setClassifier(new GBTClassifier().setMaxDepth(8)).setLabelCol("sentiment").setFeaturesCol("assembledFeatures")))

    val paramGrid = new ParamGridBuilder()
      .build()

    val sentimentCVGbtDepth5 = new SdCrossValidator().setEstimator(sentimentPipelineGBT).setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("sentiment").setMetricName("f1")).setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)

    val sentimentCVGbtDepth8 = new SdCrossValidator().setEstimator(sentimentPipelineGBTDepth8).setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("sentiment").setMetricName("f1")).setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)


    val sentimentCVGbtSenLexKMeansLTDepth5 = new SdCrossValidator().setEstimator(sentimentPipelineGBTSenLexKMeansLT).setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("sentiment").setMetricName("f1")).setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val sentimentCVGbtSenLexKMeansDepth5 = new SdCrossValidator().setEstimator(sentimentPipelineGBTSenLexKMeans).setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("sentiment").setMetricName("f1")).setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val sentimentCVGbtSenLexDepth5 = new SdCrossValidator().setEstimator(sentimentPipelineGBTSenLex).setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("sentiment").setMetricName("f1")).setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val sentimentCVGbtSenLexKMeansLTDepth8 = new SdCrossValidator().setEstimator(sentimentPipelineGBTSenLexKMeansLTDepth8).setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("sentiment").setMetricName("f1")).setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val sentimentCVGbtSenLexKMeansDepth8 = new SdCrossValidator().setEstimator(sentimentPipelineGBTSenLexKMeansDepth8).setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("sentiment").setMetricName("f1")).setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val sentimentCVGbtSenLexDepth8 = new SdCrossValidator().setEstimator(sentimentPipelineGBTSenLexDepth8).setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("sentiment").setMetricName("f1")).setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)


    val sentimentModelGbtDepth5 = sentimentCVGbtDepth5.fitSd(sentimentTrainingDataFrame)
    println("sentimentCVGbtDepth5: " + sentimentModelGbtDepth5.allMetrics.toList)
    println("\taverage: " + sentimentModelGbtDepth5.average)

    val sentimentModelGbtDepth8 = sentimentCVGbtDepth8.fitSd(sentimentTrainingDataFrame)
    println("sentimentCVGbtDepth8: " + sentimentModelGbtDepth8.allMetrics.toList)
    println("\taverage: " + sentimentModelGbtDepth8.average)

    val sentimentModelGbtSenLexKMeansDepth5 = sentimentCVGbtSenLexKMeansDepth5.fitSd(sentimentTrainingDataFrame)
    println("sentimentCVGbtSenLexKMeansDepth5: " + sentimentModelGbtSenLexKMeansDepth5.allMetrics.toList)
    println("\taverage: " + sentimentModelGbtSenLexKMeansDepth5.average)

    val sentimentModelGbtSenLexKMeansDepth8 = sentimentCVGbtSenLexKMeansDepth8.fitSd(sentimentTrainingDataFrame)
    println("sentimentCVGbtSenLexKMeansDepth8: " + sentimentModelGbtSenLexKMeansDepth8.allMetrics.toList)
    println("\taverage: " + sentimentModelGbtSenLexKMeansDepth8.average)

    val sentimentModelGbtSenLexDepth5 = sentimentCVGbtSenLexDepth5.fitSd(sentimentTrainingDataFrame)
    println("sentimentCVGbtSenLexDepth5: " + sentimentModelGbtSenLexDepth5.allMetrics.toList)
    println("\taverage: " + sentimentModelGbtSenLexDepth5.average)

    val sentimentModelGbtSenLexDepth8 = sentimentCVGbtSenLexDepth8.fitSd(sentimentTrainingDataFrame)
    println("sentimentCVGbtSenLexDepth8: " + sentimentModelGbtSenLexDepth8.allMetrics.toList)
    println("\taverage: " + sentimentModelGbtSenLexDepth8.average)

    val sentimentModelGbtSenLexKMeansLTDepth5 = sentimentCVGbtSenLexKMeansLTDepth5.fitSd(sentimentTrainingDataFrame)
    println("sentimentCVGbtSenLexKMeansLTDepth5: " + sentimentModelGbtSenLexKMeansLTDepth5.allMetrics.toList)
    println("\taverage: " + sentimentModelGbtSenLexKMeansLTDepth5.average)

    val sentimentModelGbtSenLexKMeansLTDepth8 = sentimentCVGbtSenLexKMeansLTDepth8.fitSd(sentimentTrainingDataFrame)
    println("sentimentCVGbtSenLexKMeansLTDepth8: " + sentimentModelGbtSenLexKMeansLTDepth8.allMetrics.toList)
    println("\taverage: " + sentimentModelGbtSenLexKMeansLTDepth8.average)


  }

  /*
   * ============= Auxiliary stuff ============
   */

  /*
   * Auxiliary objects
   */


  def loadReviews(filename: String): Seq[Review] = {
    Source.fromFile(filename, "utf-8")
      .getLines().toSeq.tail.par
      .map(_.split("\t"))
      .map(x =>
        new Review(
          terms = GermevalWordSequenceProcessor.processWordSequence(x(1)),
          relevant = x(2) == "true",
          sentiClass = if (x(3) == "negative") 0 else if (x(3) == "neutral") 1 else 2,
          liwcFeatures = x.slice(5, x.length).map(_.replace(",", ".")).map(_.toDouble)
        )).seq
  }

  /*
   * Auxiliary classes
   */

  class Review(val terms: Seq[String], val relevant: Boolean, val sentiClass: Int, val liwcFeatures: Array[Double])

  class TermWithClass(val term: String, val sentimentClass: Int, val occurrences: Int)

}