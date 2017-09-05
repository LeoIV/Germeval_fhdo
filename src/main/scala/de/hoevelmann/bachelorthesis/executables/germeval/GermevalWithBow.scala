package de.hoevelmann.bachelorthesis.executables.germeval

import de.hoevelmann.bachelorthesis.modelling.transformers._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.io.Source
import scala.util.Random

object GermevalWithBow {

  /**
    * Execute the Germeval task
    *
    * @param spark the SparkSession
    */
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]").getOrCreate()
    import spark.implicits._
    val trainingReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_train.csv")
    val testReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_dev.csv")

    spark.sparkContext.setLogLevel("ERROR")

    println("creating dataframes")

    val sentimentTrainingDataFrame: DataFrame = (for (review <- Random.shuffle(trainingReviews ++ testReviews)) yield {
      (review.sentiClass, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("sentiment", "input", "liwcFeatures")
    val relevanceTrainingDataFrame: DataFrame = (for (review <- Random.shuffle(trainingReviews ++ testReviews)) yield {
      (if (review.relevant) 1 else 0, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("relevanceLabel", "input", "liwcFeatures")

    println("creating preprocessors")

    /*
     * define the pipeline steps
     */

    //val languageTool: LanguageToolTransformer = new LanguageToolTransformer().setInputCol("input").setOutputCol("languageToolFeatures")
    val sentimentLexicon: SentimentLexiconCounter = new SentimentLexiconCounter().setInputCol("input").setOutputCol("senLexiconFeatures")
    val germanStemmer: GermanStemmer = new GermanStemmer().setInputCol("input").setOutputCol("stems")
    val word2vecTransformer = new FastTextSentenceVector().setInputCol("input").setOutputCol("word2vecs")
    val kMeansTransformer = new KMeansTransformer().setInputCol("word2vecs").setOutputCol("kMeansFeatures")
    // val wordCounter: WordCounter = new WordCounter().setInputCol("input").setOutputCol("wordCounts")
    //val nGramStep2: NGram = new NGram().setInputCol("stems").setOutputCol("bigrams").setN(2)
    //val nGramAssembler: StringArrayAssembler = new StringArrayAssembler().setInputCols(Array("bigrams", "stems")).setOutputCol("ngrams")
    val hashingTermFrequencies: HashingTF = new HashingTF().setInputCol("stems").setOutputCol("tfs").setNumFeatures(16384)
    val idf: IDF = new IDF().setInputCol("tfs").setOutputCol("idfs")
    val sentimentAssembler: VectorAssembler = new VectorAssembler().setInputCols(Array("idfs", "liwcFeatures")).setOutputCol("assembledFeatures")
    val relevanceAssembler: VectorAssembler = new VectorAssembler().setInputCols(Array("idfs", "liwcFeatures")).setOutputCol("assembledFeatures")

    val sentimentChiSqSelector: ChiSqSelector = new ChiSqSelector().setFeaturesCol("assembledFeatures").setLabelCol("sentiment").setOutputCol("topFeatures").setNumTopFeatures(1000)
    val relevanceChiSqSelector: ChiSqSelector = new ChiSqSelector().setFeaturesCol("assembledFeatures").setLabelCol("relevanceLabel").setOutputCol("topFeatures").setNumTopFeatures(1000)

    val postChiqSqSelectionAssembler: VectorAssembler = new VectorAssembler().setInputCols(Array("topFeatures", "senLexiconFeatures", "kMeansFeatures")).setOutputCol("topAndSenFeatures")


    /*
     * mlp model definition
     */

    def sentimentLayers: Array[Int] = Array(1100 + sentimentLexicon.featuresLength, 500, 3)

    def relevanceLayers: Array[Int] = Array(1100 + sentimentLexicon.featuresLength, 500, 2)

    val sentimentMLPTrainer = new MultilayerPerceptronClassifier().setLabelCol("sentiment").setFeaturesCol("topAndSenFeatures").setLayers(sentimentLayers).setSeed(42L).setMaxIter(100)
    val sentimentGBTTrainer = new OneVsRest().setClassifier(new GBTClassifier().setMaxDepth(10).setMaxIter(100)).setLabelCol("sentiment").setFeaturesCol("topAndSenFeatures")
    val sentimentSVCTrainer = new OneVsRest().setClassifier(new LinearSVC()).setLabelCol("sentiment").setFeaturesCol("topAndSenFeatures")
    val sentimentMLRTrainer = new LogisticRegression().setLabelCol("sentiment").setFeaturesCol("topAndSenFeatures")

    val relevanceGBTTrainer = new GBTClassifier().setLabelCol("relevanceLabel").setFeaturesCol("topAndSenFeatures")
    val relevanceMLPTrainer = new MultilayerPerceptronClassifier().setLabelCol("relevanceLabel").setFeaturesCol("topAndSenFeatures").setLayers(relevanceLayers).setSeed(42L).setMaxIter(100)
    val relevanceSVCTrainer = new LinearSVC().setLabelCol("relevanceLabel").setFeaturesCol("topAndSenFeatures")
    val relevanceMLRTrainer = new LogisticRegression().setLabelCol("relevanceLabel").setFeaturesCol("topAndSenFeatures")

    println("fitting models")

    //val sentimentPipelineMLP = new Pipeline().setStages(Array(sentimentLexicon, word2vecTransformer, kMeansTransformer, germanStemmer, new Cacher(), hashingTermFrequencies, idf, sentimentAssembler,
    // sentimentChiSqSelector, postChiqSqSelectionAssembler, sentimentMLPTrainer))
    val sentimentPipelineGBT = new Pipeline().setStages(Array(sentimentLexicon, word2vecTransformer, kMeansTransformer, germanStemmer, new Cacher(), hashingTermFrequencies, idf, sentimentAssembler,
      sentimentChiSqSelector, postChiqSqSelectionAssembler, sentimentGBTTrainer))
    /* val sentimentPipelineSVC = new Pipeline().setStages(Array(sentimentLexicon, word2vecTransformer, kMeansTransformer, germanStemmer, new Cacher(), hashingTermFrequencies, idf, sentimentAssembler,
       sentimentChiSqSelector, postChiqSqSelectionAssembler, sentimentSVCTrainer))
     val sentimentPipelineMLR = new Pipeline().setStages(Array(sentimentLexicon, word2vecTransformer, kMeansTransformer, germanStemmer, new Cacher(), hashingTermFrequencies, idf, sentimentAssembler,
       sentimentChiSqSelector, postChiqSqSelectionAssembler, sentimentMLRTrainer)) */

    val relevancePipelineGBT = new Pipeline().setStages(Array(sentimentLexicon, word2vecTransformer, kMeansTransformer, germanStemmer, new Cacher(), hashingTermFrequencies, idf, relevanceAssembler,
      relevanceChiSqSelector, postChiqSqSelectionAssembler, relevanceGBTTrainer))
    /* val relevancePipelineMLP = new Pipeline().setStages(Array(sentimentLexicon, word2vecTransformer, kMeansTransformer, germanStemmer, new Cacher(), hashingTermFrequencies, idf, relevanceAssembler,
      relevanceChiSqSelector, postChiqSqSelectionAssembler, relevanceMLPTrainer))
    val relevancePipelineSVC = new Pipeline().setStages(Array(sentimentLexicon, word2vecTransformer, kMeansTransformer, germanStemmer, new Cacher(), hashingTermFrequencies, idf, relevanceAssembler,
      relevanceChiSqSelector, postChiqSqSelectionAssembler, relevanceSVCTrainer))
    val relevancePipelineMLR = new Pipeline().setStages(Array(sentimentLexicon, word2vecTransformer, kMeansTransformer, germanStemmer, new Cacher(), hashingTermFrequencies, idf, relevanceAssembler,
      relevanceChiSqSelector, postChiqSqSelectionAssembler, relevanceMLRTrainer)) */

    val paramGrid = new ParamGridBuilder()
      .build()

   /* val sentimentCVMLp = new SdCrossValidator()
      .setEstimator(sentimentPipelineMLP)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("sentiment").setMetricName("f1"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)*/

     val sentimentCVGbt = new SdCrossValidator()
         .setEstimator(sentimentPipelineGBT)
         .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("sentiment").setMetricName("f1"))
         .setEstimatorParamMaps(paramGrid)
         .setNumFolds(5)

    /*    val sentimentCVSvc = new SdCrossValidator()
        .setEstimator(sentimentPipelineSVC)
        .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("sentiment").setMetricName("f1"))
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(5)

      val sentimentCVMlr = new SdCrossValidator()
        .setEstimator(sentimentPipelineMLR)
        .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("sentiment").setMetricName("f1"))
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(5) */

       /*   val relevanceCVMlp = new SdCrossValidator()
           .setEstimator(relevancePipelineMLP)
           .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("relevanceLabel").setMetricName("f1"))
           .setEstimatorParamMaps(paramGrid)
           .setNumFolds(5)*/

           val relevanceCVGbt = new SdCrossValidator()
              .setEstimator(relevancePipelineGBT)
              .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("relevanceLabel").setMetricName("f1"))
              .setEstimatorParamMaps(paramGrid)
              .setNumFolds(5)

      /*      val relevanceCVSvc = new SdCrossValidator()
              .setEstimator(relevancePipelineSVC)
              .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("relevanceLabel").setMetricName("f1"))
              .setEstimatorParamMaps(paramGrid)
              .setNumFolds(5)

            val relevanceCVMlr = new SdCrossValidator()
              .setEstimator(relevancePipelineMLR)
              .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("relevanceLabel").setMetricName("f1"))
              .setEstimatorParamMaps(paramGrid)*/
      .setNumFolds(5)

    //val sentimentModelMlp = sentimentCVMLp.fitSd(sentimentTrainingDataFrame)
     val sentimentModelGbt = sentimentCVGbt.fitSd(sentimentTrainingDataFrame)
    // val sentimentModelSvc = sentimentCVSvc.fitSd(sentimentTrainingDataFrame)
    //val sentimentModelMlr = sentimentCVMlr.fitSd(sentimentTrainingDataFrame)
    println("fitted sentiment models")
     val relevanceGbtModel = relevanceCVGbt.fitSd(relevanceTrainingDataFrame)
    //val relevanceModelMlp = relevanceCVMlp.fitSd(relevanceTrainingDataFrame)
    // val relevanceModelSvc = relevanceCVSvc.fitSd(relevanceTrainingDataFrame)
    // val relevanceModelMlr = relevanceCVMlr.fitSd(relevanceTrainingDataFrame)
    println("fitted relevance models")

    //println("GermevalWithMLPAndBow sentiment mlp: " + sentimentModelMlp.allMetrics.toList)
    //println("\taverage: " + sentimentModelMlp.average)
     println("GermevalWithMLPAndBow sentiment gbt: " + sentimentModelGbt.allMetrics.toList)
     println("\taverage: " + sentimentModelGbt.average)
    /* println("GermevalWithMLPAndBow sentiment svc: " + sentimentModelSvc.allMetrics.toList)
     println("\taverage: " + sentimentModelSvc.average)
     println("GermevalWithMLPAndBow sentiment mlr: " + sentimentModelMlr.allMetrics.toList)
     println("\taverage: " + sentimentModelMlr.average) */

    //println("GermevalWithMLPAndBow relevance mlp: " + relevanceModelMlp.allMetrics.toList)
    //println("\taverage: " + relevanceModelMlp.average)
     println("GermevalWithMLPAndBow relevance gbt: " + relevanceGbtModel.allMetrics.toList)
    println("\taverage: " + relevanceGbtModel.average)
    /*println("GermevalWithMLPAndBow relevance svc: " + relevanceModelSvc.allMetrics.toList)
    println("\taverage: " + relevanceModelSvc.average)
    println("GermevalWithMLPAndBow relevance mlr: " + relevanceModelMlr.allMetrics.toList)
    println("\taverage: " + relevanceModelMlr.average) */

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
          terms = GermevalWordSequenceProcessor.processWordSequence(x(1)),
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

  class Review(val terms: Seq[String], val relevant: Boolean, val sentiClass: Int, val liwcFeatures: Array[Double])

  class TermWithClass(val term: String, val sentimentClass: Int, val occurrences: Int)

}