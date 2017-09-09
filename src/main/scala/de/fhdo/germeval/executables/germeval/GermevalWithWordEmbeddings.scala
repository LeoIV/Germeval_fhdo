package de.hoevelmann.bachelorthesis.executables.germeval

import de.fhdo.germeval.modelling.transformers.{FastTextSentenceVector, LanguageToolTransformer, SdCrossValidator, SentimentLexicon}
import de.hoevelmann.bachelorthesis.modelling.transformers.{LanguageToolTransformer, SdCrossValidator, SentimentLexicon}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.io.Source
import scala.util.Random

object GermevalWithWordEmbeddings {

  /**
    * Execute the Germeval task
    *
    */
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]").getOrCreate()
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    Logger.getLogger("TaskSetManager").setLevel(Level.ERROR)

    import spark.implicits._
    val trainingReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_train.csv")
    val validationReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_dev.csv")

    println("creating dataframes")

    val sentimentTrainingDataFrame: DataFrame = (for (review <- Random.shuffle(trainingReviews ++ validationReviews).take(200)) yield {
      (review.sentiClass, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("sentiment", "input", "liwcFeatures")
    val relevanceTrainingDataFrame: DataFrame = (for (review <- Random.shuffle(trainingReviews ++ validationReviews).take(200)) yield {
      (if (review.relevant) 1 else 0, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("relevanceLabel", "input", "liwcFeatures")

    println("creating preprocessors")

    val word2vecTransformer = new FastTextSentenceVector().setInputCol("input").setOutputCol("word2vecs")
    val sentimentLexicon = new SentimentLexicon().setInputCol("input").setOutputCol("senLexFeatures")
    val languageTool: LanguageToolTransformer = new LanguageToolTransformer().setInputCol("input").setOutputCol("languageToolFeatures")
    val assembler = new VectorAssembler().setInputCols(Array("word2vecs", "liwcFeatures", "senLexFeatures")).setOutputCol("assembledFeatures")
    val assemblerWithoutSenlex = new VectorAssembler().setInputCols(Array("word2vecs", "liwcFeatures","languageTool")).setOutputCol("assembledFeatures")


    val sentimentMLP = new MultilayerPerceptronClassifier()
      .setLabelCol("sentiment")
      .setFeaturesCol("assembledFeatures")
      .setLayers(Array(trainingReviews.head.liwcFeatures.length + 101 + sentimentLexicon.featuresLength, 3))
      .setBlockSize(128)
      .setSeed((Math.random() * 100000).toLong)
      .setMaxIter(100)

    val relevanceMLP = new MultilayerPerceptronClassifier()
      .setLabelCol("relevanceLabel")
      .setFeaturesCol("assembledFeatures")
      .setLayers(Array(trainingReviews.head.liwcFeatures.length + 101 + sentimentLexicon.featuresLength, 2))
      .setBlockSize(128)
      .setSeed((Math.random() * 100000).toLong)
      .setMaxIter(100)

    val sentimentMLR = new LogisticRegression().setLabelCol("sentiment").setFeaturesCol("assembledFeatures")
    val relevanceMLR = new LogisticRegression().setLabelCol("relevanceLabel").setFeaturesCol("assembledFeatures")

    val sentimentGBT = new OneVsRest().setClassifier(new GBTClassifier().setMaxDepth(10)).setLabelCol("sentiment").setFeaturesCol("assembledFeatures")
    val relevanceGBT = new GBTClassifier().setLabelCol("relevanceLabel").setFeaturesCol("assembledFeatures")

    val sentimentSVC = new OneVsRest().setClassifier(new LinearSVC()).setLabelCol("sentiment").setFeaturesCol("assembledFeatures")
    val relevanceSVC = new LinearSVC().setLabelCol("relevanceLabel").setFeaturesCol("assembledFeatures")

    println("fitting models")


    val sentimentPipelineMLP = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, assembler, sentimentMLP))
    val sentimentPipelineMLR = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, assembler, sentimentMLR))
    val sentimentPipelineGBT = new Pipeline().setStages(Array(word2vecTransformer, languageTool, sentimentLexicon, assemblerWithoutSenlex, sentimentGBT))
    val sentimentPipelineGBTWithoutSenlex = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, assemblerWithoutSenlex, sentimentGBT))
    val sentimentPipelineSVC = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, assembler, sentimentSVC))

    val relevancePipelineMLP = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, assembler, relevanceMLP))
    val relevancePipelineMLR = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, assembler, relevanceMLR))
    val relevancePipelineGBT = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, assembler, relevanceGBT))
    val relevancePipelineSVC = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, assembler, relevanceSVC))

    val paramGrid = new ParamGridBuilder()
      .build()


    val sentimentCVMlp = new SdCrossValidator()
      .setEstimator(sentimentPipelineMLP)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("sentiment").setMetricName("f1"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val sentimentCVMlr = new SdCrossValidator()
      .setEstimator(sentimentPipelineMLR)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("sentiment").setMetricName("f1"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val sentimentCVGbt = new SdCrossValidator()
      .setEstimator(sentimentPipelineGBT)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("sentiment").setMetricName("f1"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)

    val sentimentCVGbtWithoutSenlex = new SdCrossValidator()
      .setEstimator(sentimentPipelineGBTWithoutSenlex)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("sentiment").setMetricName("f1"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val sentimentCVSvc = new SdCrossValidator()
      .setEstimator(sentimentPipelineSVC)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("sentiment").setMetricName("f1"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val relevanceCVMlp = new SdCrossValidator()
      .setEstimator(relevancePipelineMLP)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("relevanceLabel").setMetricName("f1"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val relevanceCVMlr = new SdCrossValidator()
      .setEstimator(relevancePipelineMLR)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("relevanceLabel").setMetricName("f1"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val relevanceCVGbt = new SdCrossValidator()
      .setEstimator(relevancePipelineGBT)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("relevanceLabel").setMetricName("f1"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val relevanceCVSvc = new SdCrossValidator()
      .setEstimator(relevancePipelineSVC)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("relevanceLabel").setMetricName("f1"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)


    /*val sentimentModelMlp = sentimentCVMlp.fitSd(sentimentTrainingDataFrame)
    println("GermevalWithMLPAndWord2Vec sentiment mlp: " + sentimentModelMlp.allMetrics.toList)
    println("\taverage: " + sentimentModelMlp.average)

    val sentimentModelMlr = sentimentCVMlr.fitSd(sentimentTrainingDataFrame)
    println("GermevalWithMLPAndWord2Vec sentiment mlr: " + sentimentModelMlr.allMetrics.toList)
    println("\taverage: " + sentimentModelMlr.average)*/

    val sentimentModelGbt = sentimentCVGbt.fitSd(sentimentTrainingDataFrame)
    println("GermevalWithMLPAndWord2Vec sentiment gbt: " + sentimentModelGbt.allMetrics.toList)
    println("\taverage: " + sentimentModelGbt.average)

    val sentimentModelGbtWithoutSenlex = sentimentCVGbtWithoutSenlex.fitSd(sentimentTrainingDataFrame)
    println("GermevalWithMLPAndWord2Vec sentiment gbt without senlex: " + sentimentModelGbtWithoutSenlex.allMetrics.toList)
    println("\taverage: " + sentimentModelGbtWithoutSenlex.average)

    /*  val sentimentModelSvc = sentimentCVSvc.fitSd(sentimentTrainingDataFrame)
      println("fitted sentiment models")
      println("GermevalWithMLPAndWord2Vec sentiment svc: " + sentimentModelSvc.allMetrics.toList)
      println("\taverage: " + sentimentModelSvc.average)

      val relevanceMlpModel = relevanceCVMlp.fitSd(relevanceTrainingDataFrame)
      println("GermevalWithMLPAndWord2Vec relevance mlp: " + relevanceMlpModel.allMetrics.toList)
      println("\taverage: " + relevanceMlpModel.average)

      val relevanceMlrModel = relevanceCVMlr.fitSd(relevanceTrainingDataFrame)
      println("GermevalWithMLPAndWord2Vec relevance mlr: " + relevanceMlrModel.allMetrics.toList)
      println("\taverage: " + relevanceMlrModel.average)*/

    val relevanceGbtModel = relevanceCVGbt.fitSd(relevanceTrainingDataFrame)
    println("GermevalWithMLPAndWord2Vec relevance gbt: " + relevanceGbtModel.allMetrics.toList)
    println("\taverage: " + relevanceGbtModel.average)

    /* val relevanceSvcModel = relevanceCVSvc.fitSd(relevanceTrainingDataFrame)
     println("GermevalWithMLPAndWord2Vec relevance svc: " + relevanceSvcModel.allMetrics.toList)
     println("\taverage: " + relevanceSvcModel.average)
     println("fitted relevance models")*/



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
          id = x(1),
          termsUnprocessed = x(2),
          terms = GermevalWordSequenceProcessor.processWordSequence(x(1)),
          relevant = x(2) == "true",
          sentiClass = if (x(3) == "negative") 0 else if (x(3) == "neutral") 1 else 2,
          liwcFeatures = x.slice(5, x.length).map(_.replace(",", ".")).map(_.toDouble)
        )).seq
  }

  /*
   * Auxiliary classes
   */

  class Review(val id: String, val termsUnprocessed: String, val terms: Seq[String], val relevant: Boolean, val sentiClass: Int, val liwcFeatures: Array[Double])

  class TermWithClass(val term: String, val sentimentClass: Int, val occurrences: Int)

}