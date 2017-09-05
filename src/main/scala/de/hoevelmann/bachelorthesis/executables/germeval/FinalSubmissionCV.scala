package de.hoevelmann.bachelorthesis.executables.germeval

import de.hoevelmann.bachelorthesis.modelling.transformers.{SdCrossValidatorModel, _}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Estimator, Pipeline}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.io.Source
import scala.util.{Failure, Random, Success, Try}

object FinalSubmissionCV {

  /**
    * Execute the Germeval task
    *
    */
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    Logger.getLogger("TaskSetManager").setLevel(Level.ERROR)

    import spark.implicits._
    val trainingReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_train.csv")
    val validationReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_dev.csv")

    println("FINAL SUBMISSION")

    val sentimentTrainingDf: DataFrame = (for (review <- Random.shuffle(trainingReviews)) yield {
      (review.sentiClass, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("sentiment", "input", "liwcFeatures").cache()

    val sentimentValidationDf: DataFrame = (for (review <- Random.shuffle(validationReviews)) yield {
      (review.sentiClass, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("sentiment", "input", "liwcFeatures").cache()

    val relevanceTrainingDf: DataFrame = (for (review <- Random.shuffle(trainingReviews)) yield {
      (if (review.relevant) 1 else 0, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("relevanceLabel", "input", "liwcFeatures").cache()

    val relevanceValidationgDf: DataFrame = (for (review <- Random.shuffle(validationReviews)) yield {
      (if (review.relevant) 1 else 0, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("relevanceLabel", "input", "liwcFeatures").cache()

    def mergedSentimentDf: DataFrame = sentimentTrainingDf.union(sentimentValidationDf)

    def mergedRelevanceDf: DataFrame = relevanceTrainingDf.union(relevanceValidationgDf)

    def crossvalidatorSentiment(pipeline: Estimator[_]): String = Try(new SdCrossValidator().setEstimator(pipeline).setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("sentiment").setMetricName("f1")).setEstimatorParamMaps(new ParamGridBuilder()
      .build()).setNumFolds(5).fitSd(mergedSentimentDf)) match {
      case Failure(thrown) =>
        thrown.printStackTrace()
        "failure"
      case Success(s) => s.allMetrics.mkString(" | ")
    }

    def trainValidationSplit(pipeline: Pipeline, trainingDf: DataFrame, validationDf: DataFrame, labelColumn: String): String =
      Try {
        val model = pipeline.fit(trainingDf)
        val predictions = model.transform(validationDf)
        val scorer = new MulticlassClassificationEvaluator()
          .setLabelCol(labelColumn)
          .setMetricName("f1")
        scorer.evaluate(predictions)
      } match {
        case Failure(thrown) =>
          thrown.printStackTrace()
          "failure"
        case Success(s) => s.toString
      }


    def crossvalidatorRelevance(pipeline: Estimator[_]): String = Try(new SdCrossValidator().setEstimator(pipeline).setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("relevanceLabel").setMetricName("f1")).setEstimatorParamMaps(new ParamGridBuilder()
      .build()).setNumFolds(5).fitSd(mergedRelevanceDf)) match {
      case Failure(thrown) =>
        thrown.printStackTrace()
        "failure"
      case Success(s) => s.allMetrics.mkString(" | ")
    }

    println("creating preprocessors")

    // TRANSFORMER DEFINITION

    val sentimentLexicon = new SentimentLexicon().setInputCol("input").setOutputCol("senLexFeatures")

    val word2vecTransformer = new FastTextSentenceVector().setInputCol("input").setOutputCol("word2vecs")

    val germanStemmer: GermanStemmer = new GermanStemmer().setInputCol("input").setOutputCol("stems")

    val hashingTf = new HashingTF().setInputCol("stems").setOutputCol("tfs").setNumFeatures(16384)

    val idfTr = new IDF().setInputCol("tfs").setOutputCol("idfs")

    val languageTool: LanguageToolTransformer = new LanguageToolTransformer().setInputCol("input").setOutputCol("languageToolFeatures")

    val bowAssemblerWithLiwc = new VectorAssembler().setInputCols(Array("idfs", "liwcFeatures")).setOutputCol("mlpAssembly")
    val bowAssemblerWithLanguageTool = new VectorAssembler().setInputCols(Array("idfs", "languageToolFeatures")).setOutputCol("mlpAssembly")
    val bowAssembler = new VectorAssembler().setInputCols(Array("idfs")).setOutputCol("mlpAssembly")
    val sentimentChi2selector = new ChiSqSelector().setFeaturesCol("mlpAssembly").setLabelCol("sentiment").setOutputCol("topFeatures").setNumTopFeatures(1000)
    val relevanceChi2selector = new ChiSqSelector().setFeaturesCol("mlpAssembly").setLabelCol("relevanceLabel").setOutputCol("topFeatures").setNumTopFeatures(1000)


    val word2VecAssembler = new VectorAssembler().setInputCols(Array("word2vecs")).setOutputCol("assembledFeatures")
    val word2VecAssemblerWithLiwc = new VectorAssembler().setInputCols(Array("word2vecs", "liwcFeatures")).setOutputCol("assembledFeatures")
    val word2VecAssemblerWithSenLex = new VectorAssembler().setInputCols(Array("word2vecs", "senLexFeatures")).setOutputCol("assembledFeatures")
    val word2VecAssemblerWithLiwcAndSenLex = new VectorAssembler().setInputCols(Array("word2vecs", "liwcFeatures", "senLexFeatures")).setOutputCol("assembledFeatures")

    //fasttext classifiers

    val fastTextSentimentPredictor = new FastTextClassifier().setLabelCol("sentiment").setFeaturesCol("stems")
    val fastTextRelevancePredictor = new FastTextClassifier().setFeaturesCol("stems").setLabelCol("relevanceLabel")

    //gbt classifiers

    val gbtW2vSentimentPredictor = new OneVsRest().setClassifier(new GBTClassifier().setMaxIter(30).setMaxDepth(10)).setLabelCol("sentiment").setFeaturesCol("assembledFeatures")
    val gbtW2vRelevancePredictor = new GBTClassifier().setMaxIter(100).setMaxDepth(10).setLabelCol("relevanceLabel").setFeaturesCol("assembledFeatures")

    val gbtBowSentimentPredictor = new OneVsRest().setClassifier(new GBTClassifier().setMaxIter(30).setMaxDepth(10)).setLabelCol("sentiment").setFeaturesCol("topFeatures")
    val gbtBowRelevancePredictor = new GBTClassifier().setMaxIter(100).setMaxDepth(10).setLabelCol("relevanceLabel").setFeaturesCol("topFeatures")

    //mlp classifiers

    def mlpSentimentPredictor(featureColumn: String, layers: Array[Int]) = new MultilayerPerceptronClassifier().setLabelCol("sentiment").setFeaturesCol(featureColumn).setLayers(layers)

    def mlpRelevancePredictor(featureColumn: String, layers: Array[Int]) = new MultilayerPerceptronClassifier().setLabelCol("relevanceLabel").setFeaturesCol(featureColumn).setLayers(layers)


    // PIPELINE DEFINITION

    //gbt

    val top1000FromIdfsAndLiwcTrainedOnGbt_sentiment = new Pipeline().setStages(Array(germanStemmer, hashingTf, idfTr, bowAssemblerWithLiwc, sentimentChi2selector, gbtBowSentimentPredictor))
    val top1000FromIdfsAndLiwcTrainedOnGbt_relevance = new Pipeline().setStages(Array(germanStemmer, hashingTf, idfTr, bowAssemblerWithLiwc, relevanceChi2selector, gbtBowRelevancePredictor))

    val top1000FromIdfsTrainedOnGbt_sentiment = new Pipeline().setStages(Array(germanStemmer, hashingTf, idfTr, bowAssembler, sentimentChi2selector, gbtBowSentimentPredictor))
    val top1000FromIdfsTrainedOnGbt_relevance = new Pipeline().setStages(Array(germanStemmer, hashingTf, idfTr, bowAssembler, relevanceChi2selector, gbtBowRelevancePredictor))

    val top1000FromIdfsAndLanguageToolTrainedOnGbt_sentiment = new Pipeline().setStages(Array(languageTool, germanStemmer, hashingTf, idfTr, bowAssemblerWithLanguageTool, sentimentChi2selector, gbtBowSentimentPredictor))
    val top1000FromIdfsAndLanguageToolTrainedOnGbt_relevance = new Pipeline().setStages(Array(languageTool, germanStemmer, hashingTf, idfTr, bowAssemblerWithLanguageTool, relevanceChi2selector, gbtBowRelevancePredictor))

    val word2vecsTrainedOnGbt_sentiment = new Pipeline().setStages(Array(word2vecTransformer, word2VecAssembler, gbtW2vSentimentPredictor))
    val word2vecsTrainedOnGbt_relevance = new Pipeline().setStages(Array(word2vecTransformer, word2VecAssembler, gbtW2vRelevancePredictor))

    val word2vecsWithLiwcTrainedOnGbt_sentiment = new Pipeline().setStages(Array(word2vecTransformer, word2VecAssemblerWithLiwc, gbtW2vSentimentPredictor))
    val word2vecsWithLiwcTrainedOnGbt_relevance = new Pipeline().setStages(Array(word2vecTransformer, word2VecAssemblerWithLiwc, gbtW2vRelevancePredictor))

    val word2vecsWithSentimentLexiconTrainedOnGbt_sentiment = new Pipeline().setStages(Array(sentimentLexicon, word2vecTransformer, word2VecAssemblerWithSenLex, gbtW2vSentimentPredictor))
    val word2vecsWithSentimentLexiconTrainedOnGbt_relevance = new Pipeline().setStages(Array(sentimentLexicon, word2vecTransformer, word2VecAssemblerWithSenLex, gbtW2vRelevancePredictor))

    val word2vecsWithSentimentLexiconAndLiwcTrainedOnGbt_sentiment = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, word2VecAssemblerWithLiwcAndSenLex, gbtW2vSentimentPredictor))
    val word2vecsWithSentimentLexiconAndLiwcTrainedOnGbt_relevance = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, word2VecAssemblerWithLiwcAndSenLex, gbtW2vRelevancePredictor))

    //mlp

    val top1000FromIdfsAndLiwcTrainedOnMlp_sentiment = new Pipeline().setStages(Array(germanStemmer, hashingTf, idfTr, bowAssemblerWithLiwc, sentimentChi2selector, mlpSentimentPredictor("topFeatures", Array(1000, 500, 3))))
    val top1000FromIdfsAndLiwcTrainedOnMlp_relevance = new Pipeline().setStages(Array(germanStemmer, hashingTf, idfTr, bowAssemblerWithLiwc, relevanceChi2selector, mlpRelevancePredictor("topFeatures", Array(1000, 500, 2))))

    val top1000FromIdfsTrainedOnMlp_sentiment = new Pipeline().setStages(Array(germanStemmer, hashingTf, idfTr, bowAssembler, sentimentChi2selector, mlpSentimentPredictor("topFeatures", Array(1000, 500, 3))))
    val top1000FromIdfsTrainedOnMlp_relevance = new Pipeline().setStages(Array(germanStemmer, hashingTf, idfTr, bowAssembler, relevanceChi2selector, mlpRelevancePredictor("topFeatures", Array(1000, 500, 2))))

    val top1000FromIdfsAndLanguageToolTrainedOnMlp_sentiment = new Pipeline().setStages(Array(languageTool, germanStemmer, hashingTf, idfTr, bowAssemblerWithLanguageTool, sentimentChi2selector, mlpSentimentPredictor("topFeatures", Array(1000, 500, 3))))
    val top1000FromIdfsAndLanguageToolTrainedOnMlp_relevance = new Pipeline().setStages(Array(languageTool, germanStemmer, hashingTf, idfTr, bowAssemblerWithLanguageTool, relevanceChi2selector, mlpRelevancePredictor("topFeatures", Array(1000, 500, 2))))

    val word2vecsTrainedOnMlp_sentiment = new Pipeline().setStages(Array(word2vecTransformer, word2VecAssembler, mlpSentimentPredictor("assembledFeatures", Array(100, 3))))
    val word2vecsTrainedOnMlp_relevance = new Pipeline().setStages(Array(word2vecTransformer, word2VecAssembler, mlpRelevancePredictor("assembledFeatures", Array(100, 2))))

    val word2vecsWithLiwcTrainedOnMlp_sentiment = new Pipeline().setStages(Array(word2vecTransformer, word2VecAssemblerWithLiwc, mlpSentimentPredictor("assembledFeatures", Array(193, 3))))
    val word2vecsWithLiwcTrainedOnMlp_relevance = new Pipeline().setStages(Array(word2vecTransformer, word2VecAssemblerWithLiwc, mlpRelevancePredictor("assembledFeatures", Array(193, 2))))

    val word2vecsWithSentimentLexiconTrainedOnMlp_sentiment = new Pipeline().setStages(Array(sentimentLexicon, word2vecTransformer, word2VecAssemblerWithSenLex, mlpSentimentPredictor("assembledFeatures", Array(sentimentLexicon.featuresLength + 100, 500, 3))))
    val word2vecsWithSentimentLexiconTrainedOnMlp_relevance = new Pipeline().setStages(Array(sentimentLexicon, word2vecTransformer, word2VecAssemblerWithSenLex, mlpRelevancePredictor("assembledFeatures", Array(sentimentLexicon.featuresLength + 100, 500, 2))))

    val word2vecsWithSentimentLexiconAndLiwcTrainedOnMlp_sentiment = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, word2VecAssemblerWithLiwcAndSenLex, mlpSentimentPredictor("assembledFeatures", Array(sentimentLexicon.featuresLength + 193, 500, 3))))
    val word2vecsWithSentimentLexiconAndLiwcTrainedOnMlp_relevance = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, word2VecAssemblerWithLiwcAndSenLex, mlpRelevancePredictor("assembledFeatures", Array(sentimentLexicon.featuresLength + 193, 500, 2))))

    //fasttext

    val germanStemsTrainedOnFasttext_sentiment = new Pipeline().setStages(Array(germanStemmer, fastTextSentimentPredictor))
    val germanStemsTrainedOnFasttext_relevance = new Pipeline().setStages(Array(germanStemmer, fastTextRelevancePredictor))

    // RUNNING MODELS

    println("germanStemsTrainedOnFasttext")
    println(crossvalidatorSentiment(germanStemsTrainedOnFasttext_sentiment))
    println(crossvalidatorRelevance(germanStemsTrainedOnFasttext_relevance))
    println(trainValidationSplit(germanStemsTrainedOnFasttext_sentiment, sentimentTrainingDf, sentimentValidationDf, "sentiment"))
    println(trainValidationSplit(germanStemsTrainedOnFasttext_relevance, relevanceTrainingDf, relevanceValidationgDf, "relevanceLabel"))

    println("top1000FromIdfsAndLiwcTrainedOnGbt")
    println(crossvalidatorSentiment(top1000FromIdfsAndLiwcTrainedOnGbt_sentiment))
    println(crossvalidatorRelevance(top1000FromIdfsAndLiwcTrainedOnGbt_relevance))
    println(trainValidationSplit(top1000FromIdfsAndLiwcTrainedOnGbt_sentiment, sentimentTrainingDf, sentimentValidationDf, "sentiment"))
    println(trainValidationSplit(top1000FromIdfsAndLiwcTrainedOnGbt_relevance, relevanceTrainingDf, relevanceValidationgDf, "relevanceLabel"))

    println("top1000FromIdfsTrainedOnGbt")
    println(crossvalidatorSentiment(top1000FromIdfsTrainedOnGbt_sentiment))
    println(crossvalidatorRelevance(top1000FromIdfsTrainedOnGbt_relevance))
    println(trainValidationSplit(top1000FromIdfsTrainedOnGbt_sentiment, sentimentTrainingDf, sentimentValidationDf, "sentiment"))
    println(trainValidationSplit(top1000FromIdfsTrainedOnGbt_relevance, relevanceTrainingDf, relevanceValidationgDf, "relevanceLabel"))

    println("top1000FromIdfsAndLanguageToolTrainedOnGbt")
    println(crossvalidatorSentiment(top1000FromIdfsAndLanguageToolTrainedOnGbt_sentiment))
    println(crossvalidatorRelevance(top1000FromIdfsAndLanguageToolTrainedOnGbt_relevance))
    println(trainValidationSplit(top1000FromIdfsAndLanguageToolTrainedOnGbt_sentiment, sentimentTrainingDf, sentimentValidationDf, "sentiment"))
    println(trainValidationSplit(top1000FromIdfsAndLanguageToolTrainedOnGbt_relevance, relevanceTrainingDf, relevanceValidationgDf, "relevanceLabel"))

    println("word2vecsTrainedOnGbt")
    println(crossvalidatorSentiment(word2vecsTrainedOnGbt_sentiment))
    println(crossvalidatorRelevance(word2vecsTrainedOnGbt_relevance))
    println(trainValidationSplit(word2vecsTrainedOnGbt_sentiment, sentimentTrainingDf, sentimentValidationDf, "sentiment"))
    println(trainValidationSplit(word2vecsTrainedOnGbt_relevance, relevanceTrainingDf, relevanceValidationgDf, "relevanceLabel"))

    println("word2vecsWithLiwcTrainedOnGbt")
    println(crossvalidatorSentiment(word2vecsWithLiwcTrainedOnGbt_sentiment))
    println(crossvalidatorRelevance(word2vecsWithLiwcTrainedOnGbt_relevance))
    println(trainValidationSplit(word2vecsWithLiwcTrainedOnGbt_sentiment, sentimentTrainingDf, sentimentValidationDf, "sentiment"))
    println(trainValidationSplit(word2vecsWithLiwcTrainedOnGbt_relevance, relevanceTrainingDf, relevanceValidationgDf, "relevanceLabel"))

    println("word2vecsWithSentimentLexiconTrainedOnGbt")
    println(crossvalidatorSentiment(word2vecsWithSentimentLexiconTrainedOnGbt_sentiment))
    println(crossvalidatorRelevance(word2vecsWithSentimentLexiconTrainedOnGbt_relevance))
    println(trainValidationSplit(word2vecsWithSentimentLexiconTrainedOnGbt_sentiment, sentimentTrainingDf, sentimentValidationDf, "sentiment"))
    println(trainValidationSplit(word2vecsWithSentimentLexiconTrainedOnGbt_relevance, relevanceTrainingDf, relevanceValidationgDf, "relevanceLabel"))

    println("word2vecsWithSentimentLexiconAndLiwcTrainedOnGbt")
    println(crossvalidatorSentiment(word2vecsWithSentimentLexiconAndLiwcTrainedOnGbt_sentiment))
    println(crossvalidatorRelevance(word2vecsWithSentimentLexiconAndLiwcTrainedOnGbt_relevance))
    println(trainValidationSplit(word2vecsWithSentimentLexiconAndLiwcTrainedOnGbt_sentiment, sentimentTrainingDf, sentimentValidationDf, "sentiment"))
    println(trainValidationSplit(word2vecsWithSentimentLexiconAndLiwcTrainedOnGbt_relevance, relevanceTrainingDf, relevanceValidationgDf, "relevanceLabel"))

    println("top1000FromIdfsAndLiwcTrainedOnMlp")
    println(crossvalidatorSentiment(top1000FromIdfsAndLiwcTrainedOnMlp_sentiment))
    println(crossvalidatorRelevance(top1000FromIdfsAndLiwcTrainedOnMlp_relevance))
    println(trainValidationSplit(top1000FromIdfsAndLiwcTrainedOnMlp_sentiment, sentimentTrainingDf, sentimentValidationDf, "sentiment"))
    println(trainValidationSplit(top1000FromIdfsAndLiwcTrainedOnMlp_relevance, relevanceTrainingDf, relevanceValidationgDf, "relevanceLabel"))

    println("top1000FromIdfsTrainedOnMlp")
    println(crossvalidatorSentiment(top1000FromIdfsTrainedOnMlp_sentiment))
    println(crossvalidatorRelevance(top1000FromIdfsTrainedOnMlp_relevance))
    println(trainValidationSplit(top1000FromIdfsTrainedOnMlp_sentiment, sentimentTrainingDf, sentimentValidationDf, "sentiment"))
    println(trainValidationSplit(top1000FromIdfsTrainedOnMlp_relevance, relevanceTrainingDf, relevanceValidationgDf, "relevanceLabel"))

    println("top1000FromIdfsAndLanguageToolTrainedOnMlp")
    println(crossvalidatorSentiment(top1000FromIdfsAndLanguageToolTrainedOnMlp_sentiment))
    println(crossvalidatorRelevance(top1000FromIdfsAndLanguageToolTrainedOnMlp_relevance))
    println(trainValidationSplit(top1000FromIdfsAndLanguageToolTrainedOnMlp_sentiment, sentimentTrainingDf, sentimentValidationDf, "sentiment"))
    println(trainValidationSplit(top1000FromIdfsAndLanguageToolTrainedOnMlp_relevance, relevanceTrainingDf, relevanceValidationgDf, "relevanceLabel"))

    println("word2vecsTrainedOnMlp")
    println(crossvalidatorSentiment(word2vecsTrainedOnMlp_sentiment))
    println(crossvalidatorRelevance(word2vecsTrainedOnMlp_relevance))
    println(trainValidationSplit(word2vecsTrainedOnMlp_sentiment, sentimentTrainingDf, sentimentValidationDf, "sentiment"))
    println(trainValidationSplit(word2vecsTrainedOnMlp_relevance, relevanceTrainingDf, relevanceValidationgDf, "relevanceLabel"))

    println("word2vecsWithLiwcTrainedOnMlp")
    println(crossvalidatorSentiment(word2vecsWithLiwcTrainedOnMlp_sentiment))
    println(crossvalidatorRelevance(word2vecsWithLiwcTrainedOnMlp_relevance))
    println(trainValidationSplit(word2vecsWithLiwcTrainedOnMlp_sentiment, sentimentTrainingDf, sentimentValidationDf, "sentiment"))
    println(trainValidationSplit(word2vecsWithLiwcTrainedOnMlp_relevance, relevanceTrainingDf, relevanceValidationgDf, "relevanceLabel"))

    println("word2vecsWithSentimentLexiconTrainedOnMlp")
    println(crossvalidatorSentiment(word2vecsWithSentimentLexiconTrainedOnMlp_sentiment))
    println(crossvalidatorRelevance(word2vecsWithSentimentLexiconTrainedOnMlp_relevance))
    println(trainValidationSplit(word2vecsWithSentimentLexiconTrainedOnMlp_sentiment, sentimentTrainingDf, sentimentValidationDf, "sentiment"))
    println(trainValidationSplit(word2vecsWithSentimentLexiconTrainedOnMlp_relevance, relevanceTrainingDf, relevanceValidationgDf, "relevanceLabel"))

    println("word2vecsWithSentimentLexiconAndLiwcTrainedOnMlp")
    println(crossvalidatorSentiment(word2vecsWithSentimentLexiconAndLiwcTrainedOnMlp_sentiment))
    println(crossvalidatorRelevance(word2vecsWithSentimentLexiconAndLiwcTrainedOnMlp_relevance))
    println(trainValidationSplit(word2vecsWithSentimentLexiconAndLiwcTrainedOnMlp_sentiment, sentimentTrainingDf, sentimentValidationDf, "sentiment"))
    println(trainValidationSplit(word2vecsWithSentimentLexiconAndLiwcTrainedOnMlp_relevance, relevanceTrainingDf, relevanceValidationgDf, "relevanceLabel"))


    def printModel(name: String, model: SdCrossValidatorModel): Unit = {
      println(name + ": " + model.average + "\n\taverage: " + model.allMetrics.mkString(", "))
    }


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