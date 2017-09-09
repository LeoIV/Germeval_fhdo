package de.fhdo.germeval.executables.germeval

import java.io.FileWriter

import de.fhdo.germeval.modelling.transformers.{FastTextClassifier, FastTextSentenceVector, GermanStemmer, SentimentLexicon}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassifier, OneVsRest}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.io.Source
import scala.util.Random

object FinalSubmission {

  /**
    * Execute the Germeval task
    *
    */
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[*]").getOrCreate()

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    import spark.implicits._
    val trainingReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_train.csv")
    val validationReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_dev.csv")

    val testReviewsTs1 = loadReviews("germeval/corpus/liwc_test_ts1.csv")
    val testReviewsTs2 = loadReviews("germeval/corpus/liwc_test_ts2.csv")


    println("creating dataframes")

    val sentimentTrainingDataFrame: DataFrame = (for (review <- Random.shuffle(trainingReviews ++ validationReviews)) yield {
      (review.sentiClass, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("sentiment", "input", "liwcFeatures").cache()
    val relevanceTrainingDataFrame: DataFrame = (for (review <- Random.shuffle(trainingReviews ++ validationReviews)) yield {
      (if (review.relevant) 1 else 0, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("relevanceLabel", "input", "liwcFeatures").cache()

    val testDf1 = (for (review <- testReviewsTs1) yield {
      (review.terms, Vectors.dense(review.liwcFeatures), review.id, review.termsUnprocessed)
    }).toDF("input", "liwcFeatures", "id", "termsUnprocessed")

    val testDf2 = (for (review <- testReviewsTs2) yield {
      (review.terms, Vectors.dense(review.liwcFeatures), review.id, review.termsUnprocessed)
    }).toDF("input", "liwcFeatures", "id", "termsUnprocessed")


    println("creating preprocessors")

    val word2vecTransformer = new FastTextSentenceVector().setInputCol("input").setOutputCol("word2vecs")
    val germanStemmer: GermanStemmer = new GermanStemmer().setInputCol("input").setOutputCol("stems")
    val hashingTf = new HashingTF().setInputCol("stems").setOutputCol("tfs").setNumFeatures(16384)
    val idfTr = new IDF().setInputCol("tfs").setOutputCol("idfs")
    val bowAssembler = new VectorAssembler().setInputCols(Array("idfs", "liwcFeatures")).setOutputCol("mlpAssembly")
    val sentimentChi2selector = new ChiSqSelector().setFeaturesCol("mlpAssembly").setLabelCol("sentiment").setOutputCol("topFeatures").setNumTopFeatures(1000)
    val relevanceChi2selector = new ChiSqSelector().setFeaturesCol("mlpAssembly").setLabelCol("relevanceLabel").setOutputCol("topFeatures").setNumTopFeatures(1000)
    val sentimentLexicon = new SentimentLexicon().setInputCol("input").setOutputCol("senLexFeatures")
    //  val languageTool: LanguageToolTransformer = new LanguageToolTransformer().setInputCol("input").setOutputCol("languageToolFeatures")
    val assembler = new VectorAssembler().setInputCols(Array("word2vecs", "liwcFeatures", "senLexFeatures")).setOutputCol("assembledFeatures")

    val fastTextSentimentPredictor = new FastTextClassifier().setLabelCol("sentiment").setFeaturesCol("stems")
    val fastTextRelevancePredictor = new FastTextClassifier().setFeaturesCol("stems").setLabelCol("relevanceLabel")

    val gbtW2vSentimentPredictor = new OneVsRest().setClassifier(new GBTClassifier().setMaxIter(30).setMaxDepth(10)).setLabelCol("sentiment").setFeaturesCol("assembledFeatures")
    val gbtW2vRelevancePredictor = new GBTClassifier().setMaxIter(100).setMaxDepth(10).setLabelCol("relevanceLabel").setFeaturesCol("assembledFeatures")

    val gbtBowSentimentPredictor = new OneVsRest().setClassifier(new GBTClassifier().setMaxIter(30).setMaxDepth(10)).setLabelCol("sentiment").setFeaturesCol("topFeatures")
    val gbtBowRelevancePredictor = new GBTClassifier().setMaxIter(100).setMaxDepth(10).setLabelCol("relevanceLabel").setFeaturesCol("topFeatures")

    val gbtBowSentimentPipeline = new Pipeline().setStages(Array(germanStemmer, hashingTf, idfTr, bowAssembler, sentimentChi2selector, gbtBowSentimentPredictor))
    val gbtBowRelevancePipeline = new Pipeline().setStages(Array(germanStemmer, hashingTf, idfTr, bowAssembler, relevanceChi2selector, gbtBowRelevancePredictor))

    val gbtW2vSentimentPipeline = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, assembler, gbtW2vSentimentPredictor))
    val gbtW2vRelevancePipeline = new Pipeline().setStages(Array(word2vecTransformer, sentimentLexicon, assembler, gbtW2vRelevancePredictor))

    val fasttextSentimentPipeline = new Pipeline().setStages(Array(germanStemmer, fastTextSentimentPredictor))
    val fasttextRelevancePipeline = new Pipeline().setStages(Array(germanStemmer, fastTextRelevancePredictor))

    def predictionDfToMap(df: DataFrame): Map[String, (String, Double)] = df.rdd.map(row => (row.getAs[String]("id"), row.getAs[String]("termsUnprocessed"), row.getAs[Double]("prediction"))).collect.flatMap(rev => Map(rev._1 -> (rev._2, rev._3))).toMap

    /*val fasttextSentimentModel = fasttextSentimentPipeline.fit(sentimentTrainingDataFrame)
    val fasttextRelevanceModel = fasttextRelevancePipeline.fit(relevanceTrainingDataFrame)

    val fasttextTestDfSen1: DataFrame = fasttextSentimentModel.transform(testDf1)
    val fasttextTestDfSen2: DataFrame = fasttextSentimentModel.transform(testDf2)
    val fasttextTestDfRel1: DataFrame = fasttextRelevanceModel.transform(testDf1)
    val fasttextTestDfRel2: DataFrame = fasttextRelevanceModel.transform(testDf2)

    val fasttextSenTs1: Map[String, (String, Double)] = predictionDfToMap(fasttextTestDfSen1)
    val fasttextSenTs2: Map[String, (String, Double)] = predictionDfToMap(fasttextTestDfSen2)
    val fasttextRelTs1: Map[String, (String, Double)] = predictionDfToMap(fasttextTestDfRel1)
    val fasttextRelTs2: Map[String, (String, Double)] = predictionDfToMap(fasttextTestDfRel2)

    val fasttextSenFwTs1 = new FileWriter("C:/Users/hoevelmann/Downloads/fhdo.task3.ts1.fasttext.tsv")
    val fasttextSenFwTs2 = new FileWriter("C:/Users/hoevelmann/Downloads/fhdo.task3.ts2.fasttext.tsv")
    val fasttextRelFwTs1 = new FileWriter("C:/Users/hoevelmann/Downloads/fhdo.task2.ts1.fasttext.tsv")
    val fasttextRelFwTs2 = new FileWriter("C:/Users/hoevelmann/Downloads/fhdo.task2.ts2.fasttext.tsv")

    fasttextSenTs1.foreach(s => {
      fasttextSenFwTs1.write(s._1 + "\t" + s._2._1 + "\t" + "unknown\t" + (if (math.round(s._2._2) == 0) "negative" else if (math.round(s._2._2) == 1) "neutral" else "positive") + "\n")
    })
    fasttextSenTs2.foreach(s => {
      fasttextSenFwTs2.write(s._1 + "\t" + s._2._1 + "\t" + "unknown\t" + (if (math.round(s._2._2) == 0) "negative" else if (math.round(s._2._2) == 1) "neutral" else "positive") + "\n")
    })
    fasttextRelTs1.foreach(s => {
      fasttextRelFwTs1.write(s._1 + "\t" + s._2._1 + "\t" + (if (math.round(s._2._2) == 0) "false" else "true") + "\tunknown\n")
    })
    fasttextRelTs2.foreach(s => {
      fasttextRelFwTs2.write(s._1 + "\t" + s._2._1 + "\t" + (if (math.round(s._2._2) == 0) "false" else "true") + "\tunknown\n")
    })
    fasttextSenFwTs1.close()
    fasttextSenFwTs2.close()
    fasttextRelFwTs1.close()
    fasttextRelFwTs2.close()

    println("done fasttext")


    val gbtBowSentimentModel = gbtBowSentimentPipeline.fit(sentimentTrainingDataFrame)
    val gbtBowRelevanceModel = gbtBowRelevancePipeline.fit(relevanceTrainingDataFrame)

    val gbtBowTestDfSen1: DataFrame = gbtBowSentimentModel.transform(testDf1)
    val gbtBowTestDfSen2: DataFrame = gbtBowSentimentModel.transform(testDf2)
    val gbtBowTestDfRel1: DataFrame = gbtBowRelevanceModel.transform(testDf1)
    val gbtBowTestDfRel2: DataFrame = gbtBowRelevanceModel.transform(testDf2)

    val gbtBowSenTs1: Map[String, (String, Double)] = predictionDfToMap(gbtBowTestDfSen1)
    val gbtBowSenTs2: Map[String, (String, Double)] = predictionDfToMap(gbtBowTestDfSen2)
    val gbtBowRelTs1: Map[String, (String, Double)] = predictionDfToMap(gbtBowTestDfRel1)
    val gbtBowRelTs2: Map[String, (String, Double)] = predictionDfToMap(gbtBowTestDfRel2)

    val gbtBowSenFwTs1 = new FileWriter("C:/Users/hoevelmann/Downloads/fhdo.task3.ts1.gbt_bow.tsv")
    val gbtBowSenFwTs2 = new FileWriter("C:/Users/hoevelmann/Downloads/fhdo.task3.ts2.gbt_bow.tsv")
    val gbtBowRelFwTs1 = new FileWriter("C:/Users/hoevelmann/Downloads/fhdo.task2.ts1.gbt_bow.tsv")
    val gbtBowRelFwTs2 = new FileWriter("C:/Users/hoevelmann/Downloads/fhdo.task2.ts2.gbt_bow.tsv")

    gbtBowSenTs1.foreach(s => {
      gbtBowSenFwTs1.write(s._1 + "\t" + s._2._1 + "\t" + "unknown\t" + (if (math.round(s._2._2) == 0) "negative" else if (math.round(s._2._2) == 1) "neutral" else "positive") + "\n")
    })
    gbtBowRelTs1.foreach(s => {
      gbtBowRelFwTs1.write(s._1 + "\t" + s._2._1 + "\t" + (if (math.round(s._2._2) == 0) "false" else "true") + "\tunknown\n")
    })
    gbtBowRelTs2.foreach(s => {
      gbtBowRelFwTs2.write(s._1 + "\t" + s._2._1 + "\t" + (if (math.round(s._2._2) == 0) "false" else "true") + "\tunknown\n")
    })
    gbtBowSenTs2.foreach(s => {
      gbtBowSenFwTs2.write(s._1 + "\t" + s._2._1 + "\t" + "unknown\t" + (if (math.round(s._2._2) == 0) "negative" else if (math.round(s._2._2) == 1) "neutral" else "positive") + "\n")
    })
    gbtBowSenFwTs1.close()
    gbtBowRelFwTs1.close()
    gbtBowRelFwTs2.close()
    gbtBowSenFwTs2.close()

    println("done bow")*/

    val gbtW2vSentimentModel = gbtW2vSentimentPipeline.fit(sentimentTrainingDataFrame)
    println("fitted sentiment")
    val gbtW2vRelevanceModel = gbtW2vRelevancePipeline.fit(relevanceTrainingDataFrame)
    println("fitted relevance")
    val gbtW2vTestDfSen1: DataFrame = gbtW2vSentimentModel.transform(testDf1)
    println("transformed 1")
    val gbtW2vTestDfSen2: DataFrame = gbtW2vSentimentModel.transform(testDf2)
    println("transformed 2")
    val gbtW2vTestDfRel1: DataFrame = gbtW2vRelevanceModel.transform(testDf1)
    println("transformed 3")
    val gbtW2vTestDfRel2: DataFrame = gbtW2vRelevanceModel.transform(testDf2)
    println("transformed 4")

    val gbtW2vSenTs1: Map[String, (String, Double)] = predictionDfToMap(gbtW2vTestDfSen1)
    val gbtW2vSenTs2: Map[String, (String, Double)] = predictionDfToMap(gbtW2vTestDfSen2)
    val gbtW2vRelTs1: Map[String, (String, Double)] = predictionDfToMap(gbtW2vTestDfRel1)
    val gbtW2vRelTs2: Map[String, (String, Double)] = predictionDfToMap(gbtW2vTestDfRel2)

    val gbtW2vSenFwTs1 = new FileWriter("fhdo.task3.ts1.gbt_w2v.tsv")
    val gbtW2vSenFwTs2 = new FileWriter("fhdo.task3.ts2.gbt_w2v.tsv")
    val gbtW2vRelFwTs1 = new FileWriter("fhdo.task2.ts1.gbt_w2v.tsv")
    val gbtW2vRelFwTs2 = new FileWriter("fhdo.task2.ts2.gbt_w2v.tsv")

    gbtW2vSenTs1.foreach(s => {
      gbtW2vSenFwTs1.write(s._1 + "\t" + s._2._1 + "\t" + "unknown\t" + (if (math.round(s._2._2) == 0) "negative" else if (math.round(s._2._2) == 1) "neutral" else "positive") + "\n")
    })
    gbtW2vSenTs2.foreach(s => {
      gbtW2vSenFwTs2.write(s._1 + "\t" + s._2._1 + "\t" + "unknown\t" + (if (math.round(s._2._2) == 0) "negative" else if (math.round(s._2._2) == 1) "neutral" else "positive") + "\n")
    })
    gbtW2vRelTs1.foreach(s => {
      gbtW2vRelFwTs1.write(s._1 + "\t" + s._2._1 + "\t" + (if (math.round(s._2._2) == 0) "false" else "true") + "\tunknown\n")
    })
    gbtW2vRelTs2.foreach(s => {
      gbtW2vRelFwTs2.write(s._1 + "\t" + s._2._1 + "\t" + (if (math.round(s._2._2) == 0) "false" else "true") + "\tunknown\n")
    })
    gbtW2vSenFwTs1.close();
    gbtW2vSenFwTs2.close();
    gbtW2vRelFwTs1.close();
    gbtW2vRelFwTs2.close()
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
          id = x(0),
          termsUnprocessed = x(1),
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