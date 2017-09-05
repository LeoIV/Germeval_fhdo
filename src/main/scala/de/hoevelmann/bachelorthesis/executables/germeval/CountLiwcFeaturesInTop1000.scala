package de.hoevelmann.bachelorthesis.executables.germeval

import java.io.FileWriter

import de.hoevelmann.bachelorthesis.executables.germeval.FinalSubmission.loadReviews
import de.hoevelmann.bachelorthesis.executables.germeval.FinalSubmissionCV.{Review, loadReviews}
import de.hoevelmann.bachelorthesis.modelling.transformers.{GermanStemmer, LanguageToolTransformer}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.io.Source
import scala.util.Random

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 14.08.2017.
  */
object CountLiwcFeaturesInTop1000 {

  def main(args: Array[String]): Unit = {

    val spark: SparkSession = SparkSession.builder().master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    Logger.getLogger("TaskSetManager").setLevel(Level.ERROR)

    import spark.implicits._
    val trainingReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_train.csv")
    val validationReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_dev.csv")


    val sentimentValidationDf: DataFrame = (for (review <- Random.shuffle(validationReviews)) yield {
      (review.sentiClass, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("sentiment", "input", "liwcFeatures").cache()
    val sentimentTrainingDf: DataFrame = (for (review <- Random.shuffle(trainingReviews)) yield {
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

    val germanStemmer: GermanStemmer = new GermanStemmer().setInputCol("input").setOutputCol("stems")

    val hashingTf = new HashingTF().setInputCol("stems").setOutputCol("tfs").setNumFeatures(16384)

    val idfTr = new IDF().setInputCol("tfs").setOutputCol("idfs")

    val bowAssemblerWithLiwc = new VectorAssembler().setInputCols(Array("idfs", "liwcFeatures")).setOutputCol("mlpAssembly")

    val sentimentChi2selector = new ChiSqSelector().setFeaturesCol("mlpAssembly").setLabelCol("sentiment").setOutputCol("topFeatures").setNumTopFeatures(16477)

    val relevanceChi2selector = new ChiSqSelector().setFeaturesCol("mlpAssembly").setLabelCol("relevanceLabel").setOutputCol("topFeatures").setNumTopFeatures(16477)

    val prePipe = new Pipeline().setStages(Array(germanStemmer, hashingTf, idfTr, bowAssemblerWithLiwc)).fit(mergedSentimentDf)

    val preSentimentDf = prePipe.transform(mergedSentimentDf)
    val preRelevanceDf = prePipe.transform(mergedRelevanceDf)

    val senChi2Mdl: ChiSqSelectorModel = sentimentChi2selector.fit(preSentimentDf)
    val relChi2Mdl: ChiSqSelectorModel = relevanceChi2selector.fit(preRelevanceDf)

    println("Sentiment #LIWC features: "+senChi2Mdl.selectedFeatures.count(_ > 16383))
    println("Sentiment LIWC features indices: "+senChi2Mdl.selectedFeatures.filter(_ > 16383).map(_ - 16383).mkString(" | "))
    println("Relevance #LIWC features: "+relChi2Mdl.selectedFeatures.count(_ > 16383))
    println("Relevance LIWC features indices: "+relChi2Mdl.selectedFeatures.filter(_ > 16383).map(_ - 16383).mkString(" | "))

  }

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

}
