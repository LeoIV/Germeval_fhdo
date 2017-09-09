package de.hoevelmann.bachelorthesis.executables.germeval

import de.fhdo.germeval.modelling.transformers.{CsvFileWriter, FastTextSentenceVector, KMeansTransformer}
import de.hoevelmann.bachelorthesis.modelling.transformers._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.io.Source
import scala.util.{Random, Try}

object PreprocessedCorpusToCsv {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]").getOrCreate()
    import spark.implicits._
    val trainingReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_train.csv")
    val testReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_dev.csv")
    val validationReviews1: Seq[Review] = loadReviews("germeval/corpus/liwc_test_ts1.csv")
    val validationReviews2: Seq[Review] = loadReviews("germeval/corpus/liwc_test_ts2.csv")

    println("len train: " + trainingReviews.length)
    println("len test: " + testReviews.length)
    println("len val: " + validationReviews1.length)
    println("len val: " + validationReviews2.length)

    spark.sparkContext.setLogLevel("ERROR")

    println("creating dataframes")

    val trainingDf: DataFrame = (for (review <- Random.shuffle(trainingReviews)) yield {
      (if (review.relevant) 1 else 0, review.sentiClass, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("relevance", "sentiment", "input", "liwcFeatures").cache()
    val testDf: DataFrame = (for (review <- Random.shuffle(testReviews)) yield {
      (if (review.relevant) 1 else 0, review.sentiClass, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("relevance", "sentiment", "input", "liwcFeatures").cache()
    val validationDfTs1 = (for (review <- validationReviews1) yield {
      (review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("input", "liwcFeatures").cache()
    val validationDfTs2 = (for (review <- validationReviews2) yield {
      (review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("input", "liwcFeatures").cache()

    println("creating preprocessors")

    /*
     * define the pipeline steps
     */

    val word2vecTransformer = new FastTextSentenceVector().setInputCol("input").setOutputCol("word2vecs")
    val kmeans = new KMeansTransformer().setInputCol("word2vecs").setOutputCol("kmeans")
    val firstAssembler: VectorAssembler = new VectorAssembler().setInputCols(Array("word2vecs", "liwcFeatures","kmeans")).setOutputCol("firstAssembly")


    println("fitting models")

    val sentimentPipeline = new Pipeline().setStages(Array(word2vecTransformer, kmeans, firstAssembler))

    println("fitting sentiment pipeline")
    val sentimentModel = sentimentPipeline.fit(trainingDf.union(testDf))
    println("fitted sentiment pipeline")

    val relevancePipeline = new Pipeline().setStages(Array(word2vecTransformer, kmeans, firstAssembler))

    println("fitting relevance pipeline")
    val relevanceModel = relevancePipeline.fit(trainingDf.union(testDf))
    println("fitted relevance pipeline")

    println("transforming all dataframes")

    //val sentimentTrainingDfTransformed = sentimentModel.transform(trainingDf)
    //val sentimentTestDfTransformed = sentimentModel.transform(testDf)
    val sentimentValidationDfTransformed1 = sentimentModel.transform(validationDfTs1)
    val sentimentValidationDfTransformed2 = sentimentModel.transform(validationDfTs2)

    //val relevanceTrainingDfTransformed = relevanceModel.transform(trainingDf)
    //val relevanceTestDfTransformed = relevanceModel.transform(testDf)
    val relevanceValidationDfTransformed1 = relevanceModel.transform(validationDfTs1)
    val relevanceValidationDfTransformed2 = relevanceModel.transform(validationDfTs2)

    println("transformed all dataframes")

    val fileEnding = "293"

    val fw1 = new CsvFileWriter().setFeaturesColumn("firstAssembly").setLabelColumn("sentiment").setOutputFilepath("C:/Users/hoevelmann/Downloads/sentiment_training_" + fileEnding + ".csv")
    val fw2 = new CsvFileWriter().setFeaturesColumn("firstAssembly").setLabelColumn("sentiment").setOutputFilepath("C:/Users/hoevelmann/Downloads/sentiment_test_" + fileEnding + ".csv")
    val fw3 = new CsvFileWriter().setFeaturesColumn("firstAssembly").setOutputFilepath("C:/Users/hoevelmann/Downloads/sentiment_unlabeled_ts1_" + fileEnding + ".csv")
    val fw4 = new CsvFileWriter().setFeaturesColumn("firstAssembly").setOutputFilepath("C:/Users/hoevelmann/Downloads/sentiment_unlabeled_ts2_" + fileEnding + ".csv")

    val fw5 = new CsvFileWriter().setFeaturesColumn("firstAssembly").setLabelColumn("relevance").setOutputFilepath("C:/Users/hoevelmann/Downloads/relevance_training_" + fileEnding + ".csv")
    val fw6 = new CsvFileWriter().setFeaturesColumn("firstAssembly").setLabelColumn("relevance").setOutputFilepath("C:/Users/hoevelmann/Downloads/relevance_test_" + fileEnding + ".csv")
    val fw7 = new CsvFileWriter().setFeaturesColumn("firstAssembly").setOutputFilepath("C:/Users/hoevelmann/Downloads/relevance_unlabeled_ts1_" + fileEnding + ".csv")
    val fw8 = new CsvFileWriter().setFeaturesColumn("firstAssembly").setOutputFilepath("C:/Users/hoevelmann/Downloads/relevance_unlabeled_ts2_" + fileEnding + ".csv")

    println("start writing")

    // fw1.transform(sentimentTrainingDfTransformed)
    //fw2.transform(sentimentTestDfTransformed)
    fw3.transform(sentimentValidationDfTransformed1)
    fw4.transform(sentimentValidationDfTransformed2)
    //fw4.transform(relevanceTrainingDfTransformed)
    //fw5.transform(relevanceTestDfTransformed)
    fw7.transform(relevanceValidationDfTransformed1)
    fw8.transform(relevanceValidationDfTransformed2)

    println("writing ended")

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
          liwcFeatures = x.slice(4, x.length).map(_.replace(",", ".")).filter(x => Try(x.toDouble).isSuccess).map(_.toDouble)))
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