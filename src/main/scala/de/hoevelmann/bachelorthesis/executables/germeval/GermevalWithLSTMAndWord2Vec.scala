package de.hoevelmann.bachelorthesis.executables.germeval

import com.intel.analytics.bigdl.utils.Engine
import de.hoevelmann.bachelorthesis.modelling.classifiers.LSTMClassifier
import de.hoevelmann.bachelorthesis.modelling.entities.ConfusionMatrix
import de.hoevelmann.bachelorthesis.modelling.transformers.{NonAveragedWord2Vec, RowPruner, VectorSequencer}
import org.apache.log4j.Logger
import org.apache.spark.api.java.function.FilterFunction
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.{Pipeline, linalg}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.io.Source
import scala.util.Random

object GermevalWithLSTMAndWord2Vec {

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
  def execute(spark: SparkSession, ngramSize: Int = 1, numFeatures: Int = 1000, intermediateLayers: Array[Int] = Array(300), useLIWCFeatures: Boolean = true, stem: Boolean = true, numIterations: Int = 100, maxDistance: Double): Unit = {
    //initializing bigdl
    Engine.init
    import spark.implicits._
    val trainingReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_train.csv")
    val testReviews: Seq[Review] = loadReviews("germeval/corpus/liwc_dev.csv")

    // Logger.getLogger("org").setLevel(Level.OFF)
    //Logger.getLogger("akka").setLevel(Level.OFF)


    val termzzz: Seq[String] = Source.fromFile("reviews.csv").getLines().toSeq.par.map(_.split(";")).map(a => a(3) + " " + a(4)).map(x => x.toLowerCase.split(" ").map(_
      .replace("#", "")
      .replace(",", "")
      .replace(".", "")
      .replace(":", "")
      .replace("!", "")
      .replace("?", "")
      .replace("(", "")
      .replace(")", "")
    ).toSeq).seq.flatten

    val w2vTraingDf = Seq(termzzz).toDF("words")

    val sentimentTrainingDataFrame: DataFrame = (for (review <- Random.shuffle(trainingReviews.seq)) yield {
      (review.sentiClass, review.terms, Vectors.dense(review.liwcFeatures))
    })
      //.take(100)
      .toDF("sentiment", "input", "liwcFeatures")
    val relevanceTrainingDataFrame: DataFrame = (for (review <- Random.shuffle(trainingReviews.seq)) yield {
      (if (review.relevant) 2 else 1, review.terms, Vectors.dense(review.liwcFeatures))
    })
      //.take(100)
      .toDF("relevanceLabel", "input", "liwcFeatures")

    val sentimentValidationDataFrame: DataFrame = (for (review <- testReviews) yield {
      (review.sentiClass, review.terms, Vectors.dense(review.liwcFeatures))
    })
      //.take(100)
      .seq.toDF("sentiment", "input", "liwcFeatures")

    val relevanceValidationDataFrame: DataFrame = (for (review <- testReviews) yield {
      (if (review.relevant) 1 else 0, review.terms, Vectors.dense(review.liwcFeatures))
    })
      //.take(100)
      .seq.toDF("relevanceLabel", "input", "liwcFeatures")


    /*
     * define the pipeline steps
     */

    // val wordSubstitutionSentiment: WordSubstitutor = new WordSubstitutor().setWordVectors(wordVectors).setInputCol("input").setOutputCol("substitutes").setMatchingTerms(wordsInSentimentTrainingSet).setMaxDistance(maxDistance)
    // val wordSubstitutionRelevance: WordSubstitutor = new WordSubstitutor().setWordVectors(wordVectors).setInputCol("input").setOutputCol("substitutes").setMatchingTerms(wordsInRelevanceTrainingSet).setMaxDistance(maxDistance)
    //val germanStemmer: GermanStemmer = new GermanStemmer().setInputCol("substitutes").setOutputCol("stems")
    //val tokenizer: Tokenizer = new Tokenizer().setInputCol("input").setOutputCol("tokens")
    val word2VecModel = new Word2Vec().setInputCol("words").fit(w2vTraingDf)
    val word2Vectors: NonAveragedWord2Vec = new NonAveragedWord2Vec().setInputCol("input").setOutputCol("vectors").setWord2VecModel(word2VecModel)
    val vectorSequencer: VectorSequencer = new VectorSequencer().setInputCol("vectors").setOutputCol("sequences").setEmbeddingSize(100).setSequenceLength(15)



    val filterFunction = new FilterFunction[Row] {
      override def call(value: Row): Boolean = value.getAs[Seq[Vector]]("vectors").nonEmpty
    }

    val processingPipeline = new Pipeline().setStages(Array(word2Vectors, new RowPruner(filterFunction), vectorSequencer)).fit(sentimentTrainingDataFrame)
    println("fitting")


    val sentimentTrainingSet = {
      val xxx = processingPipeline.transform(sentimentTrainingDataFrame)
      xxx.printSchema()
      xxx.collect()
        .map(row => (row.getAs[Double]("sentiment"), row.getAs[org.apache.spark.ml.linalg.Vector]("sequences")))
    }
    val relevanceTrainingSet = {
      val xxx = processingPipeline.transform(relevanceTrainingDataFrame)
      xxx.printSchema()
      xxx.collect()
        .map(row => (row.getAs[Int]("relevanceLabel").toDouble, row.getAs[org.apache.spark.ml.linalg.Vector]("sequences")))
    }

    val sentimentValidationSet = {
      val xxx = processingPipeline.transform(sentimentValidationDataFrame)
      xxx.printSchema()
      xxx.collect()
        .map(row => (row.getAs[Double]("sentiment"), row.getAs[org.apache.spark.ml.linalg.Vector]("sequences")))
    }
    val relevanceValidationSet = {
      val xxx = processingPipeline.transform(relevanceValidationDataFrame)
      xxx.printSchema()
      xxx.collect()
        .map(row => (row.getAs[Int]("relevanceLabel").toDouble, row.getAs[org.apache.spark.ml.linalg.Vector]("sequences")))
    }

    println("training classifiers")

    val sentimentModel = new LSTMClassifier(vectorDimensionality = 100, sequenceLength = 15, numClasses = 3).train(sentimentTrainingSet, sentimentValidationSet)
    val relevanceModel = new LSTMClassifier(vectorDimensionality = 100, sequenceLength = 15, numClasses = 2).train(relevanceTrainingSet, relevanceValidationSet)


    println("loading and transforming test reviews")


    /*val sentimentEvaluator = new MulticlassClassificationEvaluator().setMetricName("f1").setLabelCol("sentiment").set
    val relevanceEvaluator = new MulticlassClassificationEvaluator().setMetricName("f1").setLabelCol("relevanceLabel")

    println("(sentiment) f1 multilayer perceptron: " + sentimentEvaluator.evaluate(sentimentValidationSet))
    println("(relevance) f1 multilayer perceptron: " + relevanceEvaluator.evaluate(relevanceValidationSetMLP))*/
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
          sentiClass = if (x(3) == "negative") 1.0 else if (x(3) == "neutral") 2.0 else 3.0,
          liwcFeatures = x.slice(5, x.length).map(_.replace(",", ".")).map(_.toDouble)))
      .seq
    result
  }

  /*
   * Auxiliary classes
   */

  class Review(val terms: Seq[String], val relevant: Boolean, val sentiClass: Double, val liwcFeatures: Array[Double])

  class TermWithClass(val term: String, val sentimentClass: Int, val occurrences: Int)

}