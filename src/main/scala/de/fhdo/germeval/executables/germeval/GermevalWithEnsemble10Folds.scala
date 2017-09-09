package de.hoevelmann.bachelorthesis.executables.germeval

/*
To run the ensemble models, a "probability col" has to be added to the Spark MLP. This will come in future releases of Spark.
A custom version of Spark has been built to run these models, that isn't part of this repository. Since this class produces
build errors, it is commented out. It should run with a newer version of Spark.
 */

import de.hoevelmann.bachelorthesis.modelling.transformers._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{Pipeline, _}
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types.{LongType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.slf4j.{Logger, LoggerFactory}

import scala.io.Source
import scala.util.Random
/*
object GermevalWithEnsemble10Folds {

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

    spark.sparkContext.setLogLevel("ERROR")

    println("creating dataframes")

    val sentimentTrainingDataFrame: Array[Dataset[Row]] = {
      val df = (for (review <- Random.shuffle(trainingReviews)) yield {
        (review.sentiClass, review.terms, Vectors.dense(review.liwcFeatures))
      }).toDF("sentiment", "input", "liwcFeatures")
      df.randomSplit((for (_ <- 0 until 10) yield 0.1).toArray)
    }

    val sentimentTestDataFrame = (for (review <- Random.shuffle(testReviews)) yield {
      (review.sentiClass, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("sentiment", "input", "liwcFeatures")

    val relevanceTrainingDataFrame: Array[Dataset[Row]] = {
      val df = (for (review <- Random.shuffle(trainingReviews)) yield {
        (if (review.relevant) 1 else 0, review.terms, Vectors.dense(review.liwcFeatures))
      }).toDF("relevanceLabel", "input", "liwcFeatures")
      df.randomSplit((for (_ <- 0 until 10) yield 0.1).toArray)
    }

    val relevanceTestDataFrame = (for (review <- Random.shuffle(testReviews)) yield {
      (if (review.relevant) 1 else 0, review.terms, Vectors.dense(review.liwcFeatures))
    }).toDF("relevanceLabel", "input", "liwcFeatures")


    println("creating preprocessors")

    /*
     * define the pipeline steps
     */

    val languageTool: LanguageToolTransformer = new LanguageToolTransformer().setInputCol("input").setOutputCol("languageToolFeatures")
    val germanStemmer: GermanStemmer = new GermanStemmer().setInputCol("input").setOutputCol("stems")
    val nGramStep: NGram = new NGram().setInputCol("stems").setOutputCol("ngrams").setN(1)

    val hashingTermFrequencies: HashingTF = new HashingTF().setInputCol("ngrams").setOutputCol("tfs").setNumFeatures(16384)
    val idf: IDF = new IDF().setInputCol("tfs").setOutputCol("idfs")
    val bowVectorAssembler: VectorAssembler = new VectorAssembler().setInputCols(Array("idfs", "liwcFeatures", "languageToolFeatures")).setOutputCol("assembledIdfFeatures")

    val sentimentChiSqSelector: ChiSqSelector = new ChiSqSelector().setFeaturesCol("assembledIdfFeatures").setLabelCol("sentiment").setOutputCol("topFeatures").setNumTopFeatures(1000)
    val relevanceChiSqSelector: ChiSqSelector = new ChiSqSelector().setFeaturesCol("assembledIdfFeatures").setLabelCol("relevanceLabel").setOutputCol("topFeatures").setNumTopFeatures(1000)

    val sentimentMLP: MultilayerPerceptronClassifier = {
      def sentimentBowLayers: Array[Int] = Array(1000) ++ intermediateLayers ++ Array(3)

      new MultilayerPerceptronClassifier().setPredictionCol("15r643543").setProbabilityCol("5342").setLabelCol("sentiment").setFeaturesCol("topFeatures")
        .setLayers(sentimentBowLayers).setBlockSize(128).setSeed(Random.nextLong).setMaxIter(numIterations).setRawPredictionCol("bowClassifierPrediction")
    }

    val relevanceMLPs: MultilayerPerceptronClassifier = {
      def relevanceBowLayers: Array[Int] = Array(1000) ++ intermediateLayers ++ Array(2)

      new MultilayerPerceptronClassifier().setPredictionCol("3534543").setProbabilityCol("4534543").setLabelCol("relevanceLabel").setFeaturesCol("topFeatures")
        .setLayers(relevanceBowLayers).setBlockSize(128).setSeed(Random.nextLong()).setMaxIter(numIterations).setRawPredictionCol("bowClassifierPrediction")
    }

    /*
     * mlp model definition
     */



    println("fitting models")


    val sentimentEvaluator = new MulticlassClassificationEvaluator().setMetricName("f1").setLabelCol("sentiment")

    val relevanceEvaluator = new MulticlassClassificationEvaluator().setMetricName("f1").setLabelCol("relevanceLabel")

    val sentimentPipelineMLP = new Pipeline().setStages(Array(languageTool, germanStemmer, nGramStep, hashingTermFrequencies, idf, bowVectorAssembler, sentimentChiSqSelector, sentimentMLP))
    val relevancePipelineMLP = new Pipeline().setStages(Array(languageTool, germanStemmer, nGramStep, hashingTermFrequencies, idf, bowVectorAssembler, relevanceChiSqSelector, relevanceMLPs))

    val sentimentModelsMLP = sentimentTrainingDataFrame.map(dataset => sentimentPipelineMLP.fit(dataset))
    val relevanceModelsMLP = relevanceTrainingDataFrame.map(dataset => relevancePipelineMLP.fit(dataset))

    val sentimentValidationData: DataFrame = mergeSlices(sentimentModelsMLP.map(model => model.transform(sentimentTestDataFrame)), "sentiment", "topFeatures", "bowClassifierPrediction")
    val relevanceValidationData: DataFrame = mergeSlices(relevanceModelsMLP.map(model => model.transform(relevanceTestDataFrame)), "relevanceLabel", "topFeatures", "bowClassifierPrediction")


    val labelSelector = new LabelSelector().setPredictionColumnPrefix("prediction").setPredictionCol("prediction")

    val sentimentPredictions = labelSelector.transform(sentimentValidationData)
    val relevancePredictions = labelSelector.transform(relevanceValidationData)

    println("f1 sentiment: " + sentimentEvaluator.evaluate(sentimentPredictions))
    println("f1 relevance: " + relevanceEvaluator.evaluate(relevancePredictions))

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

  def mergeSlices(slices: Array[DataFrame], labelCol: String, featureCol: String, predictionCol: String): DataFrame = {
    val spark = SparkSession.builder().getOrCreate()
    //get all predictions


    val predictionsInSlice: Array[DataFrame] = slices.zipWithIndex.map(df => spark.createDataFrame(df._1.rdd.map(r => r.getAs[linalg.Vector](predictionCol)).zipWithIndex()
      .map(d => new GenericRowWithSchema(Array(d._1) ++ Array(d._2), new StructType()
        .add("prediction_" + df._2, linalg.SQLDataTypes.VectorType, false)
        .add("index", LongType, false)).asInstanceOf[Row]), new StructType()
      .add("prediction_" + df._2, linalg.SQLDataTypes.VectorType, false)
      .add("index", LongType, false)))

    var resultDf: DataFrame = spark.createDataFrame(slices.head.select(labelCol, featureCol).rdd.zipWithIndex().map(row => {
      val rowData = row._1.toSeq.toArray
      new GenericRowWithSchema(rowData ++ Array(row._2), row._1.schema.add("index", LongType, nullable = false)).asInstanceOf[Row]
    }), slices.head.select(labelCol, featureCol).schema.add("index", LongType, nullable = false))

    for (predictions <- predictionsInSlice) {
      val joinedDf = resultDf.join(predictions).where(resultDf("index") === predictions("index")).drop("index")
      resultDf = spark.createDataFrame(joinedDf.rdd.zipWithIndex().map(row => {
        val rowData = row._1.toSeq.toArray
        new GenericRowWithSchema(rowData ++ Array(row._2), row._1.schema.add("index", LongType, nullable = false)).asInstanceOf[Row]
      }), joinedDf.schema.add("index", LongType, nullable = false))
    }

    resultDf
  }


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