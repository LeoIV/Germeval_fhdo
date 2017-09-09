package de.fhdo.germeval.modelling.fasttext

import java.io.{File, FileOutputStream, FileWriter, OutputStreamWriter}
import java.util.concurrent.Executors

import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

import scala.collection.immutable.HashMap
import scala.concurrent.{Future, _}
import scala.io.Source
import scala.sys.process._
import scala.util.Try

/**
  * Wrapper object for fastText. Assumes you have fastText installed and added to your path variable
  *
  * Created by hoevelmann on 22.03.2017.
  */
object FastText {

  private val log = LoggerFactory.getLogger(this.getClass)

  private var (successCounter, timeoutCounter) = (0, 0)

  //limit the number of executors for fasttext
  implicit val executionContext = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(12))


  /**
    * Train a supervised model with fastText and write the model to modelFilePath
    *
    * @param samples                  The text samples to train on. Tuple consisting of text and label.
    * @param modelFilePath            The filepath the model is written to
    * @param learningRate             learning rate [0.1]
    * @param learningRateUpdateRate   change the rate of updates for the learning rate [100]
    * @param wordVectorDimensionality size of word vectors [100]
    * @param contextWindowSize        size of the context window [5]
    * @param numberOfEpochs           number of epochs [5]
    * @param minCount                 minimal number of word occurences [1]
    * @param minCountLabel            minimal number of label occurences [0]
    * @param neg                      number of negatives sampled [5]
    * @param wordNGrams               max length of word ngram [1]
    * @param loss                     loss function {ns, hs, softmax} [ns]
    * @param bucket                   number of buckets [2000000]
    * @param minn                     min length of char ngram [0]
    * @param maxn                     max length of char ngram [0]
    * @param thread                   number of threads [#cpu cores]
    * @param t                        sampling threshold [0.0001]
    * @param label                    labels prefix [__label__], changing this is not recommended!
    * @param verbose                  verbosity level [2]
    */
  def trainModel(samples: Seq[(String, Double)], modelFilePath: String, learningRate: Double = 0.1,
                 learningRateUpdateRate: Int = 100, wordVectorDimensionality: Int = 100, contextWindowSize: Int = 5,
                 numberOfEpochs: Int = 5, minCount: Int = 1, minCountLabel: Int = 0, neg: Int = 5, wordNGrams: Int = 1,
                 loss: String = "ns", bucket: Int = 2000000, minn: Int = 0, maxn: Int = 0,
                 thread: Int = Runtime.getRuntime.availableProcessors, t: Double = 0.0001,
                 label: String = "__label__", verbose: Int = 2): Unit = {

    //writing temporary training file
    val trainingDocumentWriter = new OutputStreamWriter(new FileOutputStream("temporary_training_file.temp"), "utf-8")
    samples.foreach(sample => {
      trainingDocumentWriter.write(sample._1 + " " + label + sample._2 + "\n")
    })
    trainingDocumentWriter.close()

    val syscall = "fasttext supervised -input temporary_training_file.temp -output " + modelFilePath + " -dim " +
      wordVectorDimensionality + " -ws " + contextWindowSize + " -lr " + learningRate + " -lrUpdateRate " + learningRateUpdateRate + " -epoch " + numberOfEpochs + " -minCount " + minCount +
      " -minCountLabel " + minCountLabel + " -neg " + neg + " -wordNgrams " + wordNGrams + " -loss " + loss + " -bucket " + bucket + " -minn +" + minn + " -maxn " + maxn +
      " -thread " + thread + " -t " + t + " -label " + label + " -verbose " + verbose
    syscall.!
    //new File("temporary_training_file.temp").delete()
  }

  def evaluateModel(modelFilePath: String, samples: Seq[(String, Double)]): Double = {
    val targetDocumentWriter = new OutputStreamWriter(new FileOutputStream("test.txt"), "utf-8")
    samples.foreach(review => {
      targetDocumentWriter.write(review._1 + "\n")
    })
    targetDocumentWriter.close()

    val systemCallString = "fasttext predict " + modelFilePath + ".bin test.txt"
    val sysCallLines = systemCallString.lineStream_!.toList.map(_.replace("__label__", ""))
    new File("text.txt").delete
    val labels: Seq[Double] = samples.map(_._2)
    println("lines. " + sysCallLines)
    val predictions: List[Double] = sysCallLines.map(_.toDouble)

    val spark = SparkSession.builder().getOrCreate()

    val labelsAndPredictionsRdd: RDD[(Double, Double)] = spark.sparkContext.parallelize(labels.zip(predictions))
    val mcm = new MulticlassMetrics(labelsAndPredictionsRdd)

    mcm.weightedFMeasure
  }

  def sentenceToVectorFromHashMap(sentences: Seq[Seq[String]], wordVectors: HashMap[String, Array[Double]]): Seq[linalg.Vector] = {
    this.synchronized {
      sentences.map(sentence => {
        val sentenceWordVectors: Seq[Array[Double]] = sentence
          .map(word => Try(wordVectors(word)))
          .filter(_.isSuccess)
          .map(_.get)
        val sentenceLength = sentenceWordVectors.length.toDouble
        val vector = Try(Vectors.dense(sentenceWordVectors
          .reduce((a, b) => a.zip(b).map { case (e1, e2) => e1 + e2 })
          .map(entry => entry / sentenceLength)))
        //if(vector.isFailure) println("failure")
        vector.getOrElse(Vectors.dense(new Array[Double](100)))
      })
    }
  }

  def sentenceToVector(sentence: String, modelFilePath: String): Array[Double] = {
    val s1 = "echo " + sentence.replaceAll("^(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]", "").replace("(", "").replace(")", "").split("\\s+\\.\\,").mkString(" ").toLowerCase;
    val s2 = "fasttext print-sentence-vectors " + modelFilePath
    val future = Future(blocking((s1 #| s2).lineStream_!))
    //wait at most 120 seconds for fasttext, drop this line otherwise
    val executionResult: Stream[String] = try {
      Await.result(future, duration.Duration(120, "sec"))
    } catch {
      case _: TimeoutException =>
        log.error("timeout during vector computation")
        Stream.empty
    }
    if (executionResult.isEmpty) {
      timeoutCounter += 1
      Array.empty[Double]
    }
    else {
      successCounter += 1
      executionResult.head.trim.split(" ").filter(f => Try(f.toDouble).toOption.isDefined).map(_.toDouble)
    }
  }

  def wordToVector(word: String, modelFilePath: String): Array[Double] = {
    val s1 = "echo " + word.toLowerCase
    val s2 = "fasttext print-word-vectors " + modelFilePath
    val future = Future(blocking((s1 #| s2).lineStream_!))
    //wait at most 120 seconds for fasttext, drop this line otherwise
    val executionResult: Stream[String] = try {
      Await.result(future, duration.Duration(120, "sec"))
    } catch {
      case _: TimeoutException =>
        log.error("timeout during vector computation")
        Stream.empty
    }
    if (executionResult.isEmpty) Array.empty[Double]
    else executionResult.head.trim.split(" ").filter(f => Try(f.toDouble).toOption.isDefined).map(_.toDouble)
  }

  def wordToVectorBatch(words: Seq[String], modelFilePath: String): Seq[Array[Float]] = {
    val tmpWriteFile = new File("tmpWriteFile")
    val tmpReadFile = new File("tmpReadFile")
    val fw = new FileWriter(tmpWriteFile)
    words.foreach(word => fw.write(word + "\n"))
    fw.close()
    println("path: " + tmpWriteFile.getAbsolutePath)
    val s1 = "cat " + tmpWriteFile.getAbsolutePath
    val s2 = "fasttext print-word-vectors " + modelFilePath
    s1 #| s2 #> tmpReadFile !
    //wait at most 120 seconds for fasttext, drop this line otherwise
    val executionResult: Seq[String] = Source.fromFile(tmpReadFile.getAbsolutePath).getLines().toSeq
    tmpWriteFile.delete()
    tmpReadFile.delete()
    if (executionResult.isEmpty) Array.empty[Array[Float]]
    else {
      val res = executionResult
      //println("res size: " + res.size)
      // res.foreach(println)
      res.map(x => x.trim.split(" ").tail.map(_.toFloat))
    }
  }


}
