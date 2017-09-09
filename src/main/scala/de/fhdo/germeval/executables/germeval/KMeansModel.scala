package de.fhdo.germeval.executables.germeval

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SparkSession}


/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 10.08.2017.
  */
object KMeansModel {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().master("local[*]").getOrCreate()


    val sc = spark.sparkContext
    // Load and parse the data
    val parsedData: RDD[String] = sc.textFile("germeval/corpus/wholeModel.vec")
    val datasetSchema = new StructType().add("features", SQLDataTypes.VectorType)
    val rowRdd: RDD[Row] = parsedData.map(s => s.split(" ").tail.map(_.toDouble))
      .filter(_.length == 100)
      .map(s => new GenericRowWithSchema(Array(Vectors.dense(s)), datasetSchema).asInstanceOf[Row])

    val dataset = spark.sqlContext.createDataFrame(rowRdd, datasetSchema)

    // Cluster the data into two classes using KMeans
    val clusters = new KMeans().setK(100).setSeed(1L).setMaxIter(20)
    val model = clusters.fit(dataset)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = model.computeCost(dataset)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    // Save and load model
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)
    model.save("kMeansModel")
  }

}
