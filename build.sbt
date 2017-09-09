
name := "Germeval_fhdo"

version := "0.1"

scalaVersion := "2.11.11"

// we need this configuration to make the languagetool work
assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs@_*) =>
    xs map (_.toLowerCase) match {
      // keep the language tool stuff in META-INF folder
      case ("org" :: "languagetool" :: xx) => {
        MergeStrategy.first
      }
      //discard everything else
      case _ => {
        MergeStrategy.discard
      }
    }
  //keep everything, that isn't in META-INF
  case x => MergeStrategy.first
}

test in assembly := {}

libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.2.0"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.2.0"
libraryDependencies += "org.apache.spark" % "spark-sql_2.11" % "2.2.0"
libraryDependencies += "org.apache.spark" % "spark-streaming_2.11" % "2.2.0"
libraryDependencies += "org.apache.spark" % "spark-streaming_2.11" % "2.2.0"
libraryDependencies += "org.ejml" % "core" % "0.30"
libraryDependencies += "org.ejml" % "simple" % "0.30"
libraryDependencies += "org.apache.opennlp" % "opennlp-tools" % "1.7.2"
libraryDependencies += "com.eatthepath" % "jvptree" % "0.1"
libraryDependencies += "org.slf4j" % "slf4j-api" % "1.7.25"
libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.1"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1"
libraryDependencies += "com.intel.analytics.bigdl" % "bigdl-SPARK_2.1" % "0.1.1" % "provided"
libraryDependencies += "org.scala-lang" % "scala-reflect" % "2.11.8"
libraryDependencies += "org.languagetool" % "language-de" % "3.8"
libraryDependencies += "com.fasterxml.jackson.core" % "jackson-databind" % "2.6.5" force()
libraryDependencies += "com.fasterxml.jackson.core" % "jackson-core" % "2.6.5" force()
libraryDependencies += "com.fasterxml.jackson.core" % "jackson-annotations" % "2.6.5" force()
libraryDependencies += "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.6.5" force()
libraryDependencies += "com.github.fommil.netlib" % "core" % "1.1.2"
libraryDependencies += "org.apache.hadoop" % "hadoop-client" % "2.7.2" force()