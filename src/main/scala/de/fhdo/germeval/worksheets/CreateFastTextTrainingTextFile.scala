package de.fhdo.germeval.worksheets

import java.io.FileWriter

import de.hoevelmann.bachelorthesis.executables.germeval.GermevalWordSequenceProcessor

import scala.io.Source

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 28.07.2017.
  */
object CreateFastTextTrainingTextFile {

  def main(args: Array[String]): Unit = {

    val trainLines = Source.fromFile("germeval/corpus/liwc_train.csv").getLines().toList

    /*   val allFiles = Array(
         ("trainFile", trainLines),
         ("devFile", devLines),
         ("testFileTs1", testFileTs1),
         ("testFileTs2", testFileTs2)
       )*/

    def processLines(lines: Seq[String]): Seq[String] = lines
      .map(_.split("\t").toList)
      //  .filter(_.length == 89)
      .map(x => GermevalWordSequenceProcessor.processWordSequence(x(1)).mkString(" ") + (if (x(3) == "negative") "__label__0.0" else if (x(3) == "positive") "__label__2.0" else "__label__1.0"))

      val fw = new FileWriter("newtrainfile")
      processLines(trainLines).foreach(x =>
        fw.write(x + "\n")
      )
      fw.close()

  }


}
