package de.hoevelmann.bachelorthesis.executables.germeval

import java.io.FileWriter

import scala.io.Source
import scala.util.Random

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 30.07.2017.
  */
object GermevalToBareText {

  def main(args: Array[String]): Unit = {
    val trainLines = Source.fromFile("germeval/corpus/liwc_train.csv")
      .getLines()
      .map(_.split("\t")(1))
      .map(GermevalWordSequenceProcessor.processWordSequence)

    val devLines = Source.fromFile("germeval/corpus/liwc_dev.csv")
      .getLines()
      .map(_.split("\t")(1))
      .map(GermevalWordSequenceProcessor.processWordSequence)

    val testLines = Source.fromFile("germeval/corpus/test_clear.csv")
      .getLines()
      .map(_.split("\t")(1))
      .map(GermevalWordSequenceProcessor.processWordSequence)

    val fw = new FileWriter("germeval-bare-text.txt")
    Random.shuffle(trainLines ++ devLines ++ testLines).foreach(x =>
      fw.write(x.mkString(" ") + "\n")
    )

    fw.close()
  }

}
