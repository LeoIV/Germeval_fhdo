package de.hoevelmann.bachelorthesis.executables.germeval


import java.io.FileWriter

import scala.io.Source

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 30.07.2017.
  */
object NewsCorpusTransformer {

  def main(args: Array[String]): Unit = {
    val newsFw = new FileWriter("news2013.txt")
    Source.fromFile("C:/Users/hoevelmann/Downloads/news.2013.de.shuffled/news.2013.de.shuffled").getLines()
      .map(_.toLowerCase)
      .map(_.replaceAll("[0-9]+", "<<<number>>>"))
      .map(line => line.filter("abcdefghijklmnopqrstuvwxyzäöüß<> ".contains(_)))

      .foreach(l => newsFw.write(l + "\n"))
    newsFw.close()
  }

}
