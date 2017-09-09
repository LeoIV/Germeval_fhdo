package de.hoevelmann.bachelorthesis.executables.germeval

import java.io.FileWriter

import scala.io.Source

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 29.07.2017.
  */
object WikipediaTransformer {

  def main(args: Array[String]) {
    val fw = new FileWriter("dewiki")
    var c = 0
    val lines = Source.fromFile("C:/Users/hoevelmann/Downloads/dewiki.xml").getLines()
    var take = false
    for (line <- lines.filterNot(l =>
      l.trim.startsWith("|")
        || l.trim.startsWith("!")
        || l.trim.startsWith("{")
        || l.trim.startsWith("*")
        || l.trim.startsWith(":")
        || l.trim.startsWith("==")
        || l.toLowerCase.indexOf("notoc") >= 0
        || l.trim.startsWith("#"))) {
      if (line.trim.startsWith("<text")) take = true
      if (line.trim.startsWith("</text")) take = false
      if (take) {
        //println(line)
        val transformedLine = line
          .trim
          .toLowerCase
          .replaceAll("<.*>", "")
          .replaceAll("&amp;", "&")
          .replaceAll("&lt;", "<")
          .replaceAll("&gt;", ">")
          .replaceAll("<ref[^<]*<\\/ref>", "")
          .replaceAll("<[^>]*>", "")
          .replaceAll("\\[http:[^] ]*", "[")
          .replaceAll("\\|thumb", "")
          .replaceAll("\\|left", "")
          .replaceAll("\\|right", "")
          .replaceAll("\\|\\d+px", "")
          .replaceAll("\\[\\[image:[^\\[\\]]*\\|", "")
          .replaceAll("\\[\\[category:([^|\\]]*)[^]]*\\]\\]/[[$1]]", "")
          .replaceAll("\\[\\[[a-z\\-]*:[^\\]]*\\]\\]", "")
          .replaceAll("\\[\\[[^\\|\\]]*\\|", "[[")
          .replaceAll("\\{\\{[^\\}]*\\}\\}", "")
          .replaceAll("\\{[^\\}]*\\}", "")
          .replaceAll("\\[", "")
          .replaceAll("\\]", "")
          .replaceAll("&[^;]*;", " ")
          .replaceAll("<", "")
          .replaceAll(">", "")
          .replaceAll("\n", "")
          .replaceAll("[0-9]+", " <<<number>>> ")
          .filter("abcdefghijklmnopqrstuvwxyzäöüß<> ".contains(_))
        if (!transformedLine.isEmpty) {
          c+=1
          if(c%100000 == 0) println(c)
          fw.write(transformedLine + "\n")
        }
      }
    }
    fw.close()
  }
}
