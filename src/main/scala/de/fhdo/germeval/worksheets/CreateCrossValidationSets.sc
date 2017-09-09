import java.io.FileWriter

import scala.io.Source

val trainList = Source.fromFile("C:/Users/hoevelmann/IdeaProjects/Bachelorarbeit/liwc_train.csv").getLines()
  .map(_.split("\t")).toList.tail

val devList = Source.fromFile("C:/Users/hoevelmann/IdeaProjects/Bachelorarbeit/liwc_dev.csv").getLines()
  .map(_.split("\t")).toList.tail

val allLines: List[Array[String]] = trainList ++ devList
val allGroups: List[(List[Array[String]], Int)] = allLines.grouped(allLines.size / 5).toList.zipWithIndex

for (i <- 0 until 5) {
  val testSet: List[Array[String]] = allGroups.filter(_._2 == i).flatMap(_._1)
  val trainingSet: List[Array[String]] = allGroups.filter(_._2 != i).flatMap(_._1)

  val testFw = new FileWriter("C:/Users/hoevelmann/Downloads/test_cv_" + i + ".tsv")
  val trainFw = new FileWriter("C:/Users/hoevelmann/Downloads/train_cv_" + i + ".tsv")
  testSet.foreach(line => testFw.write(line(0) + "\t" + line(1) + "\t" + line(2) + "\t" + line(3) + "\t" + line(4) + "\n"))
  testFw.close()

  trainingSet.foreach(line => trainFw.write(line(0) + "\t" + line(1) + "\t" + line(2) + "\t" + line(3) + "\t" + line(4) + "\n"))
  trainFw.close()
}
