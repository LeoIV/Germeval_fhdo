import java.io.FileWriter

import scala.io.Source

val trainFiles = Source.fromFile("C:/Users/hoevelmann/IdeaProjects/Bachelorarbeit/liwc_train.csv")
  .getLines().toSeq.map(_.split("\t")).tail
val devFiles = Source.fromFile("C:/Users/hoevelmann/IdeaProjects/Bachelorarbeit/liwc_dev.csv")
  .getLines().toSeq.map(_.split("\t")).tail
val testFileTs1 = Source.fromFile("C:/Users/hoevelmann/IdeaProjects/Bachelorarbeit/liwc_test_ts1.csv")
.getLines().toSeq.map(_.split("\t")).tail
val testFileTs2 = Source.fromFile("C:/Users/hoevelmann/IdeaProjects/Bachelorarbeit/liwc_test_ts2.csv")
  .getLines().toSeq.map(_.split("\t")).tail
val testFiles = testFileTs1 ++ testFileTs2

val numTrain = trainFiles.size
val numDev = devFiles.size
val numTest = testFiles.size

//relevance-
val numTrueTrain = trainFiles.map(_ (2).toBoolean).count(_ == true)
val numFalseTrain = numTrain - numTrueTrain

val numTrueDev = devFiles.map(_ (2).toBoolean).count(_ == true)
val numFalseDev = numDev - numTrueDev

//sentiment
val numPosTrain = trainFiles.map(_ (3)).count(_ == "positive")
val numNeuTrain = trainFiles.map(_ (3)).count(_ == "neutral")
val numNegTrain = trainFiles.map(_ (3)).count(_ == "negative")

val numPosDev = devFiles.map(_ (3)).count(_ == "positive")
val numNeuDev = devFiles.map(_ (3)).count(_ == "neutral")
val numNegDev = devFiles.map(_ (3)).count(_ == "negative")

//wordcounts
val avWcPosTrain = trainFiles.filter(_ (3) == "positive").map(_ (1).split(" ").length).sum.toDouble / numPosTrain.toDouble
val sdWcPosTrain = Math.sqrt((1.0 / (numPosTrain.toDouble - 1.0)) * trainFiles.filter(_ (3) == "positive").map(_ (1).split(" ").length).map(x => Math.pow(x.toDouble - avWcPosTrain, 2.0)).sum)
val avWcNeuTrain = trainFiles.filter(_ (3) == "neutral").map(_ (1).split(" ").length).sum.toDouble / numNeuTrain.toDouble
val sdWcNeuTrain = Math.sqrt((1.0 / (numNeuTrain.toDouble - 1.0)) * trainFiles.filter(_ (3) == "neutral").map(_ (1).split(" ").length).map(x => Math.pow(x.toDouble - avWcNeuTrain, 2.0)).sum)
val avWcNegTrain = trainFiles.filter(_ (3) == "negative").map(_ (1).split(" ").length).sum.toDouble / numNegTrain.toDouble
val sdWcNegTrain = Math.sqrt((1.0 / (numNegTrain.toDouble - 1.0)) * trainFiles.filter(_ (3) == "negative").map(_ (1).split(" ").length).map(x => Math.pow(x.toDouble - avWcNegTrain, 2.0)).sum)

val avWcPosDev = devFiles.filter(_ (3) == "positive").map(_ (1).split(" ").length).sum.toDouble / numPosDev.toDouble
val sdWcPosDev = Math.sqrt((1.0 / (numPosDev.toDouble - 1.0)) * devFiles.filter(_ (3) == "positive").map(_ (1).split(" ").length).map(x => Math.pow(x.toDouble - avWcPosDev, 2.0)).sum)
val avWcNeuDev = devFiles.filter(_ (3) == "neutral").map(_ (1).split(" ").length).sum.toDouble / numNeuDev.toDouble
val sdWcNeuDev = Math.sqrt((1.0 / (numNeuDev.toDouble - 1.0)) * devFiles.filter(_ (3) == "neutral").map(_ (1).split(" ").length).map(x => Math.pow(x.toDouble - avWcNeuDev, 2.0)).sum)
val avWcNegDev = devFiles.filter(_ (3) == "negative").map(_ (1).split(" ").length).sum.toDouble / numNegDev.toDouble
val sdWcNegDev = Math.sqrt((1.0 / (numNegDev.toDouble - 1.0)) * devFiles.filter(_ (3) == "negative").map(_ (1).split(" ").length).map(x => Math.pow(x.toDouble - avWcNegDev, 2.0)).sum)

val avWcTest = testFiles.map(_ (1).split(" ").length).sum.toDouble / numTest.toDouble
val sdWcTest = Math.sqrt((1.0 / (numTest.toDouble - 1.0)) * testFiles.map(_ (1).split(" ").length).map(x => Math.pow(x.toDouble - avWcTest, 2.0)).sum)

val avWcTrueTrain = trainFiles.filter(_ (2).toBoolean).map(_ (1).split(" ").length).sum.toDouble / numTrueTrain.toDouble
val sdWcTrueTrain = Math.sqrt((1.0 / (numTrueTrain.toDouble - 1.0)) * trainFiles.filter(_ (2).toBoolean).map(_ (1).split(" ").length).map(x => Math.pow(x.toDouble - avWcTrueTrain, 2.0)).sum)
val avWcFalseTrain = trainFiles.filter(!_ (2).toBoolean).map(_ (1).split(" ").length).sum.toDouble / numFalseTrain.toDouble
val sdWcFalseTrain = Math.sqrt((1.0 / (numFalseTrain.toDouble - 1.0)) * trainFiles.filter(!_ (2).toBoolean).map(_ (1).split(" ").length).map(x => Math.pow(x.toDouble - avWcFalseTrain, 2.0)).sum)

val avWcTrueDev = devFiles.filter(_ (2).toBoolean).map(_ (1).split(" ").length).sum.toDouble / numTrueDev.toDouble
val sdWcTrueDev = Math.sqrt((1.0 / (numTrueDev.toDouble - 1.0)) * devFiles.filter(_ (2).toBoolean).map(_ (1).split(" ").length).map(x => Math.pow(x.toDouble - avWcTrueDev, 2.0)).sum)
val avWcFalseDev = devFiles.filter(!_ (2).toBoolean).map(_ (1).split(" ").length).sum.toDouble / numFalseDev.toDouble
val sdWcFalseDev = Math.sqrt((1.0 / (numFalseDev.toDouble - 1.0)) * devFiles.filter(!_ (2).toBoolean).map(_ (1).split(" ").length).map(x => Math.pow(x.toDouble - avWcFalseDev, 2.0)).sum)


// === PRINTING ===

val posWcsTrain = trainFiles.filter(_ (3) == "positive").map(_ (1).split(" ").length).toList
val posWcsDev = devFiles.filter(_ (3) == "positive").map(_ (1).split(" ").length).toList
val neuWcsTrain = trainFiles.filter(_ (3) == "neutral").map(_ (1).split(" ").length).toList
val neuWcsDev = devFiles.filter(_ (3) == "neutral").map(_ (1).split(" ").length).toList
val negWcsTrain = trainFiles.filter(_ (3) == "negative").map(_ (1).split(" ").length).toList
val negWcsDev = devFiles.filter(_ (3) == "negative").map(_ (1).split(" ").length).toList

val trueWcsTrain = trainFiles.filter(_ (2).toBoolean).map(_ (1).split(" ").length).toList
val trueWcsDev = devFiles.filter(_ (2).toBoolean).map(_ (1).split(" ").length).toList
val falseWcsTrain = trainFiles.filter(!_ (2).toBoolean).map(_ (1).split(" ").length).toList
val falseWcsDev = devFiles.filter(!_ (2).toBoolean).map(_ (1).split(" ").length).toList

val maxLinesSenTrain: Int = Seq(posWcsTrain.length, neuWcsTrain.length, negWcsTrain.length).max
val maxLinesSenDev: Int = Seq(posWcsDev.length, neuWcsDev.length, negWcsDev.length).max
val maxLinesRelTrain: Int = Seq(trueWcsTrain.length, falseWcsTrain.length).max
val maxLinesRelDev: Int = Seq(trueWcsDev.length, falseWcsDev.length).max

val fwSenTrain = new FileWriter("C:/Users/hoevelmann/Downloads/wordCountsSentimentTrain.csv")
val fwRelTrain = new FileWriter("C:/Users/hoevelmann/Downloads/wordCountsRelevanceTrain.csv")
val fwSenDev = new FileWriter("C:/Users/hoevelmann/Downloads/wordCountsSentimentDev.csv")
val fwRelDev = new FileWriter("C:/Users/hoevelmann/Downloads/wordCountsRelevanceDev.csv")

for(i <- 0 until maxLinesSenDev){
  val sb = new StringBuilder
  if (i < trueWcsDev.length) sb.append(trueWcsDev(i) + ";") else sb.append(";")
  if (i < falseWcsDev.length) sb.append(falseWcsDev(i))
  fwRelDev.write(sb.result() + "\n")
}
fwRelDev.close()

for(i <- 0 until maxLinesSenTrain){
  val sb = new StringBuilder
  if (i < trueWcsTrain.length) sb.append(trueWcsTrain(i) + ";") else sb.append(";")
  if (i < falseWcsTrain.length) sb.append(falseWcsTrain(i))
  fwRelTrain.write(sb.result() + "\n")
}
fwRelTrain.close()

for (i <- 0 until maxLinesSenTrain) {
  val sb = new StringBuilder
  if (i < posWcsTrain.length) sb.append(posWcsTrain(i) + ";") else sb.append(";")
  if (i < neuWcsTrain.length) sb.append(neuWcsTrain(i) + ";") else sb.append(";")
  if (i < negWcsTrain.length) sb.append(negWcsTrain(i) + "")
  fwSenTrain.write(sb.result() + "\n")
}
fwSenTrain.close()

for (i <- 0 until maxLinesSenDev) {
  val sb = new StringBuilder
  if (i < posWcsDev.length) sb.append(posWcsDev(i) + ";") else sb.append(";")
  if (i < neuWcsDev.length) sb.append(neuWcsDev(i) + ";") else sb.append(";")
  if (i < negWcsDev.length) sb.append(negWcsDev(i) + "")
  fwSenDev.write(sb.result() + "\n")
}
fwSenDev.close()