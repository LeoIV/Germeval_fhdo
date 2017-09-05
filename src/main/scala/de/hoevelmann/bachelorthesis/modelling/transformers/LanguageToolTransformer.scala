package de.hoevelmann.bachelorthesis.modelling.transformers

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.languagetool.JLanguageTool
import org.languagetool.language.GermanyGerman
import org.languagetool.rules.Rule
import org.languagetool.rules.de.GermanSpellerRule

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.util.Try

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 19.06.2017.
  */
class LanguageToolTransformer(val uid: String) extends Transformer with DefaultParamsWritable {

  val inputCol: Param[String] = new Param[String](this, "inputCol", "inputCol", (x: String) => x.nonEmpty)

  def setInputCol(value: String): this.type = set(inputCol, value)

  def getInputCol: String = $(inputCol)

  val outputCol: Param[String] = new Param[String](this, "outputCol", "outputCol", (x: String) => x.nonEmpty)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def getOutputCol: String = $(outputCol)

  override def transform(dataset: Dataset[_]): DataFrame = {

    val ignoreTokens = List("<<<bahn_hashtag>>>", "<<<money_amount>>>", "<<<annoyed_dots>>>", "<<<happy_smiley>>>", "<<<laughing_smiley>>>", "<<<sad_smiley>>>",
      "<<<winking_smiley>>>", "<<<strong_exclamation>>>", "<<<strong_question>>>", "<<<db_username>>>", "<<<twitter_username>>>", "<<<heart>>>", "re", "rt",
      "<<<number>>>", "<<<quote>>>")

    LanguageToolTransformer.allRules().filter(_.isInstanceOf[GermanSpellerRule]).head.asInstanceOf[GermanSpellerRule].addIgnoreTokens(ignoreTokens.asJava)

    val df = dataset.toDF.cache()

    def runLanguageTool(words: Seq[String]) = {
      LanguageToolTransformer.check(words.mkString(" "))
    }

    val udfRunLanguageTool = udf(runLanguageTool _)

    val resDf = df.select(df("*"), udfRunLanguageTool(df($(inputCol))).as($(outputCol)))
    resDf
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema.add($(outputCol), DoubleType)

  def this() = this(Identifiable.randomUID("LanguageToolTransformer"))
}

private object LanguageToolTransformer {

  private val lt = new JLanguageTool(new GermanyGerman())

 private val cacheMap: mutable.HashMap[Int, Double] = scala.collection.mutable.HashMap[Int, Double]()

  private def check(sentence: String): Double = {
    val sentenceHashCode = sentence.hashCode
    val resultTry = Try(cacheMap(sentenceHashCode))
    if (resultTry.isSuccess) resultTry.get
    else {
      val result = synchronized {
        val checkResult = lt.check(sentence).asScala
        val numMatches = checkResult.count(_.getRule.isInstanceOf[GermanSpellerRule])
        val res = numMatches.toDouble / sentence.split(" ").length.toDouble
        cacheMap.put(sentenceHashCode, res)
        res
      }
      result
    }
  }

  private def allRules(): mutable.Seq[Rule] = synchronized(lt.getAllRules).asScala
}