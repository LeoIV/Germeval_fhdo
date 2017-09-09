package de.hoevelmann.bachelorthesis.executables.germeval

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 11.07.2017.
  */
object GermevalWordSequenceProcessor {

  def processWordSequence(wordSequence: String): Seq[String] = {
    wordSequence
      .toLowerCase
      .replaceAll("[0-9]*-?[0-9]+\\s?min((ute)?n?)((ütige)?r?)", " <<<tokentimeexpression>>> ")
      .replace("s bahn", "sbahn")
      .replace("s-bahn", "sbahn")
      .split(" ")
      .flatMap(_.split("-")).map(_
      .replaceAll("(https?:\\/\\/)?([a-zA-Z0-9\\-.]+\\.[a-zA-Z0-9\\-]+([\\/]([a-zA-Z0-9_\\/\\-.?&%=+])*)*)", " <<<tokenhyperlink>>> ") //replace all links
      .replace("#bahn", " <<<tokenbahnhashtag>>> ")
      .replace("#db", " <<<tokenbahnhashtag>>> ")
      .replace("#", "")
      .replaceAll("[0-9]+min", " <<<tokentime>>> ")
      .replaceAll("[0-9]+€", "<<<tokenmoneyamount>>> ")
      .replaceAll("[0-9]+", " <<<tokennumber>>> ")
      .replace("&", "und")
      .replaceAll("\\.{3,}", " <<<tokenannoyeddots>>> ")
      .replace(":-)", " <<<tokenhappysmiley>>> ") //replace happy smiley
      .replace(":)", " <<<tokenhappysmiley>>> ") //replace happy smiley
      .replace("☺", " <<<tokenhappysmiley>>> ") //replace happy smiley
      .replace(":))", " <<<tokenhappysmiley>>> ") //replace happy smiley
      .replace(":-D", " <<<tokenlaughingsmiley>>> ") //replace laughing smiley
      .replace("xD", " <<<tokenlaughingsmiley>>> ") //replace laughing smiley
      .replace(":-(", " <<<tokensadsmiley>>> ") //replace sad smiley
      .replace(":(", " <<<tokensadsmiley>>> ") //replace sad smiley
      .replace(";-)", " <<<tokenwinkingsmiley>>> ") //replace winking smiley
      .replace(",", "") //remove comma
      .replace(".", "") //remove dot
      .replace(":", "")
      .replace("-", "")
      .replace("/", " ")
      .replace("%", " ")
      .replace("+", " ")
      .replace(";", "")
      .replaceAll("!!+", " <<<tokenstrongexclamation>>> ") //replace more than two exclamation signs with <<<strong_exclamation>>>
      .replaceAll("\\?\\?+", " <<<tokenstrongquestion>>> ") //replace more than two questionmarks with <<<strong_question>>>
      .replace("?", "")
      .replace("!", "")
      .replace("(", "") //now remove braces
      .replace(")", "")
      .replace("[", "")
      .replace("]", "")
      .replace("|", "")
      .replace("\"", " <<<tokenquote>>> ") //replace quotes
      .replace("„", " <<<tokenquote>>> ") //replace quotes
      .replace("“", " <<<tokenquote>>> ") //replace quotes
      .replace("@DB_Bahn", " <<<tokendbusername>>> ")
      .replace("@BahnAnsagen", " <<<tokendbusername>>> ")
      .replace("@DB_Info", " <<<tokendbusername>>> ")
      .replace("db", " <<<tokendbusername>>> ")
      .replace("❤", " <<<tokenheart>>> ")
      .filter("abcdefghijklmnopqrstuvwxyzäöüß<> ".contains(_))
    ).map(word => if (word.contains("@")) " <<<tokentwitterusername>>> " else word)
      .filterNot(_.contains("@")).flatMap(_.split(" ")).filterNot(x => x.isEmpty || x.length < 2)

  }

}
