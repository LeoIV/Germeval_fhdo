package de.hoevelmann.bachelorthesis.worksheets

/**
  * Created by Leonard Hövelmann (leonard.hoevelmann@adesso.de) on 15.08.2017.
  */
case class EnglishReview(reviewID: String, asin: String, reviewerName: String, helpful: Array[Int], reviewText: String, overall: Int, summary: String, unixReviewTime: Long, reviewTime: String)