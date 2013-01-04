package ldatest
import scalanlp.io._
import scalanlp.stage._
import scalanlp.stage.text._
import scalanlp.text.tokenize._
import scalanlp.pipes.Pipes.global._
import edu.stanford.nlp.tmt.stage._
import edu.stanford.nlp.tmt.model.lda._
import edu.stanford.nlp.tmt.model.llda._;
import java.util.Arrays

object ComputePer {

	def main(args: Array[String]) {
		///val source = CSVFile("E:\\试验数据\\20_newsgroups\\2.csv") ~> IDColumn(1);

		val source = CSVFile("E:\\试验数据\\20_newsgroups\\2.csv") ~> IDColumn(1);

		val tokenizer = {
			SimpleEnglishTokenizer() ~> // tokenize on space and punctuation
				CaseFolder() ~> // lowercase everything
				WordsAndNumbersOnlyFilter() ~> // ignore non-words and non-numbers
				MinimumLengthFilter(3) // take terms with >=3 characters
		}

		val text = {
			source ~> // read from the source file
				Column(3) ~> // select column containing text
				TokenizeWith(tokenizer) ~> // tokenize with tokenizer above
				TermCounter() ~> // collect counts (needed below)
				TermMinimumDocumentCountFilter(20) ~> // filter terms in <4 docs
				TermDynamicStopListFilter(30) ~> // filter out 30 most common terms
				DocumentMinimumLengthFilter(5) // take only docs with >=5 terms
		}

		// define fields from the dataset we are going to slice against
		val labels = {
			source ~> // read from the source file
				Column(2) ~> // take column two, the year
				TokenizeWith(WhitespaceTokenizer()) ~> // turns label field into an array
				TermCounter() ~> // collect label counts
				TermMinimumDocumentCountFilter(10) // filter labels in < 10 docs
		}

		// set aside 80 percent of the input text as training data ...
		val numTrain = text.data.size * 80 / 100;

		// build a training dataset
		val training = LabeledLDADataset(text ~> Take(numTrain), labels ~> Take(numTrain));
		////LDADataset

		// build a test dataset, using term index from the training dataset 
		val testing = LabeledLDADataset(text ~> Drop(numTrain), labels ~> Drop(numTrain));
		///LDADataset(text ~> Drop(numTrain));

		// a list of pairs of (number of topics, perplexity)
		/////var scores = List.empty[(Int, Double)];

		// loop over various numbers of topics, training and evaluating each model
		var numTopics = 70
		val modelParams = LabeledLDAModelParams(training);
		val modelPath = file("C:\\Downloads\\llda-cvb0-" + training.signature + "-" + modelParams.signature);
		val model = TrainCVB0LabeledLDA(modelParams, training, output = modelPath, maxIterations = 100);

		testing.iterator.foreach(doc=>{
			
		})
		var t = model.infer(testing.iterator.next());

		t.iterator.foreach(k => {
			println(k._1 + ":" + k._2);
		})

		//			println("[perplexity] computing at " + numTopics);
		//
		//			val perplexity = model
		//
		//			println("[perplexity] perplexity at " + numTopics + " topics: " + perplexity);

		/////scores :+= (numTopics, perplexity);
	}

}

