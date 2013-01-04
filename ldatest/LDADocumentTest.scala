package test
import edu.stanford.nlp.tmt.model.lda.GibbsLDADocument
import edu.stanford.nlp.tmt.model.lda.GibbsLDADocument$
import edu.stanford.nlp.tmt.model.lda.LDADataset
import scalanlp.io._
import scalanlp.stage._
import scalanlp.stage.text._
import scalanlp.text.tokenize._
import scalanlp.pipes.Pipes.global._
import edu.stanford.nlp.tmt.stage._
import edu.stanford.nlp.tmt.model.lda._
import edu.stanford.nlp.tmt.model.llda._;

object LDADocumentTest {
	def main(args: Array[String]) {
		val source = CSVFile("C:\\Downloads\\34234.csv") ~> IDColumn(1);

		val tokenizer = {
			SimpleEnglishTokenizer() ~> // tokenize on space and punctuation
				CaseFolder() ~> // lowercase everything
				WordsAndNumbersOnlyFilter() ~> // ignore non-words and non-numbers
				MinimumLengthFilter(0) // take terms with >=3 characters
		}

		val text = {
			source ~> // read from the source file
				Column(2) ~> // select column containing text
				TokenizeWith(tokenizer) ~> // tokenize with tokenizer above
				TermCounter() ~> // collect counts (needed below)
				TermMinimumDocumentCountFilter(0) ~> // filter terms in <4 docs
				TermDynamicStopListFilter(30) ~> // filter out 30 most common terms
				DocumentMinimumLengthFilter(1) // take only docs with >=5 terms
		}

		// display information about the loaded dataset
		println("***********************dataset test code!!!!!!!******************")
		val dataset = LDADataset(text);
		println(dataset.size);
		var i = dataset.iterator
		while (i.hasNext) {
			var d2 = i.next()
			println("id-" + d2.id);
			println("termSize:" + d2.terms.size);
			print("terms:");
			d2.terms.foreach(k => print(k + "-"))
			println("")
		}
		println("*********************params test code!*********************");
		val params = LDAModelParams(numTopics = 30, dataset = dataset,
			topicSmoothing = 0.01, termSmoothing = 0.01);
		println(params.numTerms)
		println(params.termIndex);
		
		println("*********************params test code!*********************");
		// Name of the output model folder to generate
		val modelPath = file("lda-" + dataset.signature + "-" + params.signature);
        
		// Trains the model: the model (and intermediate models) are written to the
		// output folder.  If a partially trained model with the same dataset and
		// parameters exists in that folder, training will be resumed.
		TrainCVB0LDA(params, dataset, output = modelPath, maxIterations = 1000);

	}
}