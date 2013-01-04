package ldatest
import scalanlp.io._
import scalanlp.stage._
import scalanlp.stage.text._
import scalanlp.text.tokenize._
import scalanlp.pipes.Pipes.global._
import edu.stanford.nlp.tmt.stage._
import edu.stanford.nlp.tmt.model.lda._
import edu.stanford.nlp.tmt.model.llda._
import java.util.Date
import java.lang.Math
import java.io.PrintWriter
object ComputePerplexity {

	def main(args: Array[String]) {
		val source = CSVFile("E:\\试验数据\\enron\\enron.csv") ~> IDColumn(1);

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
				TermMinimumDocumentCountFilter(4) ~> // filter terms in <4 docs
				TermDynamicStopListFilter(30) ~> // filter out 30 most common terms
				DocumentMinimumLengthFilter(5) // take only docs with >=5 terms
		}

		// set aside 80 percent of the input text as training data ...
		val numTrain = text.data.size * 80 / 100;

		println(text.data.size);
		// build a training dataset
		val training = LDADataset(text ~> Take(numTrain));
		println(training.iterator.size);
		var trainDocNum = training.iterator.size;
		var trainDocTotalTermNum = 0;
		///var trainDocTotalLabelNum=0;

		training.iterator.foreach(doc => {
			trainDocTotalTermNum += doc.terms.size;
		});

		// build a test dataset, using term index from the training dataset 
		val testing = LDADataset(text ~> Drop(numTrain));
		println(testing.iterator.size);
		var testDocNum = testing.iterator.size;
		var testDocTermsNum = 0;
		testing.iterator.foreach(doc => {
			testDocTermsNum += doc.terms.size;
		});

		// a list of pairs of (number of topics, perplexity)
		/////var scores = List.empty[(Int, Double)];
		var printer=new PrintWriter("c:\\LDA.txt");
         var num=10
		(1 to 15).foreach(c => {

			var numTopics = c*num
			val params = LDAModelParams(numTopics = numTopics, dataset = training);
			val output = file("C:\\Downloads\\lda-" + training.signature + "-" + params.signature);

			var numOfTerms = params.numTerms;
			var numOfTopic = params.numTopics;

			var date1 = new Date();
			var iter = 10
			val model = TrainCVB0LDA(params, training, output = null, iter);
			var date2 = new Date();

			val perplexity = model.computePerplexity(testing);

			printer.append(c+",");
			printer.append(numTopics+",");
			printer.append(perplexity+",");
			printer.append((date2.getTime() - date1.getTime()) * 1.0 / 1000.0 / iter+",");
			printer.append("\n");
			
			
			
			
			
			println("模型测试完成！");
			println("*****************************************************************");
			//total document 
			print("文档总数：" + (trainDocNum + testDocNum) + "\t\t");
			print("总词数：" + (trainDocTotalTermNum + testDocTermsNum) + "\t\t");
			print("总标签数：" + "\t\t\n");
			print("词汇表长度：" + numOfTerms + "\t\t");
			print("平均文档长度：" + Math.round((trainDocTotalTermNum + testDocTermsNum) * 1.0 / (trainDocNum + testDocNum)) + "\t\t");
			print("平均文档标记数" + "\t\t");
			println();
			println();

			///////////train document
			print("训练样本总数：" + trainDocNum + "\t");
			print("总词数：" + trainDocTotalTermNum + "\t\t");
			print("总标记数:" + "\t\t\n");
			print("平均文档长度:" + Math.round(trainDocTotalTermNum * 1.0 / trainDocNum) + "\t\t");
			//println();

			print("平均文档标记数：" + "\t\t");
			print("模型训练时间:" + (date2.getTime() - date1.getTime()) * 1.0 / 1000.0 + "s\t\t\n");
			print("平均每次迭代时间：" + (date2.getTime() - date1.getTime()) * 1.0 / 1000.0 / iter + "s\t\t");
			print("迭代次数：" + iter + "\t\t");
			println();
			println();

			////////////////////testing document
			print("测试样本总数：" + testDocNum + "\t");
			print("总词数：" + testDocTermsNum + "\t\t");
			print("总标记数:" + "\t\n");
			print("平均文档长度:" + Math.round(testDocTermsNum * 1.0 / testDocNum) + "\t\t");

			print("perplexity：" + perplexity + "");
			println()

			//var k=Math.round(D);
			////println("[perplexity] perplexity at " + numTopics + " topics: " + perplexity);

			println("*****************************************************************");

			/////scores :+= (numTopics, perplexity);

		})
		// loop over various numbers of topics, training and evaluating each model

	}

}

