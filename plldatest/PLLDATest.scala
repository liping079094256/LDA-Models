package plldatest
import scalanlp.stage.text._
import scalanlp.text.tokenize._
import scalanlp.pipes.Pipes.global._
import edu.stanford.nlp.tmt.stage._
import edu.stanford.nlp.tmt.model.lda._
import edu.stanford.nlp.tmt.model.llda._
import edu.stanford.nlp.tmt.model.plda._
import scalanlp.io.CSVFile
import scalanlp.stage.IDColumn
import scalanlp.stage.Column
import scalanlp.stage.Take
import scalanlp.stage.Drop$
import scalanlp.stage.Drop
import utils.ArraySort
import java.util.ArrayList
import scala.actors.threadpool.Arrays
import java.io.PrintWriter
import java.io.File
import java.util.Date
import scala.util.control.Breaks
import scala.util.control.BreakControl
class PLLDATest(filePath: String, percentTrain: Double, numOfTopicPerLabel: Int) {
	var testDocDistribution = Array[Array[Double]]();
	var topicTermDistribution = Array[Array[Double]]();
	var perplexity = 0.0;
	var testDocs = Array[Array[Int]]();
	var correctClassifyInstance = 0.0
	var oneError=0.0;
	var averageTime=0.0;
	var coverage=0.0;
	def getNumTopicPerLabel() = {
		this.numOfTopicPerLabel;
	}
	
	//def get
	def maining() {
		var source = CSVFile(filePath) ~> IDColumn(1);

		var tokenizer = {
			SimpleEnglishTokenizer() ~> // tokenize on space and punctuation
				CaseFolder() ~> // lowercase everything
				WordsAndNumbersOnlyFilter() ~> // ignore non-words and non-numbers
				MinimumLengthFilter(3) // take terms with >=3 characters
		}

		var text = {
			source ~> // read from the source file
				Column(3) ~> // select column containing text
				TokenizeWith(tokenizer) ~> // tokenize with tokenizer above
				TermCounter() ~> // collect counts (needed below)
				TermMinimumDocumentCountFilter(1) ~> // filter terms in <4 docs
				TermDynamicStopListFilter(300) ~> // filter out 30 most common terms
				DocumentMinimumLengthFilter(5) // take only docs with >=5 terms
		}

		// define fields from the dataset we are going to slice against
		var labels = {
			source ~> // read from the source file
				Column(2) ~> // take column two, the year
				TokenizeWith(WhitespaceTokenizer()) ~> // turns label field into an array
				TermCounter() ~> // collect label counts
				TermMinimumDocumentCountFilter(5) // filter labels in < 10 docs
		}

		var numTrain = text.data.size * this.percentTrain
		println(numTrain)
		var dataset = LabeledLDADataset(text ~> Take(numTrain.intValue()), labels ~> Take(numTrain.intValue()));
		
		var testing = LabeledLDADataset(text ~> Drop(numTrain.intValue()-100), labels ~> Drop(numTrain.intValue()-100));

		
		testing.iterator.foreach(c=>{
			println(c.id+"");
		})
		dataset.iterator.foreach(c=>{
			println(c.id+"");
		})
		
		
		
		
		// define the model parameters
		val numBackgroundTopics = 0;

		val numTopicsPerLabel = SharedKTopicsPerLabel(numOfTopicPerLabel);
		// or could specify the number of topics per label based on the values
		// in a two-column CSV file containing label name and number of topics

		val modelParams = PLDAModelParams(dataset,
			numBackgroundTopics, numTopicsPerLabel,
			termSmoothing = 0.01, topicSmoothing = 0.01);

		// Name of the output model folder to generate
		val modelPath = file("C:\\Downloads\\plda-cvb0-" + dataset.signature + "-" + modelParams.signature);

		// Trains the model, writing to the given output path
		var date1=new Date();
		var m = TrainCVB0PLDA(modelParams, dataset, output = modelPath, maxIterations = 1000);
		var date2=new Date();
		this.averageTime=(date2.getTime()-date1.getTime())*1.0/1000/1000;
		
		testDocDistribution = new Array[Array[Double]](testing.iterator.size)
		(0 until testDocDistribution.size).foreach(c => {
			testDocDistribution(c) = new Array[Double](modelParams.numLabels * numOfTopicPerLabel);
		})

		var counting = 0;
		var numTotal = 0;
		var count = 0;

		println("latent topics:" + modelParams.numLabels * numOfTopicPerLabel);
		testing.iterator.foreach(doc => {
			numTotal += doc.labels.size;
			var sample = doc
			var array = new Array[Int](modelParams.numLabels);
			(0 until array.size).foreach(c => {
				array(c) = c;
			});
			var a = new LabeledLDADocumentParams("llllllllll", array, sample.terms);
			var t = m.infer(a);
			(0 until modelParams.numLabels * numOfTopicPerLabel).foreach(c => {
				testDocDistribution(count)(c) = t(c)
			})

			var labelArray = new Array[Double](modelParams.numLabels);
			(0 until labelArray.size).foreach(c => {
				(c * numOfTopicPerLabel until (c + 1) * numOfTopicPerLabel).foreach(e => {
					labelArray(c) += t(e)
				})
			})
			
			////compute percentage of correct classify!
			var order = new Array[String](modelParams.numLabels);
			var aCount = 0;
			(0 until modelParams.numLabels).foreach(aValue => {
				order(aCount) = aValue.toString();
				aCount += 1;
			})
			var sort = new ArraySort();
			var value = sort.sortArray(order, labelArray)
			var list = new ArrayList[String]();
			(0 until doc.labels.size).foreach(c => {
				list.add(value(c).s);
			})
			(0 until doc.labels.size).foreach(c => {
				if (list.contains(doc.labels(c).toString())) {
					counting += 1;
				}
				else {
				}
			});
			
			////compute one-Error loss
			var labelList=new ArrayList[String]();
			(0 until doc.labels.size).foreach(c=>{
				labelList.add(doc.labels(c).toString());
			})
			if(!labelList.contains(value(0).s))
			{
				this.oneError+=1;
			}
			//compute the coverage
			aCount=0;
			var cCount=0;
			////var i=0
			var ifBreak=false;
			while(aCount<value.size&&ifBreak==false) {
				if(labelList.contains(value(aCount).s)){
					cCount+=1;
				}
				if(cCount==doc.labels.size)
				{
					this.coverage+=aCount;
					ifBreak=true;
				}
				aCount+=1;
			}
			
			count += 1;
		})
        this.coverage=this.coverage/numTotal-1;
		
		println(counting);
		this.oneError/=count;
		
		this.correctClassifyInstance = counting * (1.0) / numTotal
		println(this.correctClassifyInstance)
		//m.  

		this.topicTermDistribution = new Array[Array[Double]](modelParams.numLabels * numOfTopicPerLabel);
		(0 until topicTermDistribution.size).foreach(c => {
			topicTermDistribution(c) = new Array[Double](modelParams.numTerms);
			topicTermDistribution(c) = m.getTopicTermDistribution(c);
			/////println(Arrays.toString(topicTermDistribution(c)));
		})
		//normalized topic term distribution
		(0 until topicTermDistribution.size).foreach(c => {
			var count = 0.0;
			(0 until topicTermDistribution(c).size).foreach(e => {
				count += topicTermDistribution(c)(e);
			})
			(0 until topicTermDistribution(c).size).foreach(e => {
				topicTermDistribution(c)(e) = topicTermDistribution(c)(e) / count;
			})

		})
		this.testDocs = new Array[Array[Int]](this.testDocDistribution.size);
		var no = 0;
		testing.iterator.foreach(doc => {
			testDocs(no) = doc.terms;
			no += 1;
		})
		this.computePerplexityTopic();

		println("perplexity:" + this.perplexity);
	}

	def computePerplexityTopic() {
		(0 until this.testDocDistribution.size).foreach(c => {
			var count = 0.0;
			(0 until this.testDocDistribution(c).size).foreach(e => {
				count += this.testDocDistribution(c)(e)
			})

			(0 until this.testDocDistribution(c).size).foreach(e => {
				this.testDocDistribution(c)(e) = this.testDocDistribution(c)(e) / count;
			})

		})
		var sumOfLog = 0.0;
		var sumOfLength = 0.0;
		(0 until this.testDocDistribution.size).foreach(c => {
			sumOfLog += computeLogTestDocument(c);
			sumOfLength += this.testDocs(c).size;
		})
		this.perplexity = Math.exp(-1 * sumOfLog / sumOfLength);
	}

	def computeLogTestDocument(docId: Int) = {
		var result = 0.0;
		(0 until this.testDocs(docId).size).foreach(c => {
			var r = 0.0;
			(0 until this.topicTermDistribution.size).foreach(topic => {
				r += this.topicTermDistribution(topic)(this.testDocs(docId)(c)) *
					this.testDocDistribution(docId)(topic)
			})
			result += Math.log(r);
		})
		result;
	}
}

object TestPLLDA {
	def main(args: Array[String]) {
		var pw = new PrintWriter(new File("c:\\1.txt"));
		(1 until 15).foreach(c => {
			var test = new PLLDATest("E:\\试验数据\\enron\\enron.csv", 0.8, c);
			test.maining();
			pw.append(test.getNumTopicPerLabel() + ", ");
			pw.append(test.perplexity + ", ");	
			pw.append(test.coverage+",")
			pw.append(test.oneError+",");
			pw.append(test.correctClassifyInstance + ", ");
			pw.append(test.averageTime+"\n");
			pw.flush();
		});

	}
}