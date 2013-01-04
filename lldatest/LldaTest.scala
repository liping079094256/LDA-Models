package lldatest
import scalanlp.io._
import scalanlp.stage._
import scalanlp.stage.text._
import scalanlp.text.tokenize._
import scalanlp.pipes.Pipes.global._
import edu.stanford.nlp.tmt.stage._
import edu.stanford.nlp.tmt.model.lda._
import edu.stanford.nlp.tmt.model.llda._
import java.util.Arrays
import scalanlp.util.Index
import utils.ArraySort
import java.util.ArrayList
import java.util.Date

object LldaTest {

	def main(args: Array[String]): Unit = {
		val source = CSVFile("E:\\试验数据\\bibtex\\bibtex.csv") ~> IDColumn(1);

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

		// define fields from the dataset we are going to slice against
		val labels = {
			source ~> // read from the source file
				Column(2) ~> // take column two, the year
				TokenizeWith(WhitespaceTokenizer()) ~> // turns label field into an array
				TermCounter() ~> // collect label counts
				TermMinimumDocumentCountFilter(10) // filter labels in < 10 docs
		}

		val numTrain = text.data.size * 80 / 100;
		println("total:"+text.data.size);
		var dataset = LabeledLDADataset(text ~> Take(numTrain), labels ~> Take(numTrain));
		println("train:"+dataset.iterator.size);
		var testing = LabeledLDADataset(text ~> Drop(numTrain), labels ~> Drop(numTrain));
		println("test:"+testing.iterator.size)
		//        println("*******************test dataset*****************************")
		//		dataset.iterator.foreach(dp=>{
		//			println("document:"+Arrays.toString(dp.terms))
		//			println("labels:"+Arrays.toString(dp.labels))
		//			
		//		})
		// define the model parameters
		val modelParams = LabeledLDAModelParams(dataset);
		//        var i=modelParams.labelIndex.get
		//        println("output labels index:")
		//        i.foreach(k=>{
		//        	println(k);
		//        })
		//        println(modelParams.termIndex)
		//        var index=modelParams.termIndex
		//        var count=0;
		//        index.get.foreach(i=>count+=1)
		//        println("vocabulary number:"+index.get.size)
		//        println()
		// Name of the output model folder to generate
		val modelPath = file("C:\\Downloads\\llda-cvb0-" + dataset.signature + "-" + modelParams.signature);

		// Trains the model, writing to the given output path
		var date1=new Date();
		var iter=10
		var m = TrainCVB0LabeledLDA(modelParams, dataset, output = modelPath, iter);
		// or could use TrainGibbsLabeledLDA(modelParams, dataset, output = modelPath, maxIterations = 1500);
		var date2=new Date();
		var counting=0;
		var numTotal=0;
		
			testing.iterator.foreach(doc => {
			numTotal+=doc.labels.size;
			var sample = doc
			var array = new Array[Int](modelParams.numLabels);
			(0 until array.size).foreach(c=>{
			  array(c)=c;
			});
			var a = new LabeledLDADocumentParams("llllllllll", array, sample.terms);
			var t = m.infer(a);
			
			var order=new Array[String](array.size);
			var aCount=0;
			array.foreach(aValue=>{
				order(aCount)=aValue.toString();
				aCount+=1;
			})
			var targetValue=new Array[Double](array.size);
			aCount=0;
			t.iterator.foreach(k => {
				targetValue(aCount)=k._2;
				aCount+=1;
			})
			var sort=new ArraySort();
			var value=sort.sortArray(order,targetValue)
			var list=new ArrayList[String]();
			(0 until doc.labels.size).foreach(c=>{
				list.add(value(c).s);
			})
//			println(list);
//			println("aaa"+Arrays.toString(doc.labels));
			var ifContains=true;
			(0 until doc.labels.size).foreach(c=>{
				
				if(list.contains(doc.labels(c).toString())){
					counting+=1;
				}
				else
				{
					//ifContains=false;
					
				}
			});
			
		})
		
		var numOfTerms=modelParams.numTerms;
	    ///var numOfTopic=modelParams.numTopics;
			
		var trainDocNum=dataset.iterator.size;
		var trainDocTotalTermNum=0;
		dataset.iterator.foreach(doc=>{
			trainDocTotalTermNum+=doc.terms.size;
		});
		
		var testDocNum=testing.iterator.size;
		var testDocTermsNum=0;
		testing.iterator.foreach(doc=>{
			testDocTermsNum+=doc.terms.size;
		});
		
		
		println("模型测试完成！");
			println("*****************************************************************");
			//total document 
			print("文档总数："+(trainDocNum+testDocNum)+"\t\t");
			print("总词数："+(trainDocTotalTermNum+testDocTermsNum)+"\t\t");
			print("总标签数："+"\t\t\n");
			print("词汇表长度："+numOfTerms+"\t\t");
			print("平均文档长度："+Math.round((trainDocTotalTermNum+testDocTermsNum)*1.0/(trainDocNum+testDocNum))+"\t\t");
			print("平均文档标记数"+"\t\t");
			println();
			println();
			
			 ///////////train document
			print("训练样本总数："+trainDocNum+"\t");
			print("总词数："+trainDocTotalTermNum+"\t\t");
			print("总标记数:"+"\t\t\n");
			print("平均文档长度:"+Math.round(trainDocTotalTermNum*1.0/trainDocNum)+"\t\t");
			//println();
			
			print("平均文档标记数："+"\t\t");
			print("模型训练时间:"+(date2.getTime()-date1.getTime())*1.0/1000.0+"s\t\t\n");
			print("平均每次迭代时间："+(date2.getTime()-date1.getTime())*1.0/1000.0/iter+"s\t\t");
			print("迭代次数："+iter+"\t\t");
			println();
			println();
			
			
			////////////////////testing document
			print("测试样本总数："+testDocNum+"\t");
			print("总词数："+testDocTermsNum+"\t\t");
			print("总标记数:"+"\t\n");
			print("平均文档长度:"+Math.round(testDocTermsNum*1.0/testDocNum)+"\t\t");
			print("正确分类的标签数："+counting+"\t");
			print("正确分类率："+counting*(1.0)/numTotal);
			////print("perplexity："+perplexity+"");
			println()

			//var k=Math.round(D);
			////println("[perplexity] perplexity at " + numTopics + " topics: " + perplexity);
			
			println("*****************************************************************");

//		println(counting);
//		println(counting*(1.0)/numTotal)
		//var target
	}

}