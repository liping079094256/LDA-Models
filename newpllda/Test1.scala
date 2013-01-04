package newpllda
import java.util.Date
import utils.ArraySort
import java.util.Arrays
import java.io.PrintWriter
import java.io.File

object Test1 {

	def main(args: Array[String]): Unit = {

		var alpha = 0.01
		var alphaL = 0.01;
		var eta = 0.01;
		var topicNumOfEachLabel = 6
		var globalTopicNum = 6;
		var filePath = "E:\\试验数据\\enron\\enron.csv"
		var percentForTest = 0.2
		var percentOfGlobalTopic = 0.3
		var trainIter = 100;
		var testIter = 100;
		
		var w=new PrintWriter(new File("c:\\1.txt"));
		/////w.append("dfdfd");
		(8 until 20).foreach(c=>{
			///percentOfGlobalTopic=0.5*globalTopicNum/(c+globalTopicNum)
			var lda = new NewPLLDA(alpha, alphaL, eta, c,
				globalTopicNum, filePath, percentForTest,
				percentOfGlobalTopic, trainIter, testIter);
			lda.initParameters();
			lda.initCountParameters();
			lda.initCountParametersForTestDocs();
			var date1 = new Date();
			lda.sampingTrainDocs();
			var date2 = new Date();
			lda.computePhi();

			////lda.computePhiValueForLabels();
			lda.sampingTestDocs();
			lda.computeTopicTheta();
		    lda.computePerplexityTopic();
		    lda.computeLabelThetaOfTestDoc();
			//		println("perplexity:"+lda.perplexity)
			lda.computeNumOfCorrectClassify();
			w.append(c+",")
			w.append(percentOfGlobalTopic+",")
			w.append(lda.perplexityOfTopics+",");
			w.append(lda.converage+",")
			w.append(lda.oneError+",")
			w.append(lda.numOfCorrect+",");
			w.append((date2.getTime()-date1.getTime())*1.0/1000/1000+"\n");
			w.flush();
		})
		
	}

}