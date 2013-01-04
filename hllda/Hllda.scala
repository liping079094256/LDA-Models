package hllda
import java.io.File
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
import java.util.ArrayList
import scala.util.Random
import java.util.HashMap
import utils.ArraySort
class Hllda(filePath: String, alpha: Double, eta: Double, percentForTest: Double) {
	/**
	 *
	 */
	var CSVFilePath = "E:\\试验数据\\20_newsgroups\\2.csv"

	var treeLevels = 0
	var treePath = Set[TreeNode]();
	///var documentNumber = 0;
	var tree: TreeNode = null
	var dataFolder: File = null
	//var termNumber=0;
	//var labelNumber=0;

	//初始化 根节点！！！！！！！！！！！！ 
	this.tree = new TreeNode()
	this.tree.nodeName = "rootNode"
	this.tree.parent = null

	val source = CSVFile(this.CSVFilePath) ~> IDColumn(1);

	val tokenizer = {
		SimpleEnglishTokenizer() ~>
			CaseFolder() ~>
			WordsAndNumbersOnlyFilter() ~> // ignore non-words and non-numbers
			MinimumLengthFilter(3) // take terms with >=3 characters
	}

	val text = {
		source ~>
			Column(3) ~> // select column containing text
			TokenizeWith(tokenizer) ~> // tokenize with tokenizer above
			TermCounter() ~> // collect counts (needed below)
			TermMinimumDocumentCountFilter(20) ~> // filter terms in <4 docs
			TermDynamicStopListFilter(80) ~> // filter out 30 most common terms
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

	val dataset = LabeledLDADataset(text, labels);
	var phiValueMap = new HashMap[Int, Array[Double]]();
	// define the model parameters
	val modelParams = LabeledLDAModelParams(dataset);
	var totalDocument = this.dataset.iterator.size
	val storeEachDocument = new Array[Array[Int]]((this.totalDocument * (1 - this.percentForTest)).toInt)
	val storeLabelsDocument = new Array[Array[Int]]((this.totalDocument * (1 - this.percentForTest)).toInt)
	val testDocuments = new Array[Array[Int]](this.totalDocument - this.storeEachDocument.size)
	val testLabelsDocument = new Array[Array[Int]](this.totalDocument - this.storeEachDocument.size)
	var docNumberCount = 0;
	var testDocNumberCount = 0;

	this.dataset.iterator.foreach(doc => {
		if (docNumberCount < this.storeEachDocument.size) {
			storeEachDocument(docNumberCount) = doc.terms
			//println(doc.id)
			storeLabelsDocument(docNumberCount) = doc.labels
			docNumberCount += 1
		}
		else {
			this.testDocuments(this.testDocNumberCount) = doc.terms
			this.testLabelsDocument(this.testDocNumberCount) = doc.labels
			this.testDocNumberCount += 1
		}
	})
	var testThetaDocument = new Array[Array[Double]](this.testDocNumberCount)

	dataFolder = new File(filePath);
	this.scanFolder(this.tree, dataFolder);

	/**
	 *
	 * init tree path Array
	 */
	val treePathArray = new Array[Array[TreeNode]](this.treePath.size)
	var treePathCount = 0
	this.treePath.iterator.foreach(node => {
		treePathArray(treePathCount) = this.initTreePathArray(node)
		this.treePathCount += 1
	})

	var docTermAssignment = new Array[Array[Int]](this.storeEachDocument.size)
	(0 until this.docNumberCount).foreach(i => {
		this.docTermAssignment(i) = new Array[Int](this.storeEachDocument(i).size)
		var eachTermCount = 0
		this.storeEachDocument(i).foreach(term => {
			this.docTermAssignment(i)(eachTermCount) = Random.nextInt(this.storeLabelsDocument(i).size - 1)
		})
	})
	var testDocTermAssignment = new Array[Array[Int]](this.testDocNumberCount)
	(0 until this.testDocNumberCount).foreach(i => {
		this.testDocTermAssignment(i) = new Array[Int](this.testDocuments(i).size)
		var eachTermCount = 0
		this.testDocuments(i).foreach(term => {
			this.testDocTermAssignment(i)(eachTermCount) = Random.nextInt(this.testLabelsDocument(i).size - 1)
		})
	})
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///////////////////////Sample code!!!!!!!!!!!///////////////////////////////
	def sample() {
		var docCount = 0;
		this.storeEachDocument.foreach(doc => {
			var terms = doc
			var labels = this.storeLabelsDocument(docCount)
			var treeArray = Array[TreeNode]()
			(0 until treePathArray.size).foreach(i => {
				if (treePathArray(i)(0).labelIndex == labels(labels.size - 1)) {
					treeArray = treePathArray(i)
				}
			})
			this.sampleDocument(docCount, terms, treeArray)

			docCount += 1
		})
	}

	def sampleDocument(docNo: Int, docArray: Array[Int], nodeArray: Array[TreeNode]) = {

		var sampleTermCount = new Array[Array[Int]](nodeArray.size)
		var sampleDocTermCount = new Array[Int](nodeArray.size)
		var iCount = 0
		nodeArray.foreach(node => {
			sampleTermCount(iCount) = node.termCount
			sampleDocTermCount(iCount) = node.documentCount(docNo)
			iCount += 1
		})
		////sampling Document
		var iTermCount = 0
		docArray.foreach(term => {
			var assignTopic = this.docTermAssignment(docNo)(iTermCount)
			var topic = nodeArray(assignTopic)
			topic.documentCount(docNo) -= 1
			///topic.documentCount(docNo) -= 1
			///topic.count -= 1
			topic.termCount(term) -= 1

			var dji = new Array[Int](nodeArray.size)
			var wji = new Array[Int](nodeArray.size)

			(0 until nodeArray.size).foreach(nodeCount => {
				dji(nodeCount) = nodeArray(nodeCount).documentCount(docNo)
				wji(nodeCount) = nodeArray(nodeCount).termCount(term)
			})
			///compute topic distribution
			var topicDistribution = new Array[Double](nodeArray.size)
			var topicCount: Double = 0.0;
			(0 until nodeArray.size).foreach(nodeCount => {
				topicDistribution(nodeCount) = (dji(nodeCount) + this.alpha) * (wji(nodeCount) + this.eta)
				topicCount += topicDistribution(nodeCount)
			})
			///normalizing the topic distribution
			var topicCount2: Double = 0.0
			var topicDis = new Array[Double](nodeArray.size)
			(0 until nodeArray.size).foreach(nodeCount => {
				topicDistribution(nodeCount) = topicDistribution(nodeCount) / topicCount
				topicCount2 += topicDistribution(nodeCount)
				topicDis(nodeCount) = topicCount2
			})

			var randomNo = Random.nextDouble()
			var selectedTopic = -1
			(0 until nodeArray.size).foreach(nodeCount => {
				if (randomNo <= topicDis(nodeCount) && selectedTopic == -1) {
					selectedTopic = nodeCount
				}
			})
			if (selectedTopic == -1) {
				selectedTopic = nodeArray.size - 1
			}
			//println("select topic :"+selectedTopic)
			this.docTermAssignment(docNo)(iTermCount) = selectedTopic
			sampleTermCount(selectedTopic)(term) += 1;
			sampleDocTermCount(selectedTopic) += 1
			//nodeArray(selectedTopic).documentCount(docNo) += 1
			//nodeArray(selectedTopic).count += 1

			iTermCount += 1
		})

	}

	def computePhiValue() {
		treePath.foreach(node => {
			this.computeEachNodePhy(node)
		})
	}

	def computeEachNodePhy(node: TreeNode) {
		var totalCount = 0.0
		node.termCount.foreach(term => {
			totalCount += term
		})
		var arr = new Array[Double](node.termCount.size)

		var id = 0
		node.termCount.foreach(term => {
			arr(id) = (term + this.eta) / (totalCount + this.eta * arr.size)

			id += 1
		})
		this.phiValueMap.put(node.labelIndex, arr)

		if (node.parent != null) {
			this.computeEachNodePhy(node.parent)
		}

	}

	///////////////////////Sample code!!!!!!!!!!!///////////////////////////////

	//////////////////////////////Sample test set code!!!!!!!!////////////////////////
	def sampleTestSet() {
		var docCount = 0;
		this.testDocuments.foreach(doc => {
			var terms = doc
			var labels = this.testLabelsDocument(docCount)
			var treeArray = Array[TreeNode]()
			(0 until treePathArray.size).foreach(i => {
				if (treePathArray(i)(0).labelIndex == labels(labels.size - 1)) {
					treeArray = treePathArray(i)
				}
			})
			this.sampleTestDocument(docCount, terms, treeArray)

			docCount += 1
		})
	}

	def sampleTestDocument(docNo: Int, docArray: Array[Int], nodeArray: Array[TreeNode]) {

		//var sampleTermCount = new Array[Array[Int]](nodeArray.size)
		var sampleDocTermCount = new Array[Int](nodeArray.size)
		var iCount = 0
		nodeArray.foreach(node => {
			//sampleTermCount(iCount) = node.termCount
			sampleDocTermCount(iCount) = node.testDocumentCount(docNo)
			iCount += 1
		})
		////sampling Document
		var iTermCount = 0
		docArray.foreach(term => {
			var assignTopic = this.testDocTermAssignment(docNo)(iTermCount)
			var topic = nodeArray(assignTopic)
			topic.testDocumentCount(docNo) -= 1
			///topic.documentCount(docNo) -= 1
			///topic.count -= 1
			///topic.termCount(term) -= 1
           
			var dji = new Array[Int](nodeArray.size)
			var wji = new Array[Double](nodeArray.size)

			(0 until nodeArray.size).foreach(nodeCount => {
				dji(nodeCount) = nodeArray(nodeCount).testDocumentCount(docNo)
				wji(nodeCount) = this.phiValueMap.get(nodeArray(nodeCount).labelIndex)(term)
			})
			///compute topic distribution
			var topicDistribution = new Array[Double](nodeArray.size)
			var topicCount: Double = 0.0;
			(0 until nodeArray.size).foreach(nodeCount => {
				topicDistribution(nodeCount) = (dji(nodeCount) + this.alpha) * wji(nodeCount)
				topicCount += topicDistribution(nodeCount)
			})
			///normalizing the topic distribution
			var topicCount2: Double = 0.0
			var topicDis = new Array[Double](nodeArray.size)
			(0 until nodeArray.size).foreach(nodeCount => {
				topicDistribution(nodeCount) = topicDistribution(nodeCount) / topicCount
				topicCount2 += topicDistribution(nodeCount)
				topicDis(nodeCount) = topicCount2
			})

			var randomNo = Random.nextDouble()
			var selectedTopic = -1
			(0 until nodeArray.size).foreach(nodeCount => {
				if (randomNo <= topicDis(nodeCount) && selectedTopic == -1) {
					selectedTopic = nodeCount
				}
			})
			if (selectedTopic == -1) {
				selectedTopic = nodeArray.size - 1
			}
			//println("select topic :"+selectedTopic)
			this.testDocTermAssignment(docNo)(iTermCount) = selectedTopic
			//sampleTermCount(selectedTopic)(term) += 1;
			//sampleDocTermCount(selectedTopic) += 1
			//nodeArray(selectedTopic).documentCount(docNo) += 1
			//nodeArray(selectedTopic).count += 1
            nodeArray(selectedTopic).testDocumentCount(docNo)+=1
			iTermCount += 1
		})
	}

	def computeTheta() {
		var docNo=0
		this.testDocuments.foreach(doc=>{
			var labels=this.testLabelsDocument(docNo)
			var treeArray = Array[TreeNode]()
			(0 until treePathArray.size).foreach(i => {
				if (treePathArray(i)(0).labelIndex == labels(labels.size - 1)) {
					treeArray = treePathArray(i)
				}
			})
			this.computeEachDocumentTheta(docNo,doc,treeArray)
			docNo+=1;
		})

	}
	
	def computeEachDocumentTheta(docNo:Int,doc:Array[Int],nodeArray: Array[TreeNode]){
		this.testThetaDocument(docNo)=new Array[Double](nodeArray.size)
		var nd=0;
		var nodeCount=0
		nodeArray.foreach(node=>{
			nd+=node.testDocumentCount(docNo)
			
			nodeCount+=1
		})
		var nodeCount2=0
		nodeArray.foreach(node=>{
			this.testThetaDocument(docNo)(nodeCount2)=(node.testDocumentCount(docNo)+this.alpha)/(nd+nodeArray.size*this.alpha)
//			println("theta-------"+this.testThetaDocument(docNo)(nodeCount2));
//			if(this.testThetaDocument(docNo)(nodeCount2)==0)
//			{
//				println("------------------------------")
//				println("nd"+nd)
//			}
			nodeCount2+=1
		})
	}
	
	def computePerplexity()={
		var sumLogP=0.0
		var sumAllTerm=0
		var docCount=0
		this.testDocuments.foreach(doc=>{
			//println(docCount)
			sumAllTerm+=doc.size;
			sumLogP+=this.computeLogPW(docCount)
			docCount+=1;
		})
		Math.exp(-1*(sumLogP)/sumAllTerm)
	}
	def computeLogPW(docNo:Int)={
		var logPW=0.0;
		var termCount=0;
		this.testDocuments(docNo).foreach(term=>{
			var labelCount=0
			var sumIn=0.0
			this.testLabelsDocument(docNo).foreach(label=>{
				//if(labelCount<)
				var in=this.phiValueMap.get(label)(term)*this.testThetaDocument(docNo)(this.testLabelsDocument(docNo).size-labelCount-1)
				sumIn+=in
				labelCount+=1;
			})
			logPW+=Math.log(sumIn)
			termCount+=1;
		})
		
		logPW;
	}

	//////////////////////////////Sample test set code!!!!!!!!////////////////////////

	def initTreePathArray(pathNode: TreeNode) = {
		var pathList = new ArrayList[TreeNode]()
		this.scanTreeForInitTreePathArray(pathList, pathNode)

		var arr = new Array[TreeNode](pathList.size())
		var i = pathList.iterator()
		var k = 0
		while (i.hasNext()) {
			arr(k) = i.next()
			k += 1
		}
		arr
	}

	def scanTreeForInitTreePathArray(list: ArrayList[TreeNode], node: TreeNode) {
		list.add(node)
		if (node.parent != null) {
			this.scanTreeForInitTreePathArray(list, node.parent)
		}

	}

	def initParameters() {
		println("init the parameters............");
		println("sum of document:" + this.docNumberCount)
		//this.treePath.foreach(node)
		var docCount = 0
		this.storeEachDocument.foreach(document => {
			var pathIndex = -1
			var countArray = 0
			var labels = this.storeLabelsDocument(docCount)
			var lastLabel = labels.size - 1
			treePathArray.foreach(path => {
				if (labels(lastLabel) == path(0).labelIndex) {
					pathIndex = countArray
				}
				countArray += 1
			})
			var termCount = 0
			document.foreach(term => {
				this.treePathArray(pathIndex).foreach(node => {
					node.documentCount(docCount) += 1
					node.termCount(term) += 1
					//node.documentCount(docCount) += 1
					//node.count += 1
				})
				termCount += 1
			})
			docCount += 1
		})
	}

	def initTestSetParameters() {
		println("init the test set parameters............");
		///println("sum of document:" + this.docNumberCount)
		//this.treePath.foreach(node)
		var docCount = 0
		this.testDocuments.foreach(document => {
			var pathIndex = -1
			var countArray = 0
			var labels = this.testLabelsDocument(docCount)
			var lastLabel = labels.size - 1
			treePathArray.foreach(path => {
				if (labels(lastLabel) == path(0).labelIndex) {
					pathIndex = countArray
				}
				countArray += 1
			})
			var termCount = 0
			document.foreach(term => {
				this.treePathArray(pathIndex).foreach(node => {
					node.testDocumentCount(docCount) += 1
					//node.termCount(term) += 1
					//node.documentCount(docCount) += 1
					//node.count += 1
				})
				termCount += 1
			})
			docCount += 1
		})
	}
	/**
	 * 根据label标签组找到相应的路径
	 * *
	 */
	def findTreePathByLabel(labels: Array[String]) = {
		var label = labels(labels.size - 1)
		this.treePath.foreach(node => {
			if (node.nodeName == label)
				node

		})
	}
	def scanFolder(pNode: TreeNode, folder: File): Int = {
		pNode.nodeName = folder.getName();
		pNode.folderPath = folder.toString()
		pNode.termCount = new Array[Int](this.modelParams.numTerms)

		pNode.documentCount = new Array[Int](this.docNumberCount)
		pNode.testDocumentCount = new Array[Int](this.testDocNumberCount)
		//pNode.documentCount = new Array[Int](this.docNumberCount)
		pNode.labelIndex = this.findLabelIndex(folder.getName())
		//		var docCount = 0
		//		this.storeEachDocument.foreach(doc => {
		//
		//			pNode.documentTermCount(docCount) = new Array[Int](doc.length)
		//			docCount += 1
		//		})

		var fileList = folder.listFiles();
		fileList.foreach(f => {
			if (f.isFile()) {
				treePath += pNode
				return 0
			}
			else {
				var node = new TreeNode();
				node.nodeName = f.getName()
				node.folderPath = f.toString()
				node.termCount = new Array[Int](this.modelParams.numTerms)
				///node.documentCount = new Array[Int](this.docNumberCount)
				node.labelIndex = this.findLabelIndex(f.getName())
				node.documentCount = new Array[Int](this.docNumberCount)
				node.testDocumentCount = new Array[Int](this.testDocNumberCount)
				//				var count = 0
				//				this.storeEachDocument.foreach(doc => {
				//					//println(count)
				//					node.documentTermCount(count) = new Array[Int](doc.length)
				//					count += 1
				//				})
				pNode.sons += node
				node.parent = pNode
				this.scanFolder(node, f)
			}
		})

		return 0;
	}

	def printTreePath(node: TreeNode) {
		print(node.nodeName + "<-")
		if (node.parent != null) {
			printTreePath(node.parent)
		}
	}

	def printFolderPath(node: TreeNode) {
		print(node.folderPath + "<--")
		if (node.parent != null) {
			printFolderPath(node.parent)
		}
	}

	def findLabelIndex(labelString: String) = {
		var labelIndex: Int = -1
		var count = 0
		var index = modelParams.labelIndex.get
		index.foreach(i => {

			if (labelString == i) { labelIndex = count }

			count += 1
		})
		labelIndex
	}

}

object Hllda1 {
	def main(args: Array[String]) {
		var lda = new Hllda("E:\\试验数据\\20_newsgroups\\20_newsgroups", 1, 0.1, 0.3);

		//		lda.treePath.foreach(p => {
		//			println()
		//			lda.printFolderPath(p)    
		//		})

		//		lda.treePathArray.foreach(arr =>
		//			{
		//				arr.foreach(node => {
		//					print(node.nodeName + "<=")
		//				})
		//				println()
		//			})
		println("**************************初始化参数********************************")
		lda.initParameters();
		lda.initTestSetParameters();
		println("**************************输出所有路径********************************")
		lda.treePath.foreach(p => {
			println()
			lda.printTreePath(p)
		})
		//		lda.treePathArray.foreach(arr =>
		//			{
		//				arr.foreach(node => {
		//					println(node.nodeName + ":" + node.termCount(1))
		//				})
		//				println()
		//			})

		println()
		println()

		println("**************************开始采样!**************************")
		(0 to 100).foreach(i => {
			if (i % 10 == 0) {
				println("[Train]    iterator:" + i)
			}
			lda.sample();
		})
		lda.treePathArray.foreach(arr =>
			{
				arr.foreach(node => {
					println(node.nodeName + ":" + node.termCount(100))
				})
				println()
			})
		lda.computePhiValue(); 
		var sort=new ArraySort();
		var arr=sort.DoubleSort(lda.phiValueMap.get(2))
		println(Arrays.toString(arr))
		println("**************************开始测试!**************************")
		(0 to 100).foreach(i => {
			if (i % 10 == 0) {
				println("[Test]     iterator:" + i)
			}
			lda.sampleTestSet();
		})
		lda.computeTheta();
		println(Arrays.toString(lda.testThetaDocument(0)));
		println(Arrays.toString(lda.testThetaDocument(2)));
		println("perplexity:"+lda.computePerplexity())
		//		lda.phiValueMap.get(2).foreach(value=>{
		//			println(value)
		//		})
	}
}