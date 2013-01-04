package newpllda
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
import java.util.Date
class NewPLLDA(alpha: Double, alphaL: Double,
	eta: Double, topicNumOfEachLabel: Int, globalTopicNum: Int,
	filePath: String, percentForTest: Double, percentOfGlobalTopic: Double,
	trainIter: Int, testIter: Int) {
	var totalNumOfLocalTopic = 0;
	var totalDocNum = 0;
	var numOfTrainDoc = 0;
	var numOfTestDoc = 0;
	var numOfLabels = 0;
	var numOfTerms = 0;
	var docsTrain = Array[Array[Int]]()
	var docsTest = Array[Array[Int]]()
	var labelsTrain = Array[Array[Int]]();
	var labelsTest = Array[Array[Int]]();
	//////local topic count arrays
	var labelToLocalTopic = Array[Array[Int]]();
	var localTopicTerm = Array[Array[Int]]();
	var localTopicCount = Array[Int]();
	var docTopic = Array[Array[Int]]();

	var phiLocalTopicTerm = Array[Array[Double]]();
	//var docLable = Array[Array[Int]]();

	//////global topic count arrays
	var gLabelTopicTerm = Array[Array[Array[Int]]]();
	var gLabelTopic = Array[Array[Int]]();
	var gDocLabelTopic = Array[Array[Array[Int]]]();
	var gDocLabel = Array[Array[Int]]();
	var phiGlobalLabelTopicTerm = Array[Array[Array[Double]]]();
	var phiGlobalTopicTerm = Array[Array[Double]]();
	/*** phiLabelTerm  **/
	var phiLabelTerm = Array[Array[Double]]();

	/**
	 * topic Assignment for each term of each Document!
	 */
	var docTrainTopicAssignment = Array[Array[TermAssignment]]()

	/**
	 * test document counts!!!!!!!!!!!!
	 * *
	 */
	var docTopicTest = Array[Array[Int]]();
	var gDocLabelTopicTest = Array[Array[Array[Int]]]();
	var gDocLabelTest = Array[Array[Int]]();
	var testDocTrainTopicAssignment = Array[Array[TermAssignment]]()

	/**
	 * theta of each test document
	 * **
	 */
	var thetaOfTestDocs = Array[Array[Double]]();
	var thetaTestDocTopic = Array[Array[Double]]();
	var thetaTestDocGlobalTopic = Array[Array[Double]]();
	/***   perplexity   **/
	var perplexity = 0.0;
	var perplexityOfTopics = 0.0;
	/***number of  correct classify document's labels**/
	var numOfCorrect = 0.0;
	var oneError=0.0;
	var converage=0.0;

	def initCountParametersForTestDocs() {
		thetaOfTestDocs = new Array[Array[Double]](this.numOfTestDoc);
		this.thetaTestDocTopic = new Array[Array[Double]](this.numOfTestDoc);
		this.thetaTestDocGlobalTopic = new Array[Array[Double]](this.numOfTestDoc);
		(0 until this.numOfTestDoc).foreach(c => {
			thetaOfTestDocs(c) = new Array[Double](this.numOfLabels);
			thetaTestDocTopic(c) = new Array[Double](this.totalNumOfLocalTopic);
			thetaTestDocGlobalTopic(c) = new Array[Double](this.globalTopicNum);
		})

		docTopicTest = new Array[Array[Int]](this.numOfTestDoc);
		(0 until this.numOfTestDoc).foreach(c => {
			docTopicTest(c) = new Array[Int](this.totalNumOfLocalTopic)
		})

		gDocLabelTopicTest = new Array[Array[Array[Int]]](this.numOfTestDoc);

		(0 until this.numOfTestDoc).foreach(c => {
			gDocLabelTopicTest(c) = new Array[Array[Int]](this.numOfLabels);
			(0 until this.numOfLabels).foreach(e => {
				gDocLabelTopicTest(c)(e) = new Array[Int](this.globalTopicNum);
			});
		})
		gDocLabelTest = new Array[Array[Int]](this.numOfTestDoc);
		(0 until this.numOfTestDoc).foreach(c => {
			gDocLabelTest(c) = new Array[Int](this.numOfLabels);
		});

		testDocTrainTopicAssignment = new Array[Array[TermAssignment]](this.numOfTestDoc);
		var docCount = 0;
		this.docsTest.foreach(doc => {
			testDocTrainTopicAssignment(docCount) = new Array[TermAssignment](doc.size);
			var termCount = 0
			doc.foreach(term => {
				var d = Random.nextDouble();
				if (d < this.percentOfGlobalTopic) {
					var topic = Random.nextInt(this.globalTopicNum);
					var label = Random.nextInt(this.numOfLabels);
					this.testDocTrainTopicAssignment(docCount)(termCount) = new TermAssignment()
					this.testDocTrainTopicAssignment(docCount)(termCount).isLocalTopic = false;
					this.testDocTrainTopicAssignment(docCount)(termCount).labelId = label;
					this.testDocTrainTopicAssignment(docCount)(termCount).topicId = topic;

					//////global topic count arrays
					//gLabelTopicTerm(label)(topic)(term) += 1;
					//gLabelTopic(label)(topic) += 1;
					gDocLabelTopicTest(docCount)(label)(topic) += 1;
					gDocLabelTest(docCount)(label) += 1;
				}
				else {
					var topicId = Random.nextInt(this.totalNumOfLocalTopic)
					this.testDocTrainTopicAssignment(docCount)(termCount) = new TermAssignment();
					this.testDocTrainTopicAssignment(docCount)(termCount).isLocalTopic = true;
					this.testDocTrainTopicAssignment(docCount)(termCount).topicId = topicId;

					//////local topic count arrays
					var label = topicId / this.topicNumOfEachLabel
					var topic = topicId % this.topicNumOfEachLabel
					this.testDocTrainTopicAssignment(docCount)(termCount).labelId = label;
					//labelToLocalTopic(label)(topic) += 1;
					//localTopicTerm(topicId)(term) += 1;
					//localTopicCount(topicId) += 1;
					docTopicTest(docCount)(topicId) += 1;
					//docLable(docCount)(label) += 1;
					gDocLabelTest(docCount)(label) += 1;
				}
				termCount += 1;
			})
			docCount += 1;
		});
	}

	def initCountParameters() {
		phiLabelTerm = new Array[Array[Double]](this.numOfLabels)
		(0 until this.numOfLabels).foreach(c => {
			phiLabelTerm(c) = new Array[Double](this.numOfTerms)
		})

		//////local topic count arrays
		labelToLocalTopic = new Array[Array[Int]](this.numOfLabels);
		var topicCount = 0;
		(0 until this.numOfLabels).foreach(count => {
			this.labelToLocalTopic(count) = new Array[Int](this.topicNumOfEachLabel);
			(0 until this.topicNumOfEachLabel).foreach(c => {
				this.labelToLocalTopic(count)(c) = topicCount;
				topicCount += 1;
			})
		})

		localTopicTerm = new Array[Array[Int]](this.totalNumOfLocalTopic);
		phiLocalTopicTerm = new Array[Array[Double]](this.totalNumOfLocalTopic);
		(0 until this.totalNumOfLocalTopic).foreach(topic => {
			localTopicTerm(topic) = new Array[Int](this.numOfTerms);
			phiLocalTopicTerm(topic) = new Array[Double](this.numOfTerms);
		})
		localTopicCount = new Array[Int](this.totalNumOfLocalTopic);
		docTopic = new Array[Array[Int]](this.numOfTrainDoc);
		(0 until this.numOfTrainDoc).foreach(doc => {
			docTopic(doc) = new Array[Int](this.totalNumOfLocalTopic);
		})
		//		docLable = new Array[Array[Int]](this.numOfTrainDoc);
		//		(0 until this.numOfTrainDoc).foreach(c => {
		//			docLable(c) = new Array[Int](this.numOfLabels);
		//		})

		//////global topic count arrays
		gLabelTopic = new Array[Array[Int]](this.numOfLabels);
		gLabelTopicTerm = new Array[Array[Array[Int]]](this.numOfLabels);
		phiGlobalLabelTopicTerm = new Array[Array[Array[Double]]](this.numOfLabels);
		(0 until this.numOfLabels).foreach(c => {
			gLabelTopicTerm(c) = new Array[Array[Int]](this.globalTopicNum);
			phiGlobalLabelTopicTerm(c) = new Array[Array[Double]](this.globalTopicNum);
			gLabelTopic(c) = new Array[Int](this.globalTopicNum);
			(0 until this.globalTopicNum).foreach(e => {
				gLabelTopicTerm(c)(e) = new Array[Int](this.numOfTerms);
				phiGlobalLabelTopicTerm(c)(e) = new Array[Double](this.numOfTerms);
			})
		});

		gDocLabelTopic = new Array[Array[Array[Int]]](this.numOfTrainDoc);
		(0 until this.numOfTrainDoc).foreach(c => {
			gDocLabelTopic(c) = new Array[Array[Int]](this.numOfLabels);
			(0 until this.numOfLabels).foreach(e => {
				gDocLabelTopic(c)(e) = new Array[Int](this.globalTopicNum)
			})
		})
		this.gDocLabel = new Array[Array[Int]](this.numOfTrainDoc);
		(0 until this.numOfTrainDoc).foreach(c => {
			gDocLabel(c) = new Array[Int](this.numOfLabels);
		})

		//init topic Assignment for each term !
		this.docTrainTopicAssignment = new Array[Array[TermAssignment]](this.numOfTrainDoc)
		var docCount = 0;
		this.docsTrain.foreach(doc => {
			this.docTrainTopicAssignment(docCount) = new Array[TermAssignment](doc.size)

			var termCount = 0
			doc.foreach(term => {
				var d = Random.nextDouble();
				if (d < this.percentOfGlobalTopic) {
					var topic = Random.nextInt(this.globalTopicNum);
					var label = Random.nextInt(this.numOfLabels);
					this.docTrainTopicAssignment(docCount)(termCount) = new TermAssignment()
					this.docTrainTopicAssignment(docCount)(termCount).isLocalTopic = false;
					this.docTrainTopicAssignment(docCount)(termCount).labelId = label;
					this.docTrainTopicAssignment(docCount)(termCount).topicId = topic;

					//////global topic count arrays
					gLabelTopicTerm(label)(topic)(term) += 1;
					gLabelTopic(label)(topic) += 1;
					gDocLabelTopic(docCount)(label)(topic) += 1;
					gDocLabel(docCount)(label) += 1;
				}
				else {
					var topicId = Random.nextInt(this.totalNumOfLocalTopic)
					this.docTrainTopicAssignment(docCount)(termCount) = new TermAssignment()
					this.docTrainTopicAssignment(docCount)(termCount).isLocalTopic = true;
					this.docTrainTopicAssignment(docCount)(termCount).topicId = topicId;

					//////local topic count arrays
					var label = topicId / this.topicNumOfEachLabel
					var topic = topicId % this.topicNumOfEachLabel
					this.docTrainTopicAssignment(docCount)(termCount).labelId = label;
					//labelToLocalTopic(label)(topic) += 1;
					localTopicTerm(topicId)(term) += 1;
					localTopicCount(topicId) += 1;
					docTopic(docCount)(topicId) += 1;
					//docLable(docCount)(label) += 1;
					gDocLabel(docCount)(label) += 1;
				}
				termCount += 1;
			})

			docCount += 1;
		})

	}

	def initParameters() {

		val source = CSVFile(this.filePath) ~> IDColumn(1);

		val tokenizer = {
			SimpleEnglishTokenizer() ~>
				CaseFolder() ~>
				WordsAndNumbersOnlyFilter() ~> // ignore non-words and non-numbers
				MinimumLengthFilter(1) // take terms with >=3 characters
		}

		val text = {
			source ~>
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

		val dataset = LabeledLDADataset(text, labels);
		var phiValueMap = new HashMap[Int, Array[Double]]();
		// define the model parameters
		val modelParams = LabeledLDAModelParams(dataset);
		this.totalDocNum = dataset.iterator.size
		this.numOfLabels = modelParams.numLabels
		println("labels:");
		modelParams.labelIndex.get.foreach(label => {
			println(label);
		})
		///println(Arrays.toString())

		this.numOfTerms = modelParams.numTerms
		this.docsTrain = new Array[Array[Int]]((this.totalDocNum * (1 - this.percentForTest)).toInt)
		this.labelsTrain = new Array[Array[Int]]((this.totalDocNum * (1 - this.percentForTest)).toInt)
		this.numOfTrainDoc = this.docsTrain.size;
		this.numOfTestDoc = this.totalDocNum - this.numOfTrainDoc;
		this.docsTest = new Array[Array[Int]](this.numOfTestDoc);
		this.labelsTest = new Array[Array[Int]](this.numOfTestDoc);
		this.totalNumOfLocalTopic = this.numOfLabels * this.topicNumOfEachLabel;

		var docCount = 0;
		dataset.iterator.foreach(doc => {
			if (docCount < this.numOfTrainDoc) {
				this.docsTrain(docCount) = doc.terms;
				this.labelsTrain(docCount) = doc.labels;
			}
			else {
				this.docsTest(docCount - this.numOfTrainDoc) = doc.terms;
				this.labelsTest(docCount - this.numOfTrainDoc) = doc.labels;
			}
			docCount += 1;
		})
	}

	/**
	 * sampling Train docs
	 */
	def sampingTrainDocs() {
		(1 to this.trainIter).foreach(iter => {
			println("[Train]    iterator:" + iter + "th")
			var docCount = 0
			this.docsTrain.foreach(doc => {
				var termCount = 0;
				doc.foreach(term => {
					samplingTrainDocTerm(docCount, termCount);

					termCount += 1;
				})

				docCount += 1;
			})
		})

	}
	/**
	 * sampling one term of the training set doc
	 * **
	 */
	def samplingTrainDocTerm(doc: Int, term: Int) {
		/**
		 * *
		 * --the topic assignment arrays
		 * **
		 */
		var assignment = this.docTrainTopicAssignment(doc)(term);
		if (assignment.isLocalTopic) {
			var label = assignment.labelId;
			var topicId = assignment.topicId;

			var topic = topicId % this.topicNumOfEachLabel
			//labelToLocalTopic(label)(topic) -= 1;
			localTopicTerm(topicId)(this.docsTrain(doc)(term)) -= 1;
			localTopicCount(topicId) -= 1;
			docTopic(doc)(topicId) -= 1;
			//docLable(doc)(label) += 1;
			gDocLabel(doc)(label) -= 1;
		}
		else {
			var label = assignment.labelId;
			var topic = assignment.topicId;

			gLabelTopicTerm(label)(topic)(this.docsTrain(doc)(term)) -= 1;
			gLabelTopic(label)(topic) -= 1;
			gDocLabelTopic(doc)(label)(topic) -= 1;
			gDocLabel(doc)(label) -= 1;

		}

		//sampling code
		var labels = this.labelsTrain(doc)
		var d = Random.nextDouble();
		if (d < this.percentOfGlobalTopic) {
			var p = new Array[Array[Double]](labels.size);
			(0 until p.size).foreach(c => {
				p(c) = new Array[Double](this.globalTopicNum);
			})

			var sumP = 0.0;
			(0 until p.size).foreach(c => {
				(0 until p(c).size).foreach(e => {
					var curLabel = labels(c);
					var curTopic = e;
					var curTerm = this.docsTrain(doc)(term);
					p(c)(e) = (gLabelTopicTerm(curLabel)(curTopic)(curTerm) + eta) /
						(gLabelTopic(curLabel)(curTopic) + numOfTerms * eta) *
						(gDocLabel(doc)(curLabel) + alphaL) /
						(docsTrain(doc).size - 1 + labels.size * alphaL) *
						(gDocLabelTopic(doc)(curLabel)(curTopic) + alpha) /
						(gDocLabel(doc)(curLabel) + globalTopicNum * alpha);
					sumP += p(c)(e)
				});
			})
			//normalize the distributions
			(0 until p.size).foreach(c => {
				(0 until p(c).size).foreach(e => {
					p(c)(e) /= sumP;
				});
			})
			var sumDis = 0.0;
			var distribution = new Array[Double](this.globalTopicNum * this.numOfLabels)
			var count = 0;
			(0 until p.size).foreach(c => {
				(0 until p(c).size).foreach(e => {
					sumDis += p(c)(e)
					distribution(count) += sumDis;
					count += 1;
				})
			})
			var r = Random.nextDouble();
			var selectId = -1;
			(0 until distribution.size).foreach(c => {
				if (r < distribution(c) && selectId == -1) {
					selectId = c;
				}
			})
			var selLabel = labels(selectId / this.globalTopicNum);
			var selTopic = selectId % this.globalTopicNum

			this.docTrainTopicAssignment(doc)(term).isLocalTopic = false;
			this.docTrainTopicAssignment(doc)(term).labelId = selLabel;
			this.docTrainTopicAssignment(doc)(term).topicId = selTopic;
			gLabelTopicTerm(selLabel)(selTopic)(this.docsTrain(doc)(term)) += 1;
			gLabelTopic(selLabel)(selTopic) += 1;
			gDocLabelTopic(doc)(selLabel)(selTopic) += 1;
			gDocLabel(doc)(selLabel) += 1;

		}
		else {
			var p = new Array[Double](labels.size * this.topicNumOfEachLabel);
			var countP: Double = 0;
			(0 until p.size).foreach(c => {
				var curTopic = 0;
				var curTerm = 0;

				curTopic = this.labelToLocalTopic(c / this.topicNumOfEachLabel)(c % this.topicNumOfEachLabel)
				curTerm = this.docsTrain(doc)(term);

				p(c) = (this.localTopicTerm(curTopic)(curTerm) + this.eta) / (this.localTopicCount(curTopic) + this.numOfTerms * this.eta) * (this.docTopic(doc)(curTopic) + this.alpha);
				countP += p(c);
			})

			//normalize  p
			(0 until p.size).foreach(c => {
				p(c) = p(c) / countP
			})

			var pp = new Array[Double](p.size);
			var ppCount: Double = 0;
			var iter = 0;
			p.foreach(pValue => {
				ppCount += pValue;
				pp(iter) = ppCount;
				iter += 1;
			})

			var r = Random.nextDouble();
			iter = 0;
			var selectTopic = -1;
			pp.foreach(ppValue => {
				if (r < ppValue && selectTopic == -1) {
					selectTopic = iter;
				}
				iter += 1;
			})
			var selectTopicId = -1;
			var selectLabelId = -1;
			selectLabelId = labels(selectTopic / this.topicNumOfEachLabel)
			selectTopicId = this.labelToLocalTopic(selectLabelId)(selectTopic % this.topicNumOfEachLabel)
			this.docTrainTopicAssignment(doc)(term).isLocalTopic = true;
			this.docTrainTopicAssignment(doc)(term).topicId = selectTopicId;
			this.docTrainTopicAssignment(doc)(term).labelId = selectLabelId;
			////labelToLocalTopic(selectLabelId)(topic) -= 1;
			localTopicTerm(selectTopicId)(this.docsTrain(doc)(term)) += 1;
			localTopicCount(selectTopicId) += 1;
			docTopic(doc)(selectTopicId) += 1;
			//docLable(doc)(label) += 1;
			gDocLabel(doc)(selectLabelId) += 1;
		}
	}

	/**
	 * *
	 * sampling test doc
	 * **
	 */
	def sampingTestDocs() {
		(1 to this.testIter).foreach(iter => {
			println("[Test]    iterator:" + iter + "th")
			var docCount = 0
			this.docsTest.foreach(doc => {
				var termCount = 0;
				doc.foreach(term => {
					samplingTestDocTerm(docCount, termCount);
					termCount += 1;
				})
				docCount += 1;
			})
		})
	}
	def samplingTestDocTerm(doc: Int, term: Int) {
		/**--the topic assignment arrays*/
		var assignment = this.testDocTrainTopicAssignment(doc)(term);
		if (assignment.isLocalTopic) {
			var label = assignment.labelId;
			var topicId = assignment.topicId;

			var topic = topicId % this.topicNumOfEachLabel
			docTopicTest(doc)(topicId) -= 1;
			gDocLabelTest(doc)(label) -= 1;
		}
		else {
			var label = assignment.labelId;
			var topic = assignment.topicId;

			gDocLabelTopic(doc)(label)(topic) -= 1;
			gDocLabel(doc)(label) -= 1;
		}

		//sampling code
		var d = Random.nextDouble();
		if (d < this.percentOfGlobalTopic) {
			var p = new Array[Array[Double]](this.numOfLabels);
			(0 until p.size).foreach(c => {
				p(c) = new Array[Double](this.globalTopicNum);
			})

			var sumP = 0.0;
			(0 until p.size).foreach(c => {
				(0 until p(c).size).foreach(e => {
					p(c)(e) = (this.gDocLabelTest(doc)(c) + this.alphaL) /
						(this.docsTest(doc).size - 1 + this.globalTopicNum * this.alphaL) *
						(this.gDocLabelTopicTest(doc)(c)(e) + this.alpha)
					sumP += p(c)(e)
				});
			})
			//normalize the distributions
			(0 until p.size).foreach(c => {
				(0 until p(c).size).foreach(e => {
					p(c)(e) /= sumP;
				});
			})
			var sumDis = 0.0;
			var distribution = new Array[Double](this.globalTopicNum * this.numOfLabels)
			var count = 0;
			(0 until p.size).foreach(c => {
				(0 until p(c).size).foreach(e => {
					sumDis += p(c)(e)
					distribution(count) += sumDis;
					count += 1;
				})
			})
			var r = Random.nextDouble();
			var selectId = -1;
			(0 until distribution.size).foreach(c => {
				if (r < distribution(c) && selectId == -1) {
					selectId = c;
				}
			})
			var selLabel = selectId / this.globalTopicNum;
			var selTopic = selectId % this.globalTopicNum

			this.testDocTrainTopicAssignment(doc)(term).isLocalTopic = false;
			this.testDocTrainTopicAssignment(doc)(term).labelId = selLabel;
			this.testDocTrainTopicAssignment(doc)(term).topicId = selTopic;

			gDocLabelTopicTest(doc)(selLabel)(selTopic) += 1;
			gDocLabelTest(doc)(selLabel) += 1;
		}
		else {
			var p = new Array[Double](this.totalNumOfLocalTopic);
			var countP: Double = 0;
			(0 until p.size).foreach(c => {

				p(c) = (this.docTopicTest(doc)(c) + this.alpha) *
					this.phiLocalTopicTerm(c)(this.docsTest(doc)(term));
				countP += p(c);
			})

			//normalize  p
			(0 until p.size).foreach(c => {
				p(c) = p(c) / countP
			})

			var pp = new Array[Double](p.size);
			var ppCount: Double = 0;
			var iter = 0;
			p.foreach(pValue => {
				ppCount += pValue;
				pp(iter) = ppCount;
				iter += 1;
			})

			var r = Random.nextDouble();
			iter = 0;
			var selectTopic = -1;
			pp.foreach(ppValue => {
				if (r <= ppValue && selectTopic == -1) {
					selectTopic = iter;
				}
				iter += 1;
			})
			var selectTopicId = -1;
			var selectLabelId = -1;
			selectLabelId = selectTopic / this.topicNumOfEachLabel
			selectTopicId = selectTopic
			this.testDocTrainTopicAssignment(doc)(term).isLocalTopic = true;
			this.testDocTrainTopicAssignment(doc)(term).topicId = selectTopicId;
			this.testDocTrainTopicAssignment(doc)(term).labelId = selectLabelId;
			docTopicTest(doc)(selectTopicId) += 1;
			gDocLabelTest(doc)(selectLabelId) += 1;
		}
	}

	def computePhi() {
		this.computeGlobalPhi();
		this.computeLocalPhi();
	}
	def computeLocalPhi() {
		(0 until this.phiLocalTopicTerm.size).foreach(c => {
			(0 until this.numOfTerms).foreach(e => {
				phiLocalTopicTerm(c)(e) = (this.localTopicTerm(c)(e) + this.eta) /
					(this.localTopicCount(c) + this.numOfTerms * this.eta) //*(1 - this.percentOfGlobalTopic)
			})
		})
	}
	def computeGlobalPhi() {
		(0 until this.numOfLabels).foreach(j => {
			(0 until this.globalTopicNum).foreach(c => {
				(0 until this.numOfTerms).foreach(e => {
					phiGlobalLabelTopicTerm(j)(c)(e) = (this.gLabelTopicTerm(j)(c)(e) + this.eta) /
						(this.gLabelTopic(j)(c) + this.numOfTerms * this.eta);
				})
			})
		})
	}

	/**  compute phi value for labels**/
	def computePhiValueForLabels() {
		(0 until this.numOfLabels).foreach(c => {
			var count = 0.0;
			(0 until this.numOfTerms).foreach(e => {
				(c * this.topicNumOfEachLabel until (c + 1) * topicNumOfEachLabel).foreach(k => {
					this.phiLabelTerm(c)(e) += this.phiLocalTopicTerm(k)(e);
					count += this.phiLocalTopicTerm(k)(e)
				})
				(0 until this.globalTopicNum).foreach(k => {
					this.phiLabelTerm(c)(e) += this.phiGlobalLabelTopicTerm(c)(k)(e)
					count += this.phiGlobalLabelTopicTerm(c)(k)(e)
				})
			})
			(0 until this.numOfTerms).foreach(e => {
				this.phiLabelTerm(c)(e) /= count;
			})

		})
	}
	def getNkt(labelId: Int, termId: Int) = {
		var nkt = 0;
		this.gLabelTopicTerm(labelId).foreach(c => {
			nkt += c(termId)
		})
		(labelId * this.topicNumOfEachLabel until (labelId + 1) * this.topicNumOfEachLabel).foreach(c => {
			nkt += this.localTopicTerm(c)(termId)
		})
		nkt;
	}
	def getNk(labelId: Int) = {
		var nk = 0;
		(0 until this.numOfTerms).foreach(c => {
			nk += getNkt(labelId, c);
		})
		nk;
	}

	/**
	 * *
	 * compute number of correct classify document's labels
	 * **
	 */
	def computeNumOfCorrectClassify() {
		var docLabelCount = 0;
		(0 until this.numOfTestDoc).foreach(docCount => {
			docLabelCount += this.labelsTest(docCount).size;
			var labels = this.labelsTest(docCount)
			//			println(Arrays.toString(labels));
			//			println("num of labels:"+labels.size);
			var labelsOrder = new Array[String](this.numOfLabels);
			(0 until labelsOrder.size).foreach(c => {
				labelsOrder(c) = c.toString();
			});
			var distribution = this.thetaOfTestDocs(docCount)
			var sort = new ArraySort();
			var result = sort.sortArray(labelsOrder, distribution);
			var list = new ArrayList[String]();
			(0 until labels.size).foreach(c => {
				list.add(result(c).s);
			})
			(0 until labels.size).foreach(c => {
				if (list.contains(labels(c).toString())) {
					this.numOfCorrect += 1;
				}
			})
			var labelList=new ArrayList[String]();
			(0 until labels.size).foreach(c=>{
				labelList.add(labels(c).toString());
			})
			if(labelList.contains(result(0).s))
			{
				
			}
			else
			{
				this.oneError+=1;
			}
			//compute converage!!!!!!!!!!!!
			var i=0;
			var ifBreak=false;
			var c=0;
			while(i<result.size&&ifBreak==false)
			{
				if(labelList.contains(result(i).s))
				{
					c+=1;
				}
				if(c==labelList.size())
				{
					this.converage+=i;
					ifBreak=true;
				}
				
				i+=1;
			}
		});
		this.converage=this.converage/docLabelCount-1;
		this.oneError/=this.docsTest.size;
		this.numOfCorrect /= docLabelCount;

	}

	//compute topic theta for all of test Documents

	def computeTopicTheta() {
		(0 until this.numOfTestDoc).foreach(docId => {
			var sum = 0.0;
			var array = new Array[Double](this.globalTopicNum);
			(0 until this.totalNumOfLocalTopic + this.globalTopicNum).foreach(c => {
				if (c < totalNumOfLocalTopic) {
					sum += this.docTopicTest(docId)(c)
				}
				else {

					(0 until this.numOfLabels).foreach(g => {
						sum += this.gDocLabelTopicTest(docId)(g)(c - totalNumOfLocalTopic)
						array(c - totalNumOfLocalTopic) += this.gDocLabelTopicTest(docId)(g)(c - totalNumOfLocalTopic)
					})
				}
			})
			(0 until this.totalNumOfLocalTopic + this.globalTopicNum).foreach(c => {
				if (c < totalNumOfLocalTopic) {
					this.thetaTestDocTopic(docId)(c) = (this.docTopicTest(docId)(c) + this.alphaL) /
						(sum + (totalNumOfLocalTopic + globalTopicNum) * this.alphaL);
				}
				else {
					this.thetaTestDocGlobalTopic(docId)(c - totalNumOfLocalTopic) = (array(c - totalNumOfLocalTopic) + this.alphaL) /
						(sum + (totalNumOfLocalTopic + globalTopicNum) * this.alphaL);
				}
			})
			var counting = 0.0;
			(0 until this.totalNumOfLocalTopic ).foreach(c => {
					counting += this.thetaTestDocTopic(docId)(c)
			})
			(0 until this.totalNumOfLocalTopic).foreach(c => {
					this.thetaTestDocTopic(docId)(c) /= counting
			})
		})
	}
	def computeLabelThetaOfTestDoc() {
		(0 until this.docsTest.size).foreach(doc => {
			var count = 0.0;
			(0 until this.numOfLabels).foreach(label => {
				(label * topicNumOfEachLabel until (label + 1) * topicNumOfEachLabel).foreach(topic => {
					this.thetaOfTestDocs(doc)(label) += this.thetaTestDocTopic(doc)(topic);
					count += this.thetaTestDocTopic(doc)(topic);
				})
			})
			(0 until this.numOfLabels).foreach(label => {
				this.thetaOfTestDocs(doc)(label) /= count;
			})
		})
	}

	//compute topic's perplexity for all of testing documents

	def computeGlobalTopicPhi() {
		this.phiGlobalTopicTerm = new Array[Array[Double]](this.globalTopicNum);
		(0 until this.globalTopicNum).foreach(c => {
			this.phiGlobalTopicTerm(c) = new Array[Double](this.numOfTerms);
		})
		(0 until this.globalTopicNum).foreach(c => {
			var count = 0.0
			(0 until this.numOfTerms).foreach(e => {
				(0 until this.numOfLabels).foreach(k => {
					this.phiGlobalTopicTerm(c)(e) += this.phiGlobalLabelTopicTerm(k)(c)(e)
					count += this.phiGlobalLabelTopicTerm(k)(c)(e)
				})
			})
			(0 until this.numOfTerms).foreach(e => {
				this.phiGlobalTopicTerm(c)(e) /= count;
			})
		})
	}

	def computePerplexityTopic() {
		computeGlobalTopicPhi();
		var sumOfLog = 0.0;
		var sumOfLength = 0.0;
		(0 until this.numOfTestDoc).foreach(c => {
			sumOfLog += computeLogTestDocument(c);
			sumOfLength += this.docsTest(c).size;
		})
		this.perplexityOfTopics = Math.exp(-1 * sumOfLog / sumOfLength);
	}

	def computeLogTestDocument(docId: Int) = {
		var result = 0.0;
		(0 until this.docsTest(docId).size).foreach(c => {
			var r = 0.0;
			(0 until this.totalNumOfLocalTopic + this.globalTopicNum).foreach(e => {
				if (e < this.totalNumOfLocalTopic) {
					r += this.phiLocalTopicTerm(e)(this.docsTest(docId)(c)) *
						this.thetaTestDocTopic(docId)(e)
				}
				else {
					r += this.phiGlobalTopicTerm(e - totalNumOfLocalTopic)(this.docsTest(docId)(c)) *
						this.thetaTestDocGlobalTopic(docId)(e - totalNumOfLocalTopic)
				}
			})
			result += Math.log(r);
		})
		result;
	}

} //class end code!!!!!!!!!!

object testLDA {
	def main(args: Array[String]) {
		var alpha = 0.01
		var alphaL = 0.01;

		var eta = 0.01;
		var topicNumOfEachLabel = 6
		var globalTopicNum = 6;
		var filePath = "E:\\试验数据\\enron\\enron.csv"
		var percentForTest = 0.2
		var percentOfGlobalTopic = 0.5
		var trainIter = 100;
		var testIter = 100;
		var lda = new NewPLLDA(alpha, alphaL, eta, topicNumOfEachLabel,
			globalTopicNum, filePath, percentForTest,
			percentOfGlobalTopic, trainIter, testIter);
		lda.initParameters();
		lda.initCountParameters();
		lda.initCountParametersForTestDocs();
		var date1 = new Date();
		lda.sampingTrainDocs();
		var date2 = new Date();
		lda.computePhi();

		///lda.computePhiValueForLabels();
		lda.sampingTestDocs()
		var aSort = new ArraySort();
		var k = aSort.DoubleSort(lda.phiLocalTopicTerm(0))
		println(Arrays.toString(k));
		//lda.computeTheta();
		(0 until 2).foreach(c => {
			println(Arrays.toString(lda.thetaOfTestDocs(c)));
		})
		//lda.computePerplexity();
		lda.computeTopicTheta();
		lda.computePerplexityTopic();
		//		println("perplexity:"+lda.perplexity)
		lda.computeLabelThetaOfTestDoc();
		lda.computeNumOfCorrectClassify();
		
		//		println("correct:"+lda.numOfCorrect);
		//		
		var trainDocNum = lda.docsTrain.size;
		var testDocNum = lda.docsTest.size;
		var numOfTerms = lda.numOfTerms;
		var trainDocTotalTermNum = 0;
		lda.docsTrain.foreach(c => {
			trainDocTotalTermNum += c.size;
		})
		var testDocTermsNum = 0;
		lda.docsTest.foreach(c => {
			testDocTermsNum += c.size
		});

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
		print("平均每次迭代时间：" + (date2.getTime() - date1.getTime()) * 1.0 / 1000.0 / trainIter + "s\t\t");
		print("迭代次数：" + trainIter + "\t\t");
		println();
		println();

		////////////////////testing document
		print("测试样本总数：" + testDocNum + "\t");
		print("总词数：" + testDocTermsNum + "\t\t");
		print("总标记数:" + "\t\n");
		print("平均文档长度:" + Math.round(testDocTermsNum * 1.0 / testDocNum) + "\t\t");
		///print("正确分类的标签数："+counting+"\t");
		print("正确分类率：" + lda.numOfCorrect);
		print("  perplexity:" + lda.perplexityOfTopics)
		////print("perplexity："+perplexity+"");
		println()
		print("one_error:"+lda.oneError);

		//var k=Math.round(D);
		////println("[perplexity] perplexity at " + numTopics + " topics: " + perplexity);

		println("*****************************************************************");

	}
}