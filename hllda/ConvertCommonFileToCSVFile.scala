package hllda
import java.io.File
import scalanlp.io.CSVFile
import scala.io.Source
import java.io.BufferedReader
import java.io.Reader
import java.io.FileReader

object ConvertCommonFileToCSVFile {

	var pathString = " "
	var set = Set[Array[String]]()
	var i = 1
	def loopOnePath(node: TreeNode) {
		pathString = node.nodeName + " " + pathString
		if (node.parent != null) {
			this.loopOnePath(node.parent)
		}
	}

	def addOneFileToSet(f: File) {
		var fileString = ""
		var s = ""
		var reader = new BufferedReader(new FileReader(f))
		s = reader.readLine()
		while (s != null) {
			fileString += s
			s = reader.readLine()
		}
		//fileString=new String(fileString.getBytes("iso-8859-1"),"utf-8")
		set += Array(i.toString(), this.pathString, fileString)
		i += 1
	}

	def writeCSVFile() {
		var arr = new Array[Array[String]](set.size)
		var k = 0
		set.iterator.foreach(s => {
			arr(k) = s
			k += 1
		})
		var csvFile = CSVFile(new File("E:\\试验数据\\20_newsgroups\\2.csv"))
		csvFile.write(arr)

	}

	def main(args: Array[String]): Unit = {
		var lda = new Hllda("E:\\试验数据\\20_newsgroups\\20_newsgroups",0.1,0.9,0.9);
		lda.treePath.foreach(p => {
			ConvertCommonFileToCSVFile.loopOnePath(p)
			println(ConvertCommonFileToCSVFile.pathString);

			var file = new File(p.folderPath)
			file.listFiles().foreach(f => {
				ConvertCommonFileToCSVFile.addOneFileToSet(f)
			})

			ConvertCommonFileToCSVFile.pathString = "";
		})
		ConvertCommonFileToCSVFile.writeCSVFile();
	}

}