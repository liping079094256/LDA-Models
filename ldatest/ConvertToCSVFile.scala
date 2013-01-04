package ldatest
import scalanlp.io.CSVFile
import java.io.File
import scala.io.Source
import java.util.ArrayList
import scala.collection.immutable.List

object ConvertToCSVFile {
	val CsvFile = CSVFile(new File("C:\\Downloads\\34234.csv"))
	var set=Set[String]()

	def addFileToList(file: File) =
		{
			 
			var s = "";
			var lineList=Source.fromFile(file).getLines().toList
			for(i<-1 to lineList.size)
			{
				s=s+lineList(i-1)
			}
		    set+=s;
		}

	def main(args: Array[String]) {
		var folder = new File("E:\\路透社数据\\20_newsgroups\\新建文件夹");
		var fileList1 = folder.listFiles()
		fileList1.foreach(sonfolder => {
			println(sonfolder);
			var fileList = sonfolder.listFiles()
			fileList.foreach(singleFile => {
				println(singleFile)
				this.addFileToList(singleFile);
			})
		})
		
		println("total number of document:" + set.size);
		var array=new Array[Array[String]](set.size)
		var i=1;
		set.iterator.foreach(s=>{
			var arr=Array[String](i.toString(),s)
			i=i+1;
			array(i-2)=arr
		})
		
		CsvFile.write(array);
	}

}