package test
import scalanlp.io.CSVFile
import java.io.File
import scala.actors.threadpool.Arrays

object CSVFileTest {
	def main(args:Array[String]){
		var f=CSVFile(new File("C:\\Downloads\\新建文件夹\\2.csv"));
		var a=Array("3434","343434");
		f.write(a);
		var b=f.read[Array[Array[String]]];
		b.foreach(i=> i.foreach(k=>println(k)))
		
	}

}