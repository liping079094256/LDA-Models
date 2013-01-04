package utils
import java.util.Random
import java.util.Arrays


object Test {

	def main(args: Array[String]): Unit = {
		
		var arr= Array[String](1.toString(),2.toString(),3.toString())
		var arr1= Array[Double](1,2,3)
		var s=new ArraySort();
		
		var a=s.sortArray(arr,arr1)
		a.foreach(k=>{
			println(k.s+":"+k.d)
		})
	}

}