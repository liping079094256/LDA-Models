package hllda
import scala.util.Random

object RandomTest {
    def main(args:Array[String]){
    	(0 to 3).foreach(i=>{
    		println(Random.nextDouble())
    	})
    }
}