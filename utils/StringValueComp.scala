package utils

class StringValueComp() extends Comparable[StringValueComp] {
	var s = ""
	var d = 0.0
	
	override def  compareTo(v2: StringValueComp) = {
		if (this.d < v2.d) {
			1
		}
		else {   
			if (this.d == v2.d) {
				0
			}
			else {
				-1
			}
		}
	}
}