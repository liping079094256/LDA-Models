package newpllda

class TermAssignment   
{
   var isLocalTopic:Boolean=false
   var topicId:Int= -1
   var labelId:Int= -1
   
   def TermAssignment(a:Boolean,b:Int,c:Int)
   {
  	 TermAssignment();
  	 this.isLocalTopic=a;
  	 this.topicId=b;
  	 this.labelId=c;
  	 
   }
   def TermAssignment()
   {
  	 
   }
}