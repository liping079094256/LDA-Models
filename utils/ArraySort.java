package utils;

import java.util.Arrays;

public class ArraySort {
	public  StringValueComp[] sortArray(String[] stringArray,
			double[] doubleArray) {

		StringValueComp[] array = new StringValueComp[doubleArray.length];

		for (int i = 0; i < doubleArray.length; i++) {
			array[i] = new StringValueComp();
			array[i].d_$eq(doubleArray[i]);
			array[i].s_$eq(stringArray[i]);

		}

		Arrays.sort(array);
		return array;
	}
	
	public  double[] DoubleSort(double []d)
	{
		double[] dd=new double[d.length];
		for(int i=0;i<d.length;i++)
		{
			dd[i]=d[i];
		}
		Arrays.sort(dd);
		
		return dd;
	}
	
	
}