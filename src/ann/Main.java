package ann;

import java.util.ArrayList;

/**
 * Main class. The executor.
 * @author Daniel Casta�o Estrella
 *
 */
public class Main
{
	public static void main (String args[])
	{
		ArrayList<Byte> genotype_xor = SetGenotype();	
		Ann my_ann = new Ann(genotype_xor, 3, 2);	//genotype, inputs and ouputs
		
		//System.out.println(sigmoid(0.80133));
		my_ann.PrintWeightMapping();
	}
	
	//from 0 to 1 ----- from -1 to 1 do hyperbolic tan
	public static double sigmoid(double x) {
	    return (1/( 1 + Math.pow(Math.E,(-1*x))));
	}
	
	private static ArrayList<Byte> SetGenotype()
	{
		ArrayList<Byte> genotype = new ArrayList<Byte>();
		//for (int i = 0; i < 4; i++) {
		//	genotype.add((byte) 1);
		//}
		
		/*
		 * i3 o2
		 */
		genotype.add((byte) 1);
		genotype.add((byte) 0);
		genotype.add((byte) 1);
		genotype.add((byte) 0);
		genotype.add((byte) 0);
		genotype.add((byte) 1);
		
		genotype.add((byte) 1);
		genotype.add((byte) 0);
		genotype.add((byte) 0);
		genotype.add((byte) 1);
		genotype.add((byte) 0);
		genotype.add((byte) 0);
		
		genotype.add((byte) 0);
		genotype.add((byte) 0);
		genotype.add((byte) 1);
		genotype.add((byte) 0);
		genotype.add((byte) 0);
		genotype.add((byte) 1);
		
		/*
		genotype.add((byte) 1);
		genotype.add((byte) 0);
		genotype.add((byte) 1);
		
		genotype.add((byte) 1);
		genotype.add((byte) 1);
		genotype.add((byte) 0);
		
		genotype.add((byte) 0);
		genotype.add((byte) 1);
		genotype.add((byte) 1);
		*/
		
		return genotype;
	}
}