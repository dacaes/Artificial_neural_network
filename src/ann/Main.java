package ann;

import java.util.ArrayList;

/**
 * Main class. The executor.
 * @author Daniel Castaño Estrella
 *
 */
public class Main
{
	public static void main (String args[])
	{
		ArrayList<Byte> genotype_xor = SetGenotype();	
		Ann my_ann = new Ann(genotype_xor, 2, 1);
		
		//System.out.println(sigmoid(0.80133));
	}
	
	public static double sigmoid(double x) {
	    return (1/( 1 + Math.pow(Math.E,(-1*x))));
	}
	
	private static ArrayList<Byte> SetGenotype()
	{
		ArrayList<Byte> genotype = new ArrayList<Byte>();
		for (int i = 0; i < 4; i++) {
			genotype.add((byte) 1);
		}
		
		return genotype;
	}
}
