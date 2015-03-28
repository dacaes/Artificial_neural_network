package executor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import ann.Ann;
import ann.Trainer;
import ann.Const;

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
		Ann ann = new Ann(genotype_xor, Const.INPUTS, Const.OUPUTS);	//genotype, inputs and ouputs
		Trainer trainer = new Trainer(ann, Const.LEARN_FACTOR);	//genotype, inputs and ouputs
		trainer.TrainingOffline(Const.TRAININGS);
		/*
		double htan = my_ann.HyperbolicTan(0.36);
		double value = my_ann.ArcHyperbolicTan(htan);
		System.out.println(htan);
		System.out.println(value);
		*/
		/*
		//System.out.println(sigmoid(0.80133));
		
		
		Random rand = new Random();
		int r_max = 1;
		int r_min = -1;

		//System.out.println(rand.nextDouble() * (r_max - r_min) + r_min);
		 */
	}
	
	
	
	private static ArrayList<Byte> SetGenotype()
	{
		ArrayList<Byte> genotype = new ArrayList<Byte>();
		for (int i = 0; i < 4; i++) {
			genotype.add((byte) 1);
		}
		
		//i3 o2
		/*
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
		*/
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
