package ann;

import java.util.ArrayList;
import java.util.Random;

import dataset.DataGen;
import ann.Const.Activation;

/**
 * Artificial neural network
 * @author Daniel Castaño Estrella
 *
 */
public class Ann
{
	//NEURONS
	final int length_I;
	final int length_H;
	final int length_O;
	
	//LEARN FACTOR
	final double learn_factor;
	
	//MAPPING
	Byte[][] 	mapping_I_O;		//input -> output mapping
	Byte[][] 	mapping_H_I;		//input -> length_H mapping
	Byte[][]	mapping_H_O;		//length_H -> output mapping
	
	//WEIGHTS
	double[][] 	weights_I_O;		//input -> output weight
	double[][] 	weights_H_I;		//input -> length_H weight
	double[][] 	weights_H_O;		//length_H -> output weight
	
	double[] 	weights_H_BIAS;		//length_H bias weight
	double[] 	weights_O_BIAS;		//output bias weight
	
	//VALUES
	double[] 	neurons_I;			//input values
	double[] 	neurons_H;			//length_H values
	double[] 	neurons_O;			//output values
	
	//ERRORS
	double[] 	errors_H;			//length_H errors
	double[] 	errors_O;			//output errors
	
	//DELTAS
	double[][] 	deltas_I_O;			//input -> output deltas
	double[][] 	deltas_H_I;			//input -> length_H deltas
	double[][] 	deltas_H_O;			//length_H -> output deltas
	
	double[] 	deltas_H_BIAS;		//length_H bias deltas
	double[] 	deltas_O_BIAS;		//output bias deltas	
	
	public Ann(ArrayList<Byte> genotype, int inputs, int outputs, double learn_factor)
	{
		this.learn_factor = learn_factor;					//store the learn factor
		this.length_I = inputs;
		this.length_O = outputs;
		final int gen_size = genotype.size();				//store the genontype size
		
		final int blocks_length = length_I * length_O;		//length of the blocks of the genotype
		this.length_H = gen_size / blocks_length - 1;		// -1 because of the i_o
		
		//arrays for storing values of the neurons
		neurons_I = new double[length_I];					//input value
		neurons_H = new double[length_H];					//length_H value
		neurons_O = new double[length_O];					//output value
		
		//arrays for storing mapping of weights
		mapping_I_O = new Byte[length_I][length_O];		//input -> output mapping
		mapping_H_I = new Byte[length_H][length_I];			//length_H -> input mapping
		mapping_H_O = new Byte[length_H][length_O];		//length_H -> output mapping
		
		//arrays for storing values of the weights
		weights_I_O = new double[length_I][length_O];		//input -> output weight
		weights_H_I = new double[length_H][length_I];		//length_H -> input weight
		weights_H_O = new double[length_H][length_O];		//length_H -> output weight
		
		weights_H_BIAS = new double[length_H];				//length_H bias weight
		weights_O_BIAS = new double[length_O];				//output bias weight
		
		//Mapping of the connections of the ann.
		WeightMapping(genotype, gen_size, blocks_length);
		
		//first random weights
		WeightsGen();
	}
	
	/**
	 * This method do the mapping of the connections of the ann.
	 * @param genotype
	 * @param gen_size
	 * @param blocks_length
	 * @param length_I
	 * @param length_O
	 */
	private void WeightMapping(ArrayList<Byte> genotype, final int gen_size, final int blocks_length)
	{
		for (int i = 0; i < gen_size; i++)
		{
			Byte val = genotype.get(i);
			
			//input ->  output connections mapping
			if(i < blocks_length)
			{
				int input = 0;
				int substraction = i;
				
				while(substraction >= length_O)
				{
					input++;
					substraction -= length_O;
				}				
				
				mapping_I_O[input][substraction] = val;
			}
				
			//length_H connections mapping
			else 
			{
				int length_H_neuron = 0;
				int substraction = i - blocks_length;
				
				while(substraction >= blocks_length)
				{
					length_H_neuron++;
					substraction -= blocks_length;
				}				
				
				int input_index, output_index;

				
				if(length_I - 1 == 0)
					input_index = 0;
				else
					input_index = substraction / (length_I - 1);

				output_index = substraction % (length_O);

				
				//input ->  length_H connections mapping
				if(mapping_H_I[length_H_neuron][input_index] == null || mapping_H_I[length_H_neuron][input_index] == 0)
					mapping_H_I[length_H_neuron][input_index] = val;
				
				//length_H ->  output connections mapping
				if(mapping_H_O[length_H_neuron][output_index] == null || mapping_H_O[length_H_neuron][output_index] == 0)
					mapping_H_O[length_H_neuron][output_index] = val;
			}
		}
		
		if(Const.DEBUG)
			PrintWeightMapping();
	}
	
	private void WeightsGen()
	{		
		Random rand = new Random();
		
		for (int i = 0; i < length_H ; i++)
		{
			//bias
			weights_H_BIAS[i] = rand.nextDouble() * (1 - -1) + -1;
			
			for (int j = 0; j < length_I ; j++)
			{
				if(mapping_H_I[i][j] == 1)
				{	
					weights_H_I[i][j] = rand.nextDouble() * (1 - -1) + -1;
				}
			}
			
			for (int j = 0; j < length_O ; j++)
			{
				if(mapping_H_O[i][j] == 1)
				{
					weights_H_O[i][j] = rand.nextDouble() * (1 - -1) + -1;
				}
			}
		}
		
		for (int i = 0; i < length_O ; i++)
		{
			//bias
			weights_O_BIAS[i] = rand.nextDouble() * (1 - -1) + -1;
			
			for (int j = 0; j < length_I ; j++)
			{
				if(mapping_I_O[j][i] == 1)
				{	
					weights_I_O[j][i] = rand.nextDouble() * (1 - -1) + -1;
				}
			}
		}
		/*
		//ñapa para debuguear
		weights_I_O[0][0] = 0.0193;
		weights_I_O[1][0] = 0.3838;
		
		weights_H_I[0][0] = -0.9561;
		weights_H_I[0][1] = -0.3124;
		weights_H_BIAS[0] = -0.3428;
		
		weights_H_O[0][0] = -0.8003;
		weights_O_BIAS[0] = 0.0914;
		*/
		if(Const.DEBUG)
			System.out.println("First random weights calculated.");
	}
	
	public void TrainingOffline(int training_iterations)
	{	
		int sets = 4;
		int min = 0;
		boolean binary = true;
		
		DataGen datagen = new DataGen(length_I, sets, min, binary);
		if(Const.DEBUG)
			datagen.PrintDataSet();
		
		double[][] dataset = datagen.GetDataset();
		
		for(int i = 0 ; i < training_iterations ; i++)
		{
			System.out.println("____________________________ITERATION___"  + i + " of "+ training_iterations);
			
			if(Const.DEBUG)
				PrintWeights();
					
			//RESET DELTAS
			deltas_I_O = new double[length_I][length_O];		//input -> output weight
			deltas_H_I = new double[length_H][length_I];		//input -> length_H weight
			deltas_H_O = new double[length_H][length_O];		//length_H -> output weight
			
			deltas_H_BIAS = new double[length_H];
			deltas_O_BIAS = new double[length_O];
			
			double error = 0;
			for (int j = 0, max = dataset.length; j < max ; j++)
			{	
				
				FeedForward(dataset,j);
				BackPropagation();
				DeltaWeights();
				
				double current_error = 0;
				for (int k = 0; k < length_O ; k++)
				{
					current_error = ExpectedValue_XOR(0) - neurons_O[k];
					error += Math.pow(current_error,2);
					//error +=errors_O[k];
				}				
				
				//System.out.println("________________________________________________________HIDDEN_____" + neurons_H[0]);
				System.out.println("________________________________________________________EXPECTED_____" + ExpectedValue_XOR(0));
				System.out.println("____________________________________________________ERROR NEURON_____" + current_error);
				System.out.println("__________________________________________________________NEURON_____" + neurons_O[0]);
				System.out.println("");
			}
			//Math.pow(error,2);
			error /= dataset.length;
			
			System.out.println("________________________________________________________GLOBAL_ERROR___" + error);
			System.out.println("\n");
			
			if(error < Const.FITNESS)
			{
				System.out.println("VICTORYYYYYYYY");
				break;
			}
			WeightsCorrection();
		}
		FeedForward(dataset,0);
		FeedForward(dataset,1);
		FeedForward(dataset,2);
		FeedForward(dataset,3);
		System.out.println("END");
	}
	
	private void FeedForward(double[][] dataset, int dataset_iteration)
	{	
		//reset neurons
		neurons_I = new double[length_I];					//input value
		neurons_H = new double[length_H];					//length_H value
		neurons_O = new double[length_O];					//output value
		
		//output of input neurons
		for (int i = 0; i < length_I; i++)
		{
			//fill input neurons with values in this iteration of dataset 
			neurons_I[i] = dataset[dataset_iteration][i];
		}
		
		//real output of length_H neurons
		for (int i = 0; i < length_H ; i++)
		{
			for (int j = 0; j < length_I ; j++)
			{
				if(mapping_H_I[i][j] == 1)
				{	
					neurons_H[i] += neurons_I[j] *  weights_H_I[i][j];
				}
			}
			
			neurons_H[i] += weights_H_BIAS[i];
			
			//Activation function
			if(Const.AFUNC == Activation.TANH)
			{
				//mytan
				//neurons_H[i] = HyperbolicTan(neurons_H[i]);
				//tanh
				neurons_H[i] = Math.tanh(neurons_H[i]);
			}
			else if(Const.AFUNC == Activation.SIGMOID)
			{
				//sigmoid
				neurons_H[i] = Sigmoid(neurons_H[i]);
			}
			else if(Const.AFUNC == Activation.UMBRAL)
			{
				//jump
				if(neurons_H[i] < 0.5)
					neurons_H[i] = 0;
				else if (neurons_H[i] >= 0.5)
					neurons_H[i] = 1;
			}
		}
		
		//real output of the output neurons
		for (int i = 0; i < length_O ; i++)
		{
			//first i_o
			for (int j = 0; j < length_I ; j++)
			{
				if(mapping_I_O[j][i] == 1)
				{	
					neurons_O[i] += neurons_I[j] *  weights_I_O[j][i];
				}
			}
			//then h_o
			for (int j = 0; j < length_H ; j++)
			{
				if(mapping_H_O[j][i] == 1)
				{	
					neurons_O[i] += neurons_H[j] *  weights_H_O[j][i];
				}
			}
			
			neurons_O[i] += weights_O_BIAS[i];
			
			//Activation function
			if(Const.AFUNC == Activation.TANH)
			{
				//mytan
				//neurons_O[i] = HyperbolicTan(neurons_O[i]);
				//tanh
				neurons_O[i] = Math.tanh(neurons_O[i]);
			}
			else if(Const.AFUNC == Activation.SIGMOID)
			{
				//sigmoid
				neurons_O[i] = Sigmoid(neurons_O[i]);
			}
			else if(Const.AFUNC == Activation.UMBRAL)
			{
				//jump
				if(neurons_O[i] < 0.5)
					neurons_O[i] = 0;
				else if (neurons_O[i] >= 0.5)
					neurons_O[i] = 1;
			}
			
				
		}
		if(Const.DEBUG)
			PrintNeuronsValues(dataset_iteration);
	}
	
	private void BackPropagation()
	{		
		errors_O = new double[length_O];
		
		for (int i = 0; i < length_O ; i++)
		{
			if(Const.AFUNC == Activation.TANH)
				errors_O[i] = 1 - Math.pow(neurons_O[i], 2) * (ExpectedValue_XOR(i) - neurons_O[i]);
			else if(Const.AFUNC == Activation.SIGMOID)
				errors_O[i] = neurons_O[i] * (1 - neurons_O[i]) * (ExpectedValue_XOR(i) - neurons_O[i]);
			else if(Const.AFUNC == Activation.UMBRAL)
				errors_O[i] = 1 * (ExpectedValue_XOR(i) - neurons_O[i]);
			
			if(Const.DEBUG)
				System.out.println("output error_ " + i + "____" + errors_O[i]);
		}
		
		//length_H errors
		errors_H = new double[length_H];
		
		for (int i = 0; i < length_H ; i++)
		{
			
			double sum_Eo_Who = 0;
			
			for (int j = 0; j < length_O ; j++)
			{
				if(mapping_H_O[i][j] == 1)
				{	
					sum_Eo_Who += weights_H_O[i][j] * errors_O[j];
				}
			}
			
			if(Const.AFUNC == Activation.TANH)
				errors_H[i] = 1 - Math.pow(neurons_H[i], 2) * (ExpectedValue_XOR(i) - neurons_H[i]);
			else if(Const.AFUNC == Activation.SIGMOID)
				errors_H[i] = neurons_H[i] * (1 - neurons_H[i]) * sum_Eo_Who;
			else if(Const.AFUNC == Activation.UMBRAL)
				errors_H[i] = 1 * sum_Eo_Who;
			
			if(Const.DEBUG)
				System.out.println("length_H error_ " + i + "____" + errors_H[i]);
		}
	}
	
	private void DeltaWeights()
	{	
		//deltas of i_o
		for(int i = 0; i < length_O ; i++)
		{
			//bias
			deltas_O_BIAS[i] += learn_factor * errors_O[i];
			
			for(int j = 0; j < length_I ; j++)
			{
				if(mapping_I_O[j][i] == 1)
					deltas_I_O[j][i] += learn_factor * errors_O[i] *  neurons_I[j];
			}
		}
		
		//deltas of h_o
		for(int i = 0; i < length_H ; i++)
		{
			//bias
			deltas_H_BIAS[i] += learn_factor * errors_H[i];
			
			for(int j = 0; j < length_O ; j++)
			{
				if(mapping_H_O[i][j] == 1)
					deltas_H_O[i][j] += learn_factor * errors_O[j] *  neurons_H[i];
			}
		}
		
		//deltas of i_h
		for(int i = 0; i < length_H ; i++)
		{
			for(int j = 0; j < length_I ; j++)
			{
				if(mapping_H_I[i][j] == 1)
					deltas_H_I[i][j] += learn_factor * errors_H[i] *  neurons_I[j];
			}
		}
		
		if(Const.DEBUG)
			PrintDeltas();
	}
	
	private void WeightsCorrection()
	{		
		//weights of i_o
		for(int i = 0; i < length_I ; i++)
		{
			for(int j = 0; j < length_O ; j++)
			{
				if(mapping_I_O[i][j] == 1)
					weights_I_O[i][j] += deltas_I_O[i][j];
			}
		}
		
		//weights of h_o
		for(int i = 0; i < length_O ; i++)
		{
			//bias
			weights_O_BIAS[i] += deltas_O_BIAS[i];
			
			for(int j = 0; j < length_H ; j++)
			{
				if(mapping_H_O[j][i] == 1)
					weights_H_O[j][i] += deltas_H_O[j][i];
			}
		}
		
		//weights of i_h
		for(int i = 0; i < length_H ; i++)
		{
			//bias
			weights_H_BIAS[i] += deltas_H_BIAS[i];
			
			for(int j = 0; j < length_I ; j++)
			{
				if(mapping_H_I[i][j] == 1)
					weights_H_I[i][j] += deltas_H_I[i][j];
			}
		}
		
		if(Const.DEBUG)
			System.out.println("WEIGHTS CORRECTED");
	}
	
	//XOR
	private int ExpectedValue_XOR(int output)
	{
		switch (output)
		{
			case 0:
			//We don't use different expected values for different OUTPUTS because there is only one.
			if(neurons_I[0] != neurons_I[1])
			{
				if(Const.DEBUG)
					System.out.println("EXPECTED___" + 1);
				return 1;
			}
			else
			{
				if(Const.DEBUG)
					System.out.println("EXPECTED___" + 0);
				return 0;
			}
			default:
				//?????????
				return 999999;
		}
	}

	/////////////////////////////////////TESTING METHODS/////////////////////////////////
	
	public void PrintNeuronsValues(int dataset_iteration)
	{
		System.out.println("###_NEURONS VALUES_### DATASET_ITERATION____"+ dataset_iteration +"\n");
		System.out.println("I_values: ");
		
		for(int i = 0; i < length_I ; i++ )
		{
			System.out.print("I[" + i + "]:__ ");
			System.out.print(neurons_I[i]);
			if(i + 1 < length_I)
				System.out.print("\n");
			else
				System.out.print("\n\n");
		}
		
		for(int i = 0; i < length_H ; i++ )
		{
			System.out.print("H[" + i + "]:__ ");
			System.out.print(neurons_H[i]);
			if(i + 1 < length_H)
				System.out.print("\n");
			else
				System.out.print("\n\n");
		}
		
		for(int i = 0; i < length_O ; i++ )
		{
			System.out.print("O[" + i + "]:__ ");
			System.out.print(neurons_O[i]);
			if(i + 1 < length_O)
				System.out.print("\n");
			else
				System.out.print("\n\n");
		}
	}
	
	public void PrintWeightMapping()
	{
		System.out.println("###_WEIGHT MAPPING_###\n");
		System.out.println("IO_WM: ");
		for(int i = 0; i < length_I ; i++ )
		{
			for(int j = 0; j < length_O ; j++ )
			{
				System.out.print("I[" + i + "] -> O[" + j + "]:__ ");
				System.out.print(mapping_I_O[i][j]);
				if(j + 1 < length_O)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		System.out.print("\n\n");
		
		System.out.println("HI_WM: ");
		for(int i = 0; i < length_H ; i++ )
		{
			for(int j = 0; j < length_I ; j++ )
			{
				System.out.print("H[" + i + "] -> I[" + j + "]:__ ");
				System.out.print(mapping_H_I[i][j]);
				if(j + 1 < length_I)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		System.out.print("\n\n");
		
		System.out.println("HO_WM: ");
		for(int i = 0; i < length_H ; i++ )
		{
			for(int j = 0; j < length_O ; j++ )
			{
				System.out.print("H[" + i + "] -> O[" + j + "]:__ ");
				System.out.print(mapping_H_O[i][j]);
				if(j + 1 < length_O)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		System.out.print("#############\n");
	}
	
	public void PrintWeights()
	{
		System.out.println("###_WEIGHTS_###\n");
		System.out.println("IO_WEIGHTS: ");
		for(int i = 0; i < length_I ; i++ )
		{
			for(int j = 0; j < length_O ; j++ )
			{
				System.out.print("I[" + i + "] -> O[" + j + "]:__ ");
				System.out.print(weights_I_O[i][j]);
				if(j + 1 < length_O)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		System.out.print("\n\n");
		
		System.out.println("HI_WEIGHTS: ");
		for(int i = 0; i < length_H ; i++ )
		{
			for(int j = 0; j < length_I ; j++ )
			{
				System.out.print("H[" + i + "] -> I[" + j + "]:__ ");
				System.out.print(weights_H_I[i][j]);
				if(j + 1 < length_I)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		System.out.print("\n\n");
		
		System.out.println("H_BIAS_WEIGHTS: ");
		for(int i = 0; i < length_H ; i++ )
		{
			System.out.print("H_BIAS[" + i + "]:__ ");
			System.out.print(weights_H_BIAS[i]);
				System.out.print("\n");
		}
		System.out.print("\n\n");
		
		System.out.println("HO_WEIGHTS: ");
		for(int i = 0; i < length_H ; i++ )
		{
			for(int j = 0; j < length_O ; j++ )
			{
				System.out.print("H[" + i + "] -> O[" + j + "]:__ ");
				System.out.print(weights_H_O[i][j]);
				if(j + 1 < length_O)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		
		System.out.println("O_BIAS_WEIGHTS: ");
		for(int i = 0; i < length_O ; i++ )
		{
			System.out.print("O_BIAS[" + i + "]:__ ");
			System.out.print(weights_O_BIAS[i]);
				System.out.print("\n");
		}
		System.out.print("\n\n");
		
		System.out.print("#############\n");
	}
	
	public void PrintDeltas()
	{
		System.out.println("###_DELTAS_###\n");
		System.out.println("IO_DELTAS: ");
		for(int i = 0; i < length_I ; i++ )
		{
			for(int j = 0; j < length_O ; j++ )
			{
				System.out.print("I[" + i + "] -> O[" + j + "]:__ ");
				System.out.print(deltas_I_O[i][j]);
				if(j + 1 < length_O)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		System.out.print("\n\n");
		
		System.out.println("HI_DELTAS: ");
		for(int i = 0; i < length_H ; i++ )
		{
			for(int j = 0; j < length_I ; j++ )
			{
				System.out.print("H[" + i + "] -> I[" + j + "]:__ ");
				System.out.print(deltas_H_I[i][j]);
				if(j + 1 < length_I)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		System.out.print("\n\n");
		
		System.out.println("H_BIAS_DELTAS: ");
		for(int i = 0; i < length_H ; i++ )
		{
			System.out.print("H_BIAS[" + i + "]:__ ");
			System.out.print(deltas_H_BIAS[i]);
				System.out.print("\n");
		}
		System.out.print("\n\n");
		
		System.out.println("HO_DELTAS: ");
		for(int i = 0; i < length_H ; i++ )
		{
			for(int j = 0; j < length_O ; j++ )
			{
				System.out.print("H[" + i + "] -> O[" + j + "]:__ ");
				System.out.print(deltas_H_O[i][j]);
				if(j + 1 < length_O)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		
		System.out.println("O_BIAS_DELTAS: ");
		for(int i = 0; i < length_O ; i++ )
		{
			System.out.print("O_BIAS[" + i + "]:__ ");
			System.out.print(deltas_O_BIAS[i]);
				System.out.print("\n");
		}
		System.out.print("\n\n");
		
		System.out.print("#############\n");
	}
	
	//from 0 to 1
	public static double Sigmoid(double x) {
	    return (1/( 1 + Math.pow(Math.E,(-1*x))));
	}
	
	//from -1 to 1
		public static double HyperbolicTan(double x) {
		    return ((1 - Math.pow(Math.E,(-2*x))) / (1 + Math.pow(Math.E,(-2*x))));
		}
		
	public static double ArcHyperbolicTan(double x) 
	{ 
		return 0.5*Math.log( (x + 1.0) / (x - 1.0) ); 
	} 
}
