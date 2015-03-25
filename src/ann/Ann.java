package ann;

import java.util.ArrayList;
import java.util.Random;

import ann.Const.Activation;
import dataset.DataGen;

/**
 * Artificial neural network
 * @author Daniel Castaño Estrella
 *
 */
public class Ann
{
	//LEARN FACTOR
	public double learn_factor;
	
	//MAPPING
	Byte[][] 	i_o_weights_mapping;	//input -> output mapping
	Byte[][] 	h_i_weights_mapping;	//input -> hidden mapping
	Byte[][]	h_o_weights_mapping;	//hidden -> output mapping
	
	//WEIGHTS
	double[][] 	i_o_weights;			//input -> output weight
	double[][] 	h_i_weights;			//input -> hidden weight
	double[][] 	h_o_weights;			//hidden -> output weight
	
	double[] 	h_bias_weights;			//hidden bias weight
	double[] 	o_bias_weights;			//output bias weight
	
	//VALUES
	double[] 	i_neurons;				//input values
	double[] 	h_neurons;				//hidden values
	double[] 	o_neurons;				//output values
	
	//ERRORS
	double[] 	h_neurons_errors;		//hidden errors
	double[] 	o_neurons_errors;		//output errors
	
	//DELTAS
	double[][] 	i_o_deltas;				//input -> output deltas
	double[][] 	h_i_deltas;				//input -> hidden deltas
	double[][] 	h_o_deltas;				//hidden -> output deltas
	
	double[] 	h_bias_deltas;			//hidden bias deltas
	double[] 	o_bias_deltas;			//output bias deltas
	
	public Ann(ArrayList<Byte> genotype, int inputs, int outputs, double learn_factor)
	{
		this.learn_factor = learn_factor;					//store the learn factor
		final int gen_size = genotype.size();				//store the genontype size
		
		final int blocks_length = inputs*outputs;			//length of the blocks of the genotype
		final int hidden = gen_size / blocks_length - 1;	// -1 because of the i_o
		
		//arrays for storing values of the neurons
		i_neurons = new double[inputs];						//input value
		h_neurons = new double[hidden];						//hidden value
		o_neurons = new double[outputs];					//output value
		
		//arrays for storing mapping of weights
		i_o_weights_mapping = new Byte[inputs][outputs];	//input -> output mapping
		h_i_weights_mapping = new Byte[hidden][inputs];		//hidden -> input mapping
		h_o_weights_mapping = new Byte[hidden][outputs];	//hidden -> output mapping
		
		//arrays for storing values of the weights
		i_o_weights = new double[inputs][outputs];			//input -> output weight
		h_i_weights = new double[hidden][inputs];			//hidden -> input weight
		h_o_weights = new double[hidden][outputs];			//hidden -> output weight
		
		h_bias_weights = new double[hidden];				//hidden bias weight
		o_bias_weights = new double[outputs];				//output bias weight
		
		//Mapping of the connections of the ann.
		WeightMapping(genotype, gen_size, blocks_length, inputs, outputs);
		
		//first random weights
		WeightsGen();
	}
	
	/**
	 * This method do the mapping of the connections of the ann.
	 * @param genotype
	 * @param gen_size
	 * @param blocks_length
	 * @param inputs
	 * @param outputs
	 */
	private void WeightMapping(ArrayList<Byte> genotype, final int gen_size, final int blocks_length, final int inputs, final int outputs)
	{
		for (int i = 0; i < gen_size; i++)
		{
			Byte val = genotype.get(i);
			
			//input ->  output connections mapping
			if(i < blocks_length)
			{
				int input = 0;
				int substraction = i;
				
				while(substraction >= outputs)
				{
					input++;
					substraction -= outputs;
				}				
				
				i_o_weights_mapping[input][substraction] = val;
			}
				
			//hidden connections mapping
			else 
			{
				int hidden_neuron = 0;
				int substraction = i - blocks_length;
				
				while(substraction >= blocks_length)
				{
					hidden_neuron++;
					substraction -= blocks_length;
				}				
				
				int input_index, output_index;

				
				if(inputs - 1 == 0)
					input_index = 0;
				else
					input_index = substraction / (inputs - 1);

				output_index = substraction % (outputs);

				
				//input ->  hidden connections mapping
				if(h_i_weights_mapping[hidden_neuron][input_index] == null || h_i_weights_mapping[hidden_neuron][input_index] == 0)
					h_i_weights_mapping[hidden_neuron][input_index] = val;
				
				//hidden ->  output connections mapping
				if(h_o_weights_mapping[hidden_neuron][output_index] == null || h_o_weights_mapping[hidden_neuron][output_index] == 0)
					h_o_weights_mapping[hidden_neuron][output_index] = val;
			}
		}
		
		if(Const.DEBUG)
			PrintWeightMapping();
	}
	
	private void WeightsGen()
	{
		int i_size = i_neurons.length;
		int h_size = h_neurons.length;
		int o_size = o_neurons.length;
		
		Random rand = new Random();
		
		for (int i = 0; i < h_size ; i++)
		{
			//bias
			h_bias_weights[i] = rand.nextDouble() * (1 - -1) + -1;
			
			for (int j = 0; j < i_size ; j++)
			{
				if(h_i_weights_mapping[i][j] == 1)
				{	
					h_i_weights[i][j] = rand.nextDouble() * (1 - -1) + -1;
				}
			}
			
			for (int j = 0; j < o_size ; j++)
			{
				if(h_o_weights_mapping[i][j] == 1)
				{
					h_o_weights[i][j] = rand.nextDouble() * (1 - -1) + -1;
				}
			}
		}
		
		for (int i = 0; i < o_size ; i++)
		{
			//bias
			o_bias_weights[i] = rand.nextDouble() * (1 - -1) + -1;
			
			for (int j = 0; j < i_size ; j++)
			{
				if(i_o_weights_mapping[j][i] == 1)
				{	
					i_o_weights[j][i] = rand.nextDouble() * (1 - -1) + -1;
				}
			}
		}
		if(Const.DEBUG)
			System.out.println("First random weights calculated.");
	}
	
	public void TrainingOffline(int training_iterations)
	{
		int i_size = i_neurons.length;
		int h_size = h_neurons.length;
		int o_size = o_neurons.length;
		
		int sets = 4;
		int min = 0;
		boolean binary = true;
		
		DataGen datagen = new DataGen(i_size, sets, min, binary);
		if(Const.DEBUG)
			datagen.PrintDataSet();
		
		double[][] dataset = datagen.GetDataset();
		
		for(int i = 0 ; i < training_iterations ; i++)
		{
			System.out.println("____________________________ITERATION___"  + i + " of "+ training_iterations);
			
			if(Const.DEBUG)
				PrintWeights();
			
			//RESET DELTAS
			i_o_deltas = new double[i_size][o_size];		//input -> output weight
			h_i_deltas = new double[h_size][i_size];		//input -> hidden weight
			h_o_deltas = new double[h_size][o_size];		//hidden -> output weight
			
			h_bias_deltas = new double[h_size];
			o_bias_deltas = new double[o_size];
			
			double error = 0;
			for (int j = 0, max = dataset.length; j < max ; j++)
			{	
				FeedForward(dataset,j);
				BackPropagation();
				
				for (int k = 0, max2 = o_neurons.length; k < max2 ; k++)
				{
					error += Math.pow( ExpectedValue_XOR(0) - o_neurons[k],2);
					//error +=o_neurons_errors[k];
				}				
				DeltaWeights();
				if(Const.DEBUG)
					PrintDeltas();
				//System.out.println("__________________________________________________________HIDDEN_____" + h_neurons[0]);
				System.out.println("________________________________________________________EXPECTED_____" + ExpectedValue_XOR(0));
				System.out.println("____________________________________________________ERROR NEURON_____" + o_neurons_errors[0]);
				System.out.println("__________________________________________________________NEURON_____" + o_neurons[0]);
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
		System.out.println("END");
	}
	
	private void FeedForward(double[][] dataset, int dataset_iteration)
	{
		int h_size = h_neurons.length;
		int i_size = i_neurons.length;
		int o_size = o_neurons.length;
		
		//output of input neurons
		for (int i = 0; i < i_size; i++)
		{
			//fill input neurons with values in this iteration of dataset 
			i_neurons[i] = dataset[dataset_iteration][i];
		}
		
		//real output of hidden neurons
		for (int i = 0; i < h_size ; i++)
		{
			for (int j = 0; j < i_size ; j++)
			{
				if(h_i_weights_mapping[i][j] == 1)
				{	
					h_neurons[i] += i_neurons[j] *  h_i_weights[i][j];
				}
			}
			
			h_neurons[i] += h_bias_weights[i];
			
			//Activation function
			if(Const.AFUNC == Activation.TANH)
			{
				//mytan
				h_neurons[i] = HyperbolicTan(h_neurons[i]);
				//tanh
				//h_neurons[i] = Math.tanh(h_neurons[i]);
			}
			else if(Const.AFUNC == Activation.SIGMOID)
			{
				//sigmoid
				h_neurons[i] = Sigmoid(h_neurons[i]);
			}
			else if(Const.AFUNC == Activation.UMBRAL)
			{
				//jump
				if(h_neurons[i] < 0.5)
					h_neurons[i] = 0;
				else if (h_neurons[i] >= 0.5)
					h_neurons[i] = 1;
			}
		}
		
		//real output of the output neurons
		for (int i = 0; i < o_size ; i++)
		{
			//first i_o
			for (int j = 0; j < i_size ; j++)
			{
				if(i_o_weights_mapping[j][i] == 1)
				{	
					o_neurons[i] += i_neurons[j] *  i_o_weights[j][i];
				}
			}
			//then h_o
			for (int j = 0; j < h_size ; j++)
			{
				if(h_o_weights_mapping[j][i] == 1)
				{	
					o_neurons[i] += h_neurons[j] *  h_o_weights[j][i];
				}
			}
			
			o_neurons[i] += o_bias_weights[i];
			
			//Activation function
			if(Const.AFUNC == Activation.TANH)
			{
				//mytan
				o_neurons[i] = HyperbolicTan(o_neurons[i]);
				//tanh
				//o_neurons[i] = Math.tanh(o_neurons[i]);
			}
			else if(Const.AFUNC == Activation.SIGMOID)
			{
				//sigmoid
				o_neurons[i] = Sigmoid(o_neurons[i]);
			}
			else if(Const.AFUNC == Activation.UMBRAL)
			{
				//jump
				if(o_neurons[i] < 0.5)
					o_neurons[i] = 0;
				else if (o_neurons[i] >= 0.5)
					o_neurons[i] = 1;
			}
			
				
		}
		if(Const.DEBUG)
			PrintNeuronsValues(dataset_iteration);
	}
	
	private void BackPropagation()
	{		
		//output errors
		int o_size = o_neurons.length;
		o_neurons_errors = new double[o_size];
		
		for (int i = 0; i < o_size ; i++)
		{
			if(Const.AFUNC == Activation.TANH)
			{}
			else if(Const.AFUNC == Activation.SIGMOID)
				o_neurons_errors[i] = o_neurons[i] * (1 - o_neurons[i]) * (ExpectedValue_XOR(i) - o_neurons[i]);
			else if(Const.AFUNC == Activation.UMBRAL)
				o_neurons_errors[i] = 1 * (ExpectedValue_XOR(i) - o_neurons[i]);
			
			if(Const.DEBUG)
				System.out.println("output error_ " + i + "____" + o_neurons_errors[i]);
		}
		
		//hidden errors
		int h_size = h_neurons.length;
		h_neurons_errors = new double[h_size];
		
		for (int i = 0; i < h_size ; i++)
		{
			
			double sum_Eo_Who = 0;
			
			for (int j = 0; j < o_size ; j++)
			{
				if(h_o_weights_mapping[i][j] == 1)
				{	
					sum_Eo_Who += h_o_weights[i][j] * o_neurons_errors[j];
				}
			}
			
			if(Const.AFUNC == Activation.TANH)
			{}
			else if(Const.AFUNC == Activation.SIGMOID)
				h_neurons_errors[i] = h_neurons[i] * (1 - h_neurons[i]) * sum_Eo_Who;
			else if(Const.AFUNC == Activation.UMBRAL)
				h_neurons_errors[i] = 1 * sum_Eo_Who;
			
			if(Const.DEBUG)
				System.out.println("hidden error_ " + i + "____" + h_neurons_errors[i]);
		}
	}
	
	private void DeltaWeights()
	{
		int i_size = i_neurons.length;
		int h_size = h_neurons.length;
		int o_size = o_neurons.length;
		
		//deltas of i_o
		for(int i = 0; i < o_size ; i++)
		{
			//bias
			o_bias_deltas[i] += learn_factor * o_neurons_errors[i];
			
			for(int j = 0; j < i_size ; j++)
			{
				if(i_o_weights_mapping[j][i] == 1)
					i_o_deltas[j][i] += learn_factor * o_neurons_errors[i] *  i_neurons[j];
			}
		}
		
		//deltas of h_o
		for(int i = 0; i < h_size ; i++)
		{
			//bias
			h_bias_deltas[i] += learn_factor * h_neurons_errors[i];
			
			for(int j = 0; j < o_size ; j++)
			{
				if(h_o_weights_mapping[i][j] == 1)
					h_o_deltas[i][j] += learn_factor * o_neurons_errors[j] *  h_neurons[i];
			}
		}
		
		//deltas of i_h
		for(int i = 0; i < h_size ; i++)
		{
			for(int j = 0; j < i_size ; j++)
			{
				if(h_i_weights_mapping[i][j] == 1)
					h_i_deltas[i][j] += learn_factor * h_neurons_errors[i] *  i_neurons[j];
			}
		}
	}
	
	private void WeightsCorrection()
	{
		int i_size = i_neurons.length;
		int h_size = h_neurons.length;
		int o_size = o_neurons.length;
		
		//weights of i_o
		for(int i = 0; i < i_size ; i++)
		{
			for(int j = 0; j < o_size ; j++)
			{
				if(i_o_weights_mapping[i][j] == 1)
					i_o_weights[i][j] += i_o_deltas[i][j];
			}
		}
		
		//weights of h_o
		for(int i = 0; i < o_size ; i++)
		{
			//bias
			o_bias_deltas[i] += o_bias_deltas[i];
			
			for(int j = 0; j < h_size ; j++)
			{
				if(h_o_weights_mapping[j][i] == 1)
					h_o_weights[j][i] += h_o_deltas[j][i];
			}
		}
		
		//weights of i_h
		for(int i = 0; i < h_size ; i++)
		{
			//bias
			h_bias_deltas[i] += h_bias_deltas[i];
			
			for(int j = 0; j < i_size ; j++)
			{
				if(h_i_weights_mapping[i][j] == 1)
					h_i_weights[i][j] += h_i_deltas[i][j];
			}
		}
	}
	
	//XOR
	private int ExpectedValue_XOR(int output)
	{
		switch (output)
		{
			case 0:
			//We don't use different expected values for different outputs because there is only one.
			if(i_neurons[0] != i_neurons[1])
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
		
		for(int i = 0, max = i_neurons.length; i < max ; i++ )
		{
			System.out.print("I[" + i + "]:__ ");
			System.out.print(i_neurons[i]);
			if(i + 1 < max)
				System.out.print("\n");
			else
				System.out.print("\n\n");
		}
		
		for(int i = 0, max = h_neurons.length; i < max ; i++ )
		{
			System.out.print("H[" + i + "]:__ ");
			System.out.print(h_neurons[i]);
			if(i + 1 < max)
				System.out.print("\n");
			else
				System.out.print("\n\n");
		}
		
		for(int i = 0, max = o_neurons.length; i < max ; i++ )
		{
			System.out.print("O[" + i + "]:__ ");
			System.out.print(o_neurons[i]);
			if(i + 1 < max)
				System.out.print("\n");
			else
				System.out.print("\n\n");
		}
	}
	
	public void PrintWeightMapping()
	{
		System.out.println("###_WEIGHT MAPPING_###\n");
		System.out.println("IO_WM: ");
		for(int i = 0, max = i_o_weights_mapping.length; i < max ; i++ )
		{
			for(int j = 0, max2 = i_o_weights_mapping[0].length; j < max2 ; j++ )
			{
				System.out.print("I[" + i + "] -> O[" + j + "]:__ ");
				System.out.print(i_o_weights_mapping[i][j]);
				if(j + 1 < max2)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		System.out.print("\n\n");
		
		System.out.println("HI_WM: ");
		for(int i = 0, max = h_i_weights_mapping.length; i < max ; i++ )
		{
			for(int j = 0, max2 = h_i_weights_mapping[i].length; j < max2 ; j++ )
			{
				System.out.print("H[" + i + "] -> I[" + j + "]:__ ");
				System.out.print(h_i_weights_mapping[i][j]);
				if(j + 1 < max2)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		System.out.print("\n\n");
		
		System.out.println("HO_WM: ");
		for(int i = 0, max = h_o_weights_mapping.length; i < max ; i++ )
		{
			for(int j = 0, max2 = h_o_weights_mapping[i].length; j < max2 ; j++ )
			{
				System.out.print("H[" + i + "] -> O[" + j + "]:__ ");
				System.out.print(h_o_weights_mapping[i][j]);
				if(j + 1 < max2)
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
		for(int i = 0, max = i_o_weights.length; i < max ; i++ )
		{
			for(int j = 0, max2 = i_o_weights[0].length; j < max2 ; j++ )
			{
				System.out.print("I[" + i + "] -> O[" + j + "]:__ ");
				System.out.print(i_o_weights[i][j]);
				if(j + 1 < max2)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		System.out.print("\n\n");
		
		System.out.println("HI_WEIGHTS: ");
		for(int i = 0, max = h_i_weights.length; i < max ; i++ )
		{
			for(int j = 0, max2 = h_i_weights[i].length; j < max2 ; j++ )
			{
				System.out.print("H[" + i + "] -> I[" + j + "]:__ ");
				System.out.print(h_i_weights[i][j]);
				if(j + 1 < max2)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		System.out.print("\n\n");
		
		System.out.println("H_BIAS_WEIGHTS: ");
		for(int i = 0, max = h_bias_weights.length; i < max ; i++ )
		{
			System.out.print("H_BIAS[" + i + "]:__ ");
			System.out.print(h_bias_weights[i]);
				System.out.print("\n");
		}
		System.out.print("\n\n");
		
		System.out.println("HO_WEIGHTS: ");
		for(int i = 0, max = h_o_weights.length; i < max ; i++ )
		{
			for(int j = 0, max2 = h_o_weights[i].length; j < max2 ; j++ )
			{
				System.out.print("H[" + i + "] -> O[" + j + "]:__ ");
				System.out.print(h_o_weights[i][j]);
				if(j + 1 < max2)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		
		System.out.println("O_BIAS_WEIGHTS: ");
		for(int i = 0, max = o_bias_weights.length; i < max ; i++ )
		{
			System.out.print("O_BIAS[" + i + "]:__ ");
			System.out.print(o_bias_weights[i]);
				System.out.print("\n");
		}
		System.out.print("\n\n");
		
		System.out.print("#############\n");
	}
	
	public void PrintDeltas()
	{
		System.out.println("###_DELTAS_###\n");
		System.out.println("IO_DELTAS: ");
		for(int i = 0, max = i_o_deltas.length; i < max ; i++ )
		{
			for(int j = 0, max2 = i_o_deltas[0].length; j < max2 ; j++ )
			{
				System.out.print("I[" + i + "] -> O[" + j + "]:__ ");
				System.out.print(i_o_deltas[i][j]);
				if(j + 1 < max2)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		System.out.print("\n\n");
		
		System.out.println("HI_DELTAS: ");
		for(int i = 0, max = h_i_deltas.length; i < max ; i++ )
		{
			for(int j = 0, max2 = h_i_deltas[i].length; j < max2 ; j++ )
			{
				System.out.print("H[" + i + "] -> I[" + j + "]:__ ");
				System.out.print(h_i_deltas[i][j]);
				if(j + 1 < max2)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		System.out.print("\n\n");
		
		System.out.println("H_BIAS_DELTAS: ");
		for(int i = 0, max = h_bias_deltas.length; i < max ; i++ )
		{
			System.out.print("H_BIAS[" + i + "]:__ ");
			System.out.print(h_bias_deltas[i]);
				System.out.print("\n");
		}
		System.out.print("\n\n");
		
		System.out.println("HO_DELTAS: ");
		for(int i = 0, max = h_o_deltas.length; i < max ; i++ )
		{
			for(int j = 0, max2 = h_o_deltas[i].length; j < max2 ; j++ )
			{
				System.out.print("H[" + i + "] -> O[" + j + "]:__ ");
				System.out.print(h_o_deltas[i][j]);
				if(j + 1 < max2)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		
		System.out.println("O_BIAS_DELTAS: ");
		for(int i = 0, max = o_bias_deltas.length; i < max ; i++ )
		{
			System.out.print("O_BIAS[" + i + "]:__ ");
			System.out.print(o_bias_deltas[i]);
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
}
