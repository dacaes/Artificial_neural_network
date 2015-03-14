package ann;

import java.util.ArrayList;
import java.util.Random;

import dataset.DataGen;

/**
 * Artificial neural network
 * @author Daniel Castaño Estrella
 *
 */
public class Ann
{
	public double learn_factor;
	//mapping of the connections
	Byte[][] 	i_o_weights_mapping;	//input -> output mapping
	Byte[][] 	h_i_weights_mapping;	//input -> hidden mapping
	Byte[][]	h_o_weights_mapping;	//hidden -> output mapping
	
	//arrays for storing values of the neurons
	double[] 	i_neurons;				//input values
	double[] 	h_neurons;				//hidden values
	double[] 	o_neurons;				//output values
	
	//arrays for storing values of the weights
	double[][] 	i_o_weights;			//input -> output weight
	double[][] 	h_i_weights;			//input -> hidden weight
	double[][] 	h_o_weights;			//hidden -> output weight
	
	//array for storing value of hidden and output errors
	double[] 	h_neurons_errors;		//hidden errors
	double[] 	o_neurons_errors;		//output errors
	
	//array for storing deltas of weights
	double[][] 	i_o_deltas;				//input -> output weight
	double[][] 	h_i_deltas;				//input -> hidden weight
	double[][] 	h_o_deltas;				//hidden -> output weight
			
	public Ann(ArrayList<Byte> genotype, int inputs, int outputs, double learn_factor)
	{
		this.learn_factor = learn_factor;
		final int gen_size = genotype.size();
		
		//length of the blocks of the genotype
		final int blocks_length = inputs*outputs;
		final int hidden = gen_size / blocks_length - 1;		// -1 for the i_o
		
		//arrays for storing values of the neurons
		i_neurons = new double[inputs];					//input
		h_neurons = new double[hidden];					//hidden
		o_neurons = new double[outputs];				//expected output
		
		//mapping of weights
		i_o_weights_mapping = new Byte[inputs][outputs];
		h_i_weights_mapping = new Byte[hidden][inputs];
		h_o_weights_mapping = new Byte[hidden][outputs];
		
		//arrays for storing values of the weights
		i_o_weights = new double[inputs][outputs];		//input -> output weight
		h_i_weights = new double[hidden][inputs];		//input -> hidden weight
		h_o_weights = new double[hidden][outputs];		//hidden -> output weight
		
		WeightMapping(genotype, gen_size, blocks_length, inputs, outputs);
		//First weights
		WeightsGen();
	}
	
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
			//deltas
			i_o_deltas = new double[i_size][o_size];		//input -> output weight
			h_i_deltas = new double[h_size][i_size];		//input -> hidden weight
			h_o_deltas = new double[h_size][o_size];		//hidden -> output weight
			
			boolean stop_condition = true;
			
			for (int j = 0, max = dataset.length; j < max ; j++)
			{		
				FeedForward(dataset,j);
				BackPropagation();
				
				stop_condition = true;
				for (int k = 0, max2 = o_neurons_errors.length; k < max2 ; k++)
				{
					if(Math.abs(o_neurons_errors[k]) > Const.FITNESS)
					{
						stop_condition = false;
						break;
					}
				}				
				DeltaWeights();
			}
			if(stop_condition)
			{
				System.out.println("VICTORYYYYYYYY");
			}
			WeightsCorrection();
		}
	}

	private void WeightsGen()
	{
		int i_size = i_neurons.length;
		int h_size = h_neurons.length;
		int o_size = o_neurons.length;
		
		Random rand = new Random();
		
		for (int i = 0; i < h_size ; i++)
		{
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
		
		for (int i = 0; i < i_size ; i++)
		{
			for (int j = 0; j < o_size ; j++)
			{
				if(i_o_weights_mapping[i][j] == 1)
				{	
					i_o_weights[i][j] = rand.nextDouble() * (1 - -1) + -1;
				}
			}
		}
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
			
			//tanh
			h_neurons[i] = Math.tanh(h_neurons[i]);
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
			
			//tanh
			o_neurons[i] = Math.tanh(o_neurons[i]);
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
			o_neurons_errors[i] = o_neurons[i] * (1 - o_neurons[i]) * (ExpectedValue_XOR(i) - o_neurons[i]);
			//if(Const.DEBUG)
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
			
			h_neurons_errors[i] = h_neurons[i] * (1 - h_neurons[i]) * sum_Eo_Who;
			if(Const.DEBUG)
				System.out.println("hidden error_ " + i + "____" + h_neurons_errors[i]);
		}
	}
	
	private void DeltaWeights()
	{
		//deltas of i_o
		for(int i = 0, max = i_o_deltas.length ; i < max ; i++)
		{
			for(int j = 0, max2 = i_o_deltas[0].length ; j < max2 ; j++)
			{
				i_o_deltas[i][j] += learn_factor * o_neurons_errors[j] *  i_neurons[i];
			}
		}
		
		//deltas of h_o
		for(int i = 0, max = h_o_deltas.length ; i < max ; i++)
		{
			for(int j = 0, max2 = h_o_deltas[0].length ; j < max2 ; j++)
			{
				h_o_deltas[i][j] += learn_factor * o_neurons_errors[j] *  h_neurons[i];
			}
		}
		
		//deltas of i_h
		for(int i = 0, max = h_i_deltas.length ; i < max ; i++)
		{
			for(int j = 0, max2 = h_i_deltas[0].length ; j < max2 ; j++)
			{
				h_i_deltas[i][j] += learn_factor * h_neurons_errors[i] *  i_neurons[j];
			}
		}
	}
	
	private void WeightsCorrection()
	{
		//weights of i_o
		for(int i = 0, max = i_o_deltas.length ; i < max ; i++)
		{
			for(int j = 0, max2 = i_o_deltas[0].length ; j < max2 ; j++)
			{
				i_o_weights[i][j] += i_o_deltas[i][j];
			}
		}
		
		//weights of h_o
		for(int i = 0, max = h_o_deltas.length ; i < max ; i++)
		{
			for(int j = 0, max2 = h_o_deltas[0].length ; j < max2 ; j++)
			{
				h_o_weights[i][j] += h_o_deltas[i][j];
			}
		}
		
		//weights of i_h
		for(int i = 0, max = h_i_deltas.length ; i < max ; i++)
		{
			for(int j = 0, max2 = h_i_deltas[0].length ; j < max2 ; j++)
			{
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
				return 0;
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
}
