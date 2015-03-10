package ann;

import java.util.ArrayList;

/**
 * Artificial neural network
 * @author Daniel Castaño Estrella
 *
 */
public class Ann
{
	//mapping of the connections
	Byte[][] 	i_o_weights_mapping;	//input -> output mapping
	Byte[][] 	h_i_weights_mapping;	//input -> hidden mapping
	Byte[][]	h_o_weights_mapping;	//hidden -> output mapping
			
	public Ann(ArrayList<Byte> genotype, int inputs, int outputs)
	{
		final int gen_size = genotype.size();
		
		//length of the blocks of the genotype
		final int blocks_length = inputs*outputs;
		final int hidden = gen_size / blocks_length - 1;		// -1 for the i_o
		
		//arrays for storing values of the neurons
		int[] i_neurons = new int[inputs];						//input
		int[] h_neurons = new int[hidden];						//hidden
		int[] o_neurons_e = new int[outputs];					//expected output
		
		//mapping of weights
		i_o_weights_mapping = new Byte[inputs][outputs];
		h_i_weights_mapping = new Byte[hidden][inputs];
		h_o_weights_mapping = new Byte[hidden][outputs];
		
		//arrays for storing values of the weights
		float[][] 	i_o_weights = new float[inputs][outputs];		//input -> output weight
		float[][] 	i_h_weights = new float[hidden][inputs];		//input -> hidden weight
		float[][] 	h_o_weights = new float[hidden][outputs];		//hidden -> output weight
		
		WeightMapping(genotype, gen_size, blocks_length, inputs, outputs);
	}
	
	private void WeightMapping(ArrayList<Byte> genotype, final int gen_size, final int blocks_length, final int inputs, final int outputs)
	{
		for (int i = 0; i < gen_size; i++)
		{
			byte val = genotype.get(i);
			
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
	
	
	//TESTING METHODS
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
				if(j + 1 < i_o_weights_mapping[i].length)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		System.out.print("\n\n");
		
		System.out.println("HI_WM: ");
		for(int i = 0, max = h_i_weights_mapping.length; i < max ; i++ )
		{
			for(int j = 0; j < h_i_weights_mapping[i].length ; j++ )
			{
				System.out.print("H[" + i + "] -> I[" + j + "]:__ ");
				System.out.print(h_i_weights_mapping[i][j]);
				if(j + 1 < h_i_weights_mapping[i].length)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		System.out.print("\n\n");
		
		System.out.println("HO_WM: ");
		for(int i = 0, max = h_o_weights_mapping.length; i < max ; i++ )
		{
			for(int j = 0; j < h_o_weights_mapping[i].length ; j++ )
			{
				System.out.print("H[" + i + "] -> O[" + j + "]:__ ");
				System.out.print(h_o_weights_mapping[i][j]);
				if(j + 1 < h_o_weights_mapping[i].length)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		System.out.print("#############\n");
	}
}
