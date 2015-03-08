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
	ArrayList<Byte> i_o_weights_mapping = new ArrayList<Byte>();						//input -> output mapping
	ArrayList<ArrayList<Byte>> i_h_weights_mapping = new ArrayList<ArrayList<Byte>>();	//input -> hidden mapping
	ArrayList<ArrayList<Byte>> h_o_weights_mapping = new ArrayList<ArrayList<Byte>>();	//hidden -> output mapping
			
	public Ann(ArrayList<Byte> genotype, int inputs, int outputs)
	{
		int gen_size = genotype.size();
		//length of the blocks of the genotype
		int blocks_length = inputs*outputs;
		int hidden = gen_size / blocks_length - 1; 
		
		//arrays for storing values of the neurons
		int[] i_neurons = new int[inputs];					//input
		int[] h_neurons = new int[hidden];					//hidden
		int[] o_neurons = new int[outputs];					//output
		 
		//arrays for storing values of the weights
		int[] i_o_weights = new int[blocks_length];			//input -> output weight
		int[] i_h_weights = new int[inputs*hidden];			//input -> hidden weight
		int[] h_o_weights = new int[hidden*outputs];		//hidden -> output weight
		
		WeightMapping(genotype, gen_size, blocks_length, inputs, outputs);
	}
	
	private void WeightMapping(ArrayList<Byte> genotype, int gen_size, int blocks_length, int inputs, int outputs)
	{
		for (int i = 0; i < gen_size; i++)
		{
			byte val = genotype.get(i);
			
			//input ->  output connections mapping
			if(i < blocks_length)
				i_o_weights_mapping.add(val);
			
			//hidden connections mapping
			else 
			{
				int hidden_neuron = -1;
				int substraction = i;
				do
				{
					hidden_neuron++;
					substraction -= blocks_length;
				}while(substraction > blocks_length);
				
				int input_index = substraction / inputs;
				
				int output_index = substraction % outputs;
				
				//input ->  hidden connections mapping
				i_h_weights_mapping.get(hidden_neuron).set(input_index, val);
				//hidden ->  output connections mapping
				h_o_weights_mapping.get(hidden_neuron).set(output_index, val);
			}	
		}		
	}
}
