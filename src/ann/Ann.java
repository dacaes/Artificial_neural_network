package ann;

public class Ann
{
	//MAPPING
	public Byte[][] 	mapping_I_O;		//input -> output mapping
	public Byte[][] 	mapping_H_I;		//input -> length_H mapping
	public Byte[][]		mapping_H_O;		//length_H -> output mapping
	
	//WEIGHTS
	public double[][] 	weights_I_O;		//input -> output weight
	public double[][] 	weights_H_I;		//input -> length_H weight
	public double[][] 	weights_H_O;		//length_H -> output weight
	
	public double[] 	weights_H_BIAS;		//length_H bias weight
	public double[] 	weights_O_BIAS;		//output bias weight
	
	//VALUES
	public double[] 	neurons_I;			//input values
	public double[] 	neurons_H;			//length_H values
	public double[] 	neurons_O;			//output values
	
	public Ann()
	{
		
	}
}
