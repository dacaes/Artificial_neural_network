package dataset;

import java.util.Random;

public final class DataGenBinary extends DataGen<Byte>
{
	int inputs,sets;
	
	public DataGenBinary(int inputs, int sets) {
		super(inputs, sets);
		// TODO Auto-generated constructor stub
	}

	@Override
	protected Byte[][] Generate(int inputs, int sets)
	{
		// TODO Auto-generated method stub
		Byte[][] dataset = new Byte[inputs][sets];
		this.inputs = inputs;
		this.sets = sets;
		
		int r_max = 1;
		int r_min = 0;
		
		dataset = new Byte[inputs][sets];
		Random rand = new Random();
		
		for (int i = 0; i < inputs; i++)
		{
			for (int j = 0; j <sets; j++)
			{
				dataset[i][j] = (byte) (rand.nextInt((r_max - r_min) + 1) + r_min);
			}
		}
		
		return dataset;
	}

	@Override
	public void PrintDataSet() {
		// TODO Auto-generated method stub
		super.PrintDataSet();
	}
	
	public Byte[][] CustomDataSet()
	{
		Byte[][] dataset = new Byte[inputs][sets];
		dataset[0][0] = (byte) 0;
		dataset[1][0] = (byte) 0;
		
		dataset[0][1] = (byte) 0;
		dataset[1][1] = (byte) 1;
		
		dataset[0][2] = (byte) 1;
		dataset[1][2] = (byte) 0;
		
		dataset[0][3] = (byte) 1;
		dataset[1][3] = (byte) 1;
		
		return dataset;
	}
}
