package dataset;

import java.util.Random;

public final class DataGenBinary extends DataGen<Byte> {


	public DataGenBinary(int inputs, int sets) {
		super(inputs, sets);
		// TODO Auto-generated constructor stub
	}

	@Override
	protected void Generate(int inputs, int sets) {
		// TODO Auto-generated method stub
		
		dataset = new Byte[inputs][sets];
		Random rand = new Random();
		//int randomNum = rand.nextInt((max - min) + 1) + min;
		
		for (int i = 0; i < inputs; i++)
		{
			for (int j = 0; j <sets; j++)
			{
				dataset[i][j] = (byte) (rand.nextInt((1 - 0) + 1) + 0);
			}
		}
	}

	@Override
	public void PrintDataSet() {
		// TODO Auto-generated method stub
		super.PrintDataSet();
	}
}
