package dataset;

public abstract class DataGen<Gen_type>
{
	Gen_type[][] dataset;	//inputs and sets
	public DataGen(int inputs, int sets)
	{
		dataset = Generate(inputs,sets);
	}
	
	protected abstract Gen_type[][] Generate(int inputs, int sets);
	
	//TESTING METHODS
	protected void PrintDataSet()
	{
		System.out.println("###_DATASET_### iteration -> INPUT\n");
		System.out.println("IO_WM: ");
		for(int i = 0, max = dataset[0].length; i < max ; i++ )
		{
			for(int j = 0, max2 = dataset.length; j < max2 ; j++ )
			{
				System.out.print("i[" + i + "] -> I[" + j + "]:__ ");
				System.out.print(dataset[j][i]);
				if(j + 1 < dataset.length)
					System.out.print("\n");
				else
					System.out.print("\n\n");
			}
		}
		System.out.print("\n\n");
	}
}
