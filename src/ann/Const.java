package ann;

public interface Const {
	public enum Activation {
	    UMBRAL,
	    SIGMOID,
	    TANH
	}
	
	public static final boolean		DEBUG 			= false;
	public static final int			TRAININGS		= 10000;
	public static final int 		INPUTS 			= 2;
	public static final int 		OUPUTS 			= 1;
	public static final double		LEARN_FACTOR 	= 0.1;
	public static final double		FITNESS 		= 0.01;
	public static final Activation	AFUNC 			= Activation.SIGMOID;
}
