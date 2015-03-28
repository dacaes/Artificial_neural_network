package ann;

public interface Const {
	public enum Activation {
	    UMBRAL,
	    SIGMOID,
	    TANH
	}
	
	public enum EvalType {
	    FITNESS,
	    EARLY_STOP
	}
	
	public static final boolean		DEBUG 			= true;
	public static final Activation	AFUNC 			= Activation.TANH;
	public static final int			TRAININGS		= 3000;
	public static final int 		INPUTS 			= 2;
	public static final int 		OUPUTS 			= 1;
	public static final double		LEARN_FACTOR 	= 0.1;
	public static final double		FITNESS 		= 0.01;
	public static final int 		RANGE 			= 100;
	public static final EvalType	ETYPE 			= EvalType.EARLY_STOP;
}
