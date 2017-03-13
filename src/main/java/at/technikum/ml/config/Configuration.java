package at.technikum.ml.config;

public abstract class Configuration {

    public static final Holdout HOLDOUT = new Holdout(2, 3);
    public static final int RANDOM_SEED = 33; //Student ID
    public static final int FOLDS = 5;

    public static int getLastIndexOfTrainSet(int size) {
        return ((size / HOLDOUT.getCombined()) * HOLDOUT.train);
    }
}
