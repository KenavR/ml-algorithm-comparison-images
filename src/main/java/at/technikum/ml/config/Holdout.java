package at.technikum.ml.config;

public class Holdout {
    public final int train;
    public final int test;

    public Holdout(int train, int test) {
        this.train = train;
        this.test = test;
    }

    public int getCombined() {
        return this.train + this.test;
    }
}
