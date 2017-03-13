package at.technikum.ml.algorithms;

import weka.classifiers.lazy.IBk;

public class NearestNeighborRunner extends MLAlgorithmRunner<NearestNeighborRunner, Double> {
    private int k;

    private NearestNeighborRunner(int k) {
        this.k = k;
        this.classifier = new IBk(k);
    }

    public static NearestNeighborRunner newNearestNeighborRunner(int k) {
        return new NearestNeighborRunner(k);
    }

    @Override
    public String getName() {
        return String.format("k-NN (%d-NN)", k);
    }
}
