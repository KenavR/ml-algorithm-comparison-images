package at.technikum.ml.algorithms;

import weka.classifiers.functions.MultilayerPerceptron;

public class PerceptronRunner extends MLAlgorithmRunner<PerceptronRunner, Double> {

    private PerceptronRunner() {
        this.classifier = new MultilayerPerceptron();
        ((MultilayerPerceptron) this.classifier).setHiddenLayers("0");
    }

    public static PerceptronRunner newPerceptronRunner() {
        return new PerceptronRunner();
    }

    @Override
    public String getName() {
        return "Perceptron";
    }
}
