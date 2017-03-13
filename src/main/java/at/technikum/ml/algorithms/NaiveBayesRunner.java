package at.technikum.ml.algorithms;

import weka.classifiers.bayes.NaiveBayes;

public class NaiveBayesRunner extends MLAlgorithmRunner<NaiveBayesRunner, Double> {
    public static NaiveBayesRunner newNaiveBayesRunner() {
        return new NaiveBayesRunner();
    }

    private NaiveBayesRunner() {
        this.classifier = new NaiveBayes();
    }

    @Override
    public String getName() {
        return "Naive Bayes";
    }
}
