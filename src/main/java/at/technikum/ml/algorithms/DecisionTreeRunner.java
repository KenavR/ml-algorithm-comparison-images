package at.technikum.ml.algorithms;

import weka.classifiers.trees.J48;

public class DecisionTreeRunner extends MLAlgorithmRunner<DecisionTreeRunner, Double> {
    private DecisionTreeRunner() {
        this.classifier = new J48();
    }

    public static DecisionTreeRunner newDecisionTreeRunner() {
        return new DecisionTreeRunner();
    }

    @Override
    public String getName() {
        return "Decision Tree";
    }
}
