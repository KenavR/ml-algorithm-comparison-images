package at.technikum.ml.algorithms;

import at.technikum.ml.utils.FileHelper;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.File;

public abstract class MLAlgorithmRunner<T, C> {
    protected Classifier classifier;
    private Instances data;

    public T buildClassifier(File datafile) throws Exception {
        BufferedReader contents = FileHelper.readDataFile(datafile);

        data = new Instances(contents);
        data.setClassIndex(data.numAttributes() - 1);
        classifier.buildClassifier(data);

        return (T) this;
    }

    public double classify(Instance instance) throws Exception {
        return classifier.classifyInstance(instance);
    }

    public Instances getData() {
        return data;
    }

    public Classifier getClassifier() {
        return classifier;
    }

    public abstract String getName();
}
