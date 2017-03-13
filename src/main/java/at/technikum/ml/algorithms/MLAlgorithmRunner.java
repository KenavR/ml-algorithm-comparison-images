package at.technikum.ml.algorithms;

import at.technikum.ml.config.Configuration;
import at.technikum.ml.utils.FileHelper;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.File;

public abstract class MLAlgorithmRunner<T, C> {
    protected Classifier classifier;
    private Instances data;
    //private Instances trainSet;
    //private Instances testSet;

    public T buildClassifier(File datafile) throws Exception {
        BufferedReader contents = FileHelper.readDataFile(datafile);

        data = new Instances(contents);

        //int trainIdx = Configuration.getLastIndexOfTrainSet(data.size());
        //this.trainSet = new Instances(data, 0, trainIdx);
        //this.testSet = new Instances(data, trainIdx+1, data.size()-(trainIdx+1));

        //this.trainSet.setClassIndex(this.trainSet.numAttributes() - 1);
        //this.testSet.setClassIndex(this.testSet.numAttributes() -1);
        data.setClassIndex(data.numAttributes() - 1);
        classifier.buildClassifier(data);
        return (T) this;
    }

    public double classify(Instance instance) throws Exception {
        return classifier.classifyInstance(instance);
    }

    /*public Instances getTestSet() {
        return this.testSet;
    }*/

    public Instances getData() {
        return data;
    }

    public Classifier getClassifier() {
        return classifier;
    }

    public abstract String getName();
}
