package at.technikum.ml.evaluation;

import at.technikum.ml.algorithms.MLAlgorithmRunner;
import at.technikum.ml.config.Configuration;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.experiment.CSVResultListener;
import weka.experiment.ClassifierSplitEvaluator;
import weka.experiment.RandomSplitResultProducer;
import weka.experiment.SplitEvaluator;
import weka.gui.experiment.Experimenter;

import java.io.File;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import static at.technikum.ml.algorithms.DecisionTreeRunner.newDecisionTreeRunner;
import static at.technikum.ml.algorithms.NaiveBayesRunner.newNaiveBayesRunner;
import static at.technikum.ml.algorithms.NearestNeighborRunner.newNearestNeighborRunner;
import static at.technikum.ml.algorithms.PerceptronRunner.newPerceptronRunner;

public class EvaluationRunner {

    private static CSVResultListener resultListener = new CSVResultListener();

    public static Metrics runEvaluation(MLAlgorithmRunner runner, SplitStrategy strategy) throws Exception {

        Random rdm = new Random(Configuration.RANDOM_SEED);
        Instances rdmData = new Instances(runner.getData());
        rdmData.randomize(rdm);

        if(strategy.equals(SplitStrategy.FOLDS)) {
            return evalFolds(rdmData, runner, Configuration.FOLDS);
        } else if(strategy.equals(SplitStrategy.TWOtoONE)){
            return eval2to1(rdmData, runner);
        } else {
            throw new IllegalArgumentException(String.format("Provided SplitStrategy '%s' is not supported.", strategy));
        }
    }

    private static Metrics evalFolds(Instances data, MLAlgorithmRunner runner, int folds) throws Exception {
        long trainTimeStart;
        long trainTimeElapsed;
        long testTimeStart;
        long testTimeElapsed;

        Classifier clsCopy = AbstractClassifier.makeCopy(runner.getClassifier());

        /* train/build classifier */
        trainTimeStart = System.currentTimeMillis();
        clsCopy.buildClassifier(data);
        trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;

        Evaluation eval = new Evaluation(data);
        Random rdm = new Random(Configuration.RANDOM_SEED);

        /* test/evaluate classifier */
        testTimeStart = System.currentTimeMillis();
        eval.crossValidateModel(runner.getClassifier(), data, folds, rdm);
        testTimeElapsed = System.currentTimeMillis() - testTimeStart;

        return new Metrics(runner.getName(), 1-eval.errorRate(), eval.weightedPrecision(), eval.weightedRecall(), trainTimeElapsed, testTimeElapsed);
    }

    private static Metrics eval2to1(Instances rdmData, MLAlgorithmRunner runner) throws Exception {
        long trainTimeStart;
        long trainTimeElapsed;
        long testTimeStart;
        long testTimeElapsed;

        Evaluation eval = new Evaluation(rdmData);

        int trainIdx = Configuration.getLastIndexOfTrainSet(rdmData.size());
        Instances train = new Instances(rdmData, 0, trainIdx);
        Instances test  = new Instances(rdmData, trainIdx+1, rdmData.size()-(trainIdx+1));

        Classifier clsCopy = AbstractClassifier.makeCopy(runner.getClassifier());

        /* train/build classifier */
        trainTimeStart = System.currentTimeMillis();
        clsCopy.buildClassifier(train);
        trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;

        /* test/evaluate classifier */
        testTimeStart = System.currentTimeMillis();
        eval.evaluateModel(clsCopy, test);
        testTimeElapsed = System.currentTimeMillis() - testTimeStart;

        return new Metrics(runner.getName(), 1-eval.errorRate(), eval.weightedPrecision(), eval.weightedRecall(),   trainTimeElapsed, testTimeElapsed);
    }


    public static Metrics runEvaluation(MLAlgorithmRunner runner, SplitStrategy strategy, File data) throws Exception {
        return runEvaluation((MLAlgorithmRunner) runner.buildClassifier(data), strategy);
    }

    public static List<Metrics> runAllEvaluation(SplitStrategy strategy, File data) throws Exception {
        List<MLAlgorithmRunner> runners = getAllRunners(data);
        return runners.stream().map(r -> {
            try {
                return runEvaluation(r, strategy);
            } catch (Exception e) {
                e.printStackTrace();
            }
            return null;
        }).collect(Collectors.toList());
    }

    private static List<MLAlgorithmRunner> getAllRunners(File file) {
        List<MLAlgorithmRunner> runners;
        try {
            runners = Arrays.asList(
                    newNearestNeighborRunner(3).buildClassifier(file),
                    newNearestNeighborRunner(5).buildClassifier(file),
                    newNearestNeighborRunner(7).buildClassifier(file),
                    newNaiveBayesRunner().buildClassifier(file),
                    newPerceptronRunner().buildClassifier(file),
                    newDecisionTreeRunner().buildClassifier(file)
            );
        } catch (Exception e) {
            throw new IllegalStateException(String.format("Error creating Runner: %s", e.getMessage()));
        }
        return runners;
    }
}
