package at.technikum.ml.evaluation;

import at.technikum.ml.algorithms.MLAlgorithmRunner;
import at.technikum.ml.config.Configuration;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import static at.technikum.ml.algorithms.DecisionTreeRunner.newDecisionTreeRunner;
import static at.technikum.ml.algorithms.NaiveBayesRunner.newNaiveBayesRunner;
import static at.technikum.ml.algorithms.NearestNeighborRunner.newNearestNeighborRunner;
import static at.technikum.ml.algorithms.PerceptronRunner.newPerceptronRunner;

public class EvaluationRunner {

    public static Metrics runEvaluation(MLAlgorithmRunner runner, SplitStrategy strategy) throws Exception {
        Random rdm = new Random(Configuration.RANDOM_SEED);
        Instances rdmData = new Instances(runner.getData());
        rdmData.randomize(rdm);
        Evaluation eval;

        if(strategy.equals(SplitStrategy.FOLDS)) {
            eval = evalFolds(rdmData, runner);
        } else if(strategy.equals(SplitStrategy.TWOtoONE)){
            eval = eval2to1(rdmData, runner);
        } else {
            throw new IllegalArgumentException(String.format("Provided SplitStrategy '%s' is not supported.", strategy));
        }

        return new Metrics(runner.getName(), 1-eval.errorRate(), eval.weightedPrecision(), eval.weightedRecall(), 0, 0);
    }

    private static Evaluation evalFolds(Instances rdmData, MLAlgorithmRunner runner) throws Exception {
        if (rdmData.classAttribute().isNominal())
            rdmData.stratify(Configuration.FOLDS);

        Evaluation eval = new Evaluation(rdmData);
        for (int n = 0; n < Configuration.FOLDS; n++) {
            Instances train = rdmData.trainCV(Configuration.FOLDS, n);
            Instances test = rdmData.testCV(Configuration.FOLDS, n);

            Classifier clsCopy = AbstractClassifier.makeCopy(runner.getClassifier());
            clsCopy.buildClassifier(train);
            eval.evaluateModel(clsCopy, test);
        }
        return eval;
    }

    private static Evaluation eval2to1(Instances rdmData, MLAlgorithmRunner runner) throws Exception {
        Evaluation eval = new Evaluation(rdmData);
        int trainIdx = Configuration.getLastIndexOfTrainSet(rdmData.size());
        Instances train = new Instances(rdmData, 0, trainIdx);
        Instances test  = new Instances(rdmData, trainIdx+1, rdmData.size()-(trainIdx+1));

        Classifier clsCopy = AbstractClassifier.makeCopy(runner.getClassifier());
        clsCopy.buildClassifier(train);
        eval.evaluateModel(clsCopy, test);
        return eval;
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
