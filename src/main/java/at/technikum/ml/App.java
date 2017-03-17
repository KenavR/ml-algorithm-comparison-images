package at.technikum.ml;

import at.technikum.ml.evaluation.EvaluationRunner;
import at.technikum.ml.evaluation.Metrics;
import at.technikum.ml.evaluation.SplitStrategy;

import java.io.File;
import java.util.List;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogManager;
import java.util.logging.Logger;


public class App {

    public static void main(String[] args) {
        configureLogger();
        List<Metrics> metrics;
        try {
            System.out.println(String.format(" ============================ Evaluation for IRIS Dataset ============================ "));
            metrics = EvaluationRunner.runAllEvaluation(SplitStrategy.FOLDS, new File(App.class.getResource("/iris.arff").getFile()));
            printTable("IRIS/5-folds", metrics);

            System.out.println(String.format(" ------------------------------------------------------------------------------------- "));
            metrics = EvaluationRunner.runAllEvaluation(SplitStrategy.TWOtoONE, new File(App.class.getResource("/iris.arff").getFile()));
            printTable("IRIS/2-to-1", metrics);

            System.out.println();
            System.out.println();
            System.out.println();

            System.out.println(String.format(" ===================== Evaluation for Handwritten Digits Dataset ===================== "));
            metrics = EvaluationRunner.runAllEvaluation(SplitStrategy.FOLDS, new File(App.class.getResource("/digits.arff").getFile()));
            printTable("Digits/5-folds", metrics);

            System.out.println(String.format(" ------------------------------------------------------------------------------------- "));
            metrics = EvaluationRunner.runAllEvaluation(SplitStrategy.TWOtoONE, new File(App.class.getResource("/digits.arff").getFile()));
            printTable("Digits/2-to-1", metrics);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void printTable(String name, List<Metrics> rows) {
        /* headline */
        System.out.println(String.format("%20s %10s %10s %10s %15s %15s", name, "Accuracy", "Precision", "Recall", "Training time", "Testing time"));
        /* data */
        rows.forEach(m -> System.out.println(String.format("%20s %10s %10s %10s %10s ms %10s ms", m.algorithm, fToS(m.accuracy, 3), fToS(m.precision, 3), fToS(m.recall, 3), fToS(m.trainingTime, 0), fToS(m.testingTime, 0))));
    }

    private static String fToS(double n, int decimal) {
        return String.format("%." + decimal + "f", n);
    }

    private static void configureLogger() {
        Logger log = LogManager.getLogManager().getLogger("");
        for (Handler h : log.getHandlers()) {
            h.setLevel(Level.SEVERE);
        }
    }
}
