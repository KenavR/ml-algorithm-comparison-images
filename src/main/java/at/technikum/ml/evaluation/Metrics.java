package at.technikum.ml.evaluation;

public class Metrics {
    public final String algorithm;
    public final double accuracy;
    public final double precision;
    public final double recall;
    public final double trainingTime;
    public final double testingTime;

    public Metrics(String algorithm, double accuracy, double precision, double recall, double trainingTime, double testingTime) {
        this.algorithm = algorithm;
        this.accuracy = accuracy;
        this.precision = precision;
        this.recall = recall;
        this.trainingTime = trainingTime;
        this.testingTime = testingTime;
    }
}
