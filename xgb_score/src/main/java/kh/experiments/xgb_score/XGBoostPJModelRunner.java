package kh.experiments.xgb_score;

import biz.k11i.xgboost.Predictor;
import biz.k11i.xgboost.util.FVec;
import org.apache.commons.lang3.time.StopWatch;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * test a pure java implementation
 * https://github.com/komiya-atsushi/xgboost-predictor-java/tree/master/xgboost-predictor
 */
class XGBoostPJModelRunner implements ExperimentRunner
{
    private final StopWatch m_timer;
    private final Predictor m_model;
    private final FVec[] m_predictionData;

    private XGBoostPJModelRunner(Predictor model, FVec[] predictionData)
    {
        m_timer = new StopWatch();
        m_model = model;
        m_predictionData = predictionData;
    }

    static XGBoostPJModelRunner create(String modelPath, String dataPath)
    {
        try
        {
            FVec[] data = DataReaderUtils.getFVecData(dataPath);
            return create(modelPath, data);
        }
        catch (IOException ex)
        {
            throw new RuntimeException("Failed to load data", ex);
        }
    }

    static XGBoostPJModelRunner create(String modelPath, FVec[] predictionData)
    {
        try
        {
            Predictor predictor = new Predictor(
                    new java.io.FileInputStream(modelPath));

            return new XGBoostPJModelRunner(predictor, predictionData);
        }
        catch (IOException ex)
        {
            throw new RuntimeException("Failed to load model and/or data", ex);
        }
    }

    /**
     * returns the time (in microseconds) that model scoring took
     */
    public List<Long> run(int replicates)
    {
        m_timer.reset();

        List<Long> runtimes = new ArrayList<>(replicates);

        for (int replicateIx = 0; replicateIx < replicates; replicateIx++)
        {
            m_timer.start();
            for (int observationIx = 0; observationIx < m_predictionData.length; observationIx++)
            {
                double[] predictions = m_model.predict(m_predictionData[observationIx]);
                System.out.println(predictions[0]);
            }
            m_timer.stop();
            runtimes.add(replicateIx, m_timer.getTime(TimeUnit.MICROSECONDS));
            m_timer.reset(); // note if an exception occurs, we may leave the timer in a bad state
        }

        return runtimes;
    }

    long getDataSize()
    {
        return m_predictionData.length;
    }

    public static void main(String[] args) throws IOException
    {
        if (args.length < 2)
        {
            throw new RuntimeException("Need to specify 1. a serialized xgboost model 2. a data location (libsvm format)");
        }

        int replicates = args.length > 2 ? Integer.parseInt(args[2]) : 1;

        XGBoostPJModelRunner runner = XGBoostPJModelRunner.create(args[0], args[1]);
        long runtime = runner.run(replicates).stream()
                .mapToLong(Long::longValue)
                .sum();

        System.out.println("Ran (xgb jvm) model evaluation on " + replicates * runner.getDataSize() +
                " observations in " + runtime + " micros");
    }
}