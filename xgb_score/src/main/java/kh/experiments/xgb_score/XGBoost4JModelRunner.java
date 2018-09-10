package kh.experiments.xgb_score;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

import org.apache.commons.lang3.time.StopWatch;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * test xgboost4j implementation - this uses JNI bindings, which could be slow, platform dependent
 */
class XGBoost4JModelRunner implements ExperimentRunner
{
    private final StopWatch m_timer;
    private final Booster m_model;
    private final DMatrix m_predictionData;

    private XGBoost4JModelRunner(Booster model, DMatrix predictionData)
    {
        m_timer = new StopWatch();
        m_model = model;
        m_predictionData = predictionData;
    }

    static XGBoost4JModelRunner create(String modelPath, String dataPath)
    {
        try
        {
            DMatrix data = DataReaderUtils.getDMatrix(dataPath);

            return create(modelPath, data);
        }
        catch (XGBoostError ex)
        {
            throw new RuntimeException("Could not load data", ex);
        }
    }

    static XGBoost4JModelRunner create(String modelPath, DMatrix predictionData)
    {
        try
        {
            Booster model = XGBoost.loadModel(modelPath);

            return new XGBoost4JModelRunner(model, predictionData);
        }
        catch (XGBoostError ex)
        {
            throw new RuntimeException("Could not load model", ex);
        }
    }

     long getDataSize() throws XGBoostError
     {
         return m_predictionData.rowNum();
     }

    /**
     * returns the time (in microseconds) that model scoring took
     */
    public List<Long> run(int replicates)
    {
        m_timer.reset();

        List<Long> runtimes = new ArrayList<>(replicates);

        try
        {
            for (int replicateIx = 0; replicateIx < replicates; replicateIx++)
            {
                m_timer.start();

                m_model.predict(m_predictionData);

                m_timer.stop();
                runtimes.add(replicateIx, m_timer.getTime(TimeUnit.MICROSECONDS));
                m_timer.reset(); // note if an exception occurs, we may leave the timer in a bad state
            }
        }
        catch (XGBoostError ex)
        {
            throw new RuntimeException("XGBoost barfed", ex);
        }
        finally
        {
            m_timer.reset();
        }


        return runtimes;
    }

    static void main(String[] args) throws XGBoostError
    {
        if (args.length < 2)
        {
            throw new RuntimeException("Need to specify 1. a serialized xgboost model 2. a data location (libsvm format)");
        }

        int replicates = args.length > 2 ? Integer.parseInt(args[2]) : 1;
        XGBoost4JModelRunner runner = XGBoost4JModelRunner.create(args[0], args[1]);
        long runtime = runner.run(replicates).stream()
                .mapToLong(Long::longValue)
                .sum();

        System.out.println("Ran (xgb native) model evaluation on " + replicates * runner.getDataSize() +
                " observations in " + runtime + " micros");
    }
}
