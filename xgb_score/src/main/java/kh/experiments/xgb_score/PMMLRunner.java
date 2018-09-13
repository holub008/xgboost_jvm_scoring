package kh.experiments.xgb_score;

import org.apache.commons.lang3.time.StopWatch;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.ModelEvaluator;
import org.jpmml.evaluator.ModelEvaluatorFactory;
import org.xml.sax.SAXException;

import javax.xml.bind.JAXBException;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * use jpmml (a generic ml evaluation library) to score the model
 */
class PMMLRunner implements ExperimentRunner
{
    private final StopWatch m_timer;
    private final ModelEvaluator m_model;
    private final Map<FieldName, FieldValue>[] m_predictionData;

    private PMMLRunner(ModelEvaluator model, Map<FieldName, FieldValue>[] predictionData)
    {
        m_timer = new StopWatch();
        m_model = model;
        m_predictionData = predictionData;

        // docs claim that first evaluation loads/caches a bunch of model stuff, so we do that outside of timing
        m_model.evaluate(m_predictionData[0]);
    }

    static PMMLRunner create(String modelPath, String dataPath, String featureMapPath)
    {
        try
        {
            PMML pmml;

            try (InputStream resource = new FileInputStream(modelPath))
            {
                pmml = org.jpmml.model.PMMLUtil.unmarshal(resource);
            }

            ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();
            ModelEvaluator evaluator = modelEvaluatorFactory.newModelEvaluator(pmml);

            Map<FieldName, FieldValue>[] data = dataPath.endsWith(".svm") ?
                    DataReaderUtils.hackReadLibSVMToPMML(dataPath, featureMapPath) :
                    DataReaderUtils.readJSONFeatures(dataPath, evaluator);

            return new PMMLRunner(evaluator, data);
        }
        catch (IOException | JAXBException | SAXException ex)
        {
            throw new RuntimeException("Failed to load data", ex);
        }
    }

    /**
     * use this creation point to avoid reloading data for different models
     * note that this depends on the predictionData & the model preprocessing to be independent (or consistent across different client models)
     */
    static PMMLRunner create(String modelPath, Map<FieldName, FieldValue>[] predictionData)
    {
        try
        {
            PMML pmml;

            try (InputStream resource = new FileInputStream(modelPath))
            {
                pmml = org.jpmml.model.PMMLUtil.unmarshal(resource);
            }

            ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();
            ModelEvaluator evaluator = modelEvaluatorFactory.newModelEvaluator(pmml);

            return new PMMLRunner(evaluator, predictionData);
        }
        catch (IOException | SAXException | JAXBException ex)
        {
            throw new RuntimeException("Failed to load model", ex);
        }
    }

    long getDataSize()
    {
        return m_predictionData.length;
    }

    /**
     * returns the time (in microseconds) that the model scoring took
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
                m_model.evaluate(m_predictionData[observationIx]);
            }
            m_timer.stop();
            runtimes.add(replicateIx, m_timer.getTime(TimeUnit.MICROSECONDS));
            m_timer.reset(); // note if an exception occurs, we may leave the timer in a bad state
        }

        return runtimes;
    }

    public static void main(String[] args)
    {
        if (args.length < 3)
        {
            throw new RuntimeException("Need to specify 1. a serialized pmml model 2. a data location (libsvm format) 3. a feature map (optional for json featureset)");
        }

        // replicates are the number of times we repeat the input data
        int replicates = args.length > 3 ? Integer.parseInt(args[3]) : 1;

        PMMLRunner runner = PMMLRunner.create(args[0], args[1], args[2]);
        long runtime = runner.run(replicates).stream()
                .mapToLong(Long::longValue)
                .sum();

        System.out.println("Ran (jpmml) model evaluation on " + replicates * runner.getDataSize() +
                " observations in " + runtime + " micros");
    }
}
