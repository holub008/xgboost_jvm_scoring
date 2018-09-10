package kh.experiments.xgb_score;

import biz.k11i.xgboost.util.FVec;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.dmg.pmml.FieldName;
import org.jpmml.evaluator.FieldValue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Score a variety of models across a variety of datasets
 */
public class ExperimentSuite
{
    private final Map<String, ExperimentRunner> m_experimentsToRun;

    public ExperimentSuite(Map<String, ExperimentRunner> experimentsToRun)
    {
        m_experimentsToRun = experimentsToRun;
    }

    public Map<String, List<Long>> run(int replicatesPerExperiment)
    {
        Map<String, List<Long>> results = new HashMap<>();
        for (Map.Entry<String, ExperimentRunner> experimentEntry : m_experimentsToRun.entrySet())
        {
            String experimentName = experimentEntry.getKey();
            ExperimentRunner experiment = experimentEntry.getValue();

            results.put(experimentName, experiment.run(replicatesPerExperiment));
        }

        return results;
    }

    public static void main(String[] args) throws IOException, XGBoostError
    {
        Path xgbModelDirectory = Paths.get(args[0]);
        Path pmmlModelDirectory = Paths.get(args[1]);
        Path dataDirectory = Paths.get(args[2]);
        String featureMapPath = args[3];
        Path resultsDestination = Paths.get(args[4]);
        int replicates = Integer.parseInt(args[5]);

        Map<String, ExperimentRunner> experiments = new HashMap<>();

        List<String> dataPaths = Files.walk(dataDirectory)
                .filter(Files::isRegularFile)
                .map(Object::toString)
                .collect(Collectors.toList());

        // technically this should be in a try with resources
        List<String> xgbModelPaths = Files.walk(xgbModelDirectory)
                .filter(Files::isRegularFile)
                .map(Object::toString)
                .collect(Collectors.toList());

        List<String> pmmlModelPaths = Files.walk(pmmlModelDirectory)
                .filter(Files::isRegularFile)
                .map(Object::toString)
                .collect(Collectors.toList());

        for (String dataPath : dataPaths)
        {
            String dataName = Paths.get(dataPath).getFileName().toString();

            DMatrix xgb4jData = DataReaderUtils.getDMatrix(dataPath);
            FVec[] xgbpjData = DataReaderUtils.getFVecData(dataPath);
            Map<FieldName, FieldValue>[] pmmlData = DataReaderUtils.hackReadLibSVMToPMML(dataPath, featureMapPath);

            for (String xgbPath : xgbModelPaths)
            {
                String xgbName = Paths.get(xgbPath).getFileName().toString();

                experiments.put(dataName + ":" + xgbName + ":XGB4J", XGBoost4JModelRunner.create(xgbPath, xgb4jData));
                experiments.put(dataName + ":" + xgbName + ":XGBPJ", XGBoostPJModelRunner.create(xgbPath, xgbpjData));
            }

            for (String pmmlModelPath : pmmlModelPaths)
            {
                String pmmlName = Paths.get(pmmlModelPath).getFileName().toString();

                experiments.put(dataName + ":" + pmmlName + ":PMML", PMMLRunner.create(pmmlModelPath, pmmlData));
            }
        }

        ExperimentSuite suite = new ExperimentSuite(experiments);

        suite.run(replicates);

        Map<String, List<Long>> results = suite.run(replicates);

        String csvResults = results.entrySet().stream()
                .map(entry -> entry.getKey() + "," + entry.getValue().stream().map(Object::toString).collect(Collectors.joining(",")))
                .collect(Collectors.joining("\n"));

        System.out.println(csvResults);
        Files.write(resultsDestination, csvResults.getBytes());
    }
}
