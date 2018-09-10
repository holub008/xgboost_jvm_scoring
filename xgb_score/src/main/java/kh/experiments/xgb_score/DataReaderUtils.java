package kh.experiments.xgb_score;

import biz.k11i.xgboost.util.FVec;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.FieldValueUtil;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * shared utils for reading data
 */
final class DataReaderUtils
{
    private DataReaderUtils() {}

    private static double[] hackParseSVMLine(String[] data)
    {
        double[] predictors = new double[data.length - 1];

        // entry 0 is the response
        for (int entry_ix = 1; entry_ix < data.length; entry_ix++)
        {
            String doublePart = data[entry_ix].split(":")[1];
            predictors[entry_ix - 1] = Double.parseDouble(doublePart);
        }

        return predictors;
    }
    /**
     * does not correctly read general svm format. this is just used for testing purposes
     */
    private static double[][] hackReadLibSVM(String filePath) throws IOException
    {
        return Files.lines(Paths.get(filePath))
                .map(line -> line.split("\\s+"))
                .map(DataReaderUtils::hackParseSVMLine)
                .toArray(double[][]::new);
    }

    /**
     * if it wasn't abundantly clear, this is hacky - ideally we let the model pre-process the feature set by having the InputFields of the model .prepare() the raw features
     * but that's irrelevant here, because the persisted data was curated
     */
    private static Map<FieldName, FieldValue> buildFeatureContext(double[] features, String[] featureMap)
    {
        Map<FieldName, FieldValue> featureContext = new HashMap<>();

        for (int feature_ix = 0; feature_ix < features.length; feature_ix++)
        {
            FieldName featureName = new FieldName(featureMap[feature_ix]);
            // wtf, kind of frustrating design choice here
            FieldValue feature = FieldValueUtil.create(DataType.FLOAT, OpType.CONTINUOUS, Double.toString(features[feature_ix]));
            featureContext.put(featureName, feature);
        }

        return featureContext;
    }

    static FVec[] getFVecData(String dataPath) throws IOException
    {
        return Arrays.stream(hackReadLibSVM(dataPath))
                .map(arr -> FVec.Transformer.fromArray(arr, false))
                .toArray(FVec[]::new);
    }

    static DMatrix getDMatrix(String dataPath) throws XGBoostError
    {
        return new DMatrix(dataPath);
    }

    /**
     * does not correctly read svm format. this is just used for testing purposes
     */
    static Map<FieldName, FieldValue>[] hackReadLibSVMToPMML(String filePath, String featureMapPath) throws IOException
    {
        String[] featureMap = Files.lines(Paths.get(featureMapPath))
                .map(line -> line.split("\\s+")[1])
                .toArray(String[]::new);

        return Files.lines(Paths.get(filePath))
                .map(line -> line.split("\\s+"))
                .map(DataReaderUtils::hackParseSVMLine)
                .map(features -> buildFeatureContext(features, featureMap))
                .toArray(Map[]::new);
    }
}
