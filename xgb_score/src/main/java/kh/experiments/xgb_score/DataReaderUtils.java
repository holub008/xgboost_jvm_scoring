package kh.experiments.xgb_score;

import biz.k11i.xgboost.util.FVec;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.FieldValueUtil;
import org.jpmml.evaluator.InputField;
import org.jpmml.evaluator.ModelEvaluator;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

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
        for (int entryIx = 1; entryIx < data.length; entryIx++)
        {
            String doublePart = data[entryIx].split(":")[1];
            predictors[entryIx - 1] = Double.parseDouble(doublePart);
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

        for (int featureIx = 0; featureIx < features.length; featureIx++)
        {
            FieldName featureName = new FieldName(featureMap[featureIx]);
            // wtf, kind of frustrating design choice here
            FieldValue feature = FieldValueUtil.create(DataType.FLOAT, OpType.CONTINUOUS, Double.toString(features[featureIx]));
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

    static Map<FieldName, FieldValue>[] readJSONFeatures(String filePath, ModelEvaluator model) throws IOException
    {
        ObjectMapper mapper = new ObjectMapper();
        TypeReference<HashMap<String,Object>> typeRef = new TypeReference<HashMap<String,Object>>() {};

        return Files.lines(Paths.get(filePath))
                .map(line ->
                {
                    try
                    {
                        Map<String, Object> featuresMap = mapper.readValue(line, typeRef);
                        Map<FieldName, FieldValue> pmmlFeaturesMap = new HashMap<>();
                        List<InputField> inputFields = model.getInputFields();
                        for (InputField inputField : inputFields)
                        {
                            FieldName featureName = inputField.getName();
                            Object featureValue = featuresMap.get(featureName.getValue());
                            if (featureValue == null)
                            {
                                throw new RuntimeException(String.format("Missing feature value %s from the data set that model expected", featureName.getValue()));
                            }
                            FieldValue feature = inputField.prepare(featureValue.toString());
                            pmmlFeaturesMap.put(featureName, feature);
                        }

                        return(pmmlFeaturesMap);
                    }
                    catch (IOException ex)
                    {
                        throw new RuntimeException("Failed to parse feature json", ex);
                    }
                })
            .toArray(Map[]::new);
    }
}
