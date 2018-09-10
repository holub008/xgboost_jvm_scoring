# Experimenting scoring libraries for the jvm

## Building
```bash
mvn clean install
```

## Running 

### xgboost-predictor-java
```bash
mvn exec:java@XGB_jvm"
```

### xgboost4j
```bash
mvn exec:java@XGB_native"
```

### jpmml
```bash
mvn exec:java@pmml"
```

### full suite
```bash
mvn exec:java@suite"
```
