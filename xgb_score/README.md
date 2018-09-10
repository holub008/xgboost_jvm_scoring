# Experimenting with different model serialization and scoring libraries for the jvm

## Building
```bash
mvn clean install
```

## Running 

### XGBoost native
```bash
mvn exec:java -Dexec.mainClass="XGBoost4JModelRunnerer" -Dexec.args="../ny.model ../sp.svm 10"
```
