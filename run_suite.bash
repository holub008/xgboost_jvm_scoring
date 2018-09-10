#! /bin/bash

##############################
## run the end to end analysis
##############################

mkdir experiment_data/ pmml_models/ xgboost_raw_models/ plots/

Rscript build_models.R

# note this forgoes simpler @ execution notation and the -f option to support older versions of maven common on CentOS
(cd xgb_score && mvn clean install)
(cd xgb_score/ &&  mvn exec:java -Dexec.mainClass="kh.experiments.xgb_score.ExperimentSuite" -Dexec.args="../xgboost_raw_models ../pmml_models ../experiment_data ../features.fmap ../results.csv 20")

Rscript run_analysis.R
