#! /bin/bash

##############################
## run the end to end analysis
##############################

mkdir experiment_data/ pmml_models/ xgboost_raw_models/ plots/

Rscript build_models.R

mvn clean install -f xgb_score
mvn exec:java@suite

Rscript run_analysis.R
