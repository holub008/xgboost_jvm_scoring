library(r2pmml) # installable via devtools::install_git("git://github.com/jpmml/r2pmml.git")
library(xgboost)
library(dplyr)
library(RCurl)

censusIncomeText <- getURL('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
censusIncome <- read.csv(textConnection(censusIncomeText), header=F, stringsAsFactors = F)
colnames(censusIncome) <- c('age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
                            'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                            'hours_per_week', 'native_country', 'above_50k')
censusIncome <- censusIncome %>%
  mutate(
    above_50k = above_50k == '>50K'
  )

censusIncomeMat <- model.matrix( ~ . - above_50k,
                             censusIncome)
colnames(censusIncomeMat) <- sapply(colnames(censusIncomeMat), function(col) {
                                gsub(" ", "", col, fixed = TRUE)}) %>% 
                             unname()

setups <- expand.grid(
  n_trees = c(1, 5, 20, 50, 200, 500),
  tree_depth = c(2,3,5)
)

batch_sizes <- c(5, 10, 20, 50, 100, 1e3, 1e4, nrow(censusIncomeMat))

## note that we re-wrap the matrix as a data.frame to capture the design matrix (one-hot encoded), not the raw features
for (batch_size in batch_sizes) {
  dmatrix <- genDMatrix(as.integer(censusIncome$above_50k), 
                        as.data.frame(censusIncomeMat[sample(nrow(censusIncomeMat), batch_size), ]), 
                        paste0('experiment_data/', as.character(batch_size), '.svm'))
}

fmap <- genFMap(as.data.frame(censusIncomeMat))
writeFMap(fmap, 'features.fmap')

# this could be trivially optimized to build on the previous n_trees within each tree_depth. but that makes the code messier and runtime is fast enough
for (setup_ix in 1:nrow(setups)) {
  setup <- setups[setup_ix,]
  setup_name <- paste0(setup$n_trees, '_', setup$tree_depth)
  m_xgb <- xgboost(data = sp.dmatrix, nrounds = setup$n_trees, max_depth = setup$tree_depth, objective = "binary:logistic")
  
  xgb.save(m_xgb, paste0('xgboost_raw_models/', setup_name, '.model'))
  r2pmml(m_xgb, paste0('pmml_models/', setup_name, '.model'), fmap = fmap, response_name = "above_50k") 
}




