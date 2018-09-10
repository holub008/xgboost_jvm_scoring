package kh.experiments.xgb_score;

import java.util.List;

interface ExperimentRunner
{
    List<Long> run(int replicates);
}
