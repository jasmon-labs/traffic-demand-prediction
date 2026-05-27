# Traffic Demand Prediction

Machine learning pipeline for traffic demand forecasting using CatBoost and spatio-temporal feature engineering.

## Team Project

This repository contains:
- Dataset files
- Final submission file
- Complete training pipeline
- Feature engineering workflow

## Features Engineered

- Geohash encoding
- Time-slot segmentation
- Cyclical time encoding
- Day-48 historical lookup feature
- Target encoding features
- Road/weather aggregation features

## Model Used

- CatBoost Regressor

## Validation Performance

- R² Score: ~0.9926
- Approx competition score: ~99.26

## Important Insight

The strongest predictive feature was:

```text
same geohash + same time slot historical demand
```

showing strong spatio-temporal repetition patterns in traffic demand.

## Files

```text
train.csv
test.csv
sample_submission.csv
submission.csv
traffic_demand_solution.py
```

## Run

```bash
pip3 install pandas numpy scikit-learn catboost
python3 traffic_demand_solution.py
```
