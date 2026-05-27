# =========================================================
# TRAFFIC DEMAND PREDICTION
# FINAL FULLY FIXED VERSION
# =========================================================

# INSTALL REQUIRED LIBRARIES:
# pip3 install pandas numpy scikit-learn catboost

# =========================================================
# IMPORTS
# =========================================================

import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# LOAD DATA
# =========================================================

train = pd.read_csv("/Users/jasmon/Downloads/dataset/train.csv")
test = pd.read_csv("/Users/jasmon/Downloads/dataset/test.csv")

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

# =========================================================
# FIX DAY COLUMN
# =========================================================

train['day'] = (
    train['day']
    .astype(str)
    .str.extract(r'(\d+)')[0]
    .astype(int)
)

test['day'] = (
    test['day']
    .astype(str)
    .str.extract(r'(\d+)')[0]
    .astype(int)
)

print("\nUnique Days:")
print(sorted(train['day'].unique()))

# =========================================================
# TIME FEATURE ENGINEERING
# =========================================================

def process_time(df):

    # Split timestamp
    df[['hour', 'minute']] = (
        df['timestamp']
        .str.split(':', expand=True)
    )

    df['hour'] = df['hour'].astype(int)
    df['minute'] = df['minute'].astype(int)

    # 15-minute slots
    df['slot'] = (
        df['hour'] * 4 +
        (df['minute'] // 15)
    )

    # Cyclical encoding
    df['sin_time'] = np.sin(
        2 * np.pi * df['slot'] / 96
    )

    df['cos_time'] = np.cos(
        2 * np.pi * df['slot'] / 96
    )

    # Peak hours
    df['peak_hour'] = (
        df['hour']
        .isin([7,8,9,17,18,19])
        .astype(int)
    )

    return df

train = process_time(train)
test = process_time(test)

# =========================================================
# GEOHASH FEATURES
# =========================================================

train['geo_4'] = (
    train['geohash']
    .astype(str)
    .str[:4]
)

train['geo_5'] = (
    train['geohash']
    .astype(str)
    .str[:5]
)

test['geo_4'] = (
    test['geohash']
    .astype(str)
    .str[:4]
)

test['geo_5'] = (
    test['geohash']
    .astype(str)
    .str[:5]
)

# =========================================================
# TARGET ENCODING FEATURES
# =========================================================

# Geohash mean demand
geo_mean = (
    train.groupby('geohash')['demand']
    .mean()
)

train['geo_mean_demand'] = (
    train['geohash']
    .map(geo_mean)
)

test['geo_mean_demand'] = (
    test['geohash']
    .map(geo_mean)
)

# Slot mean demand
slot_mean = (
    train.groupby('slot')['demand']
    .mean()
)

train['slot_mean_demand'] = (
    train['slot']
    .map(slot_mean)
)

test['slot_mean_demand'] = (
    test['slot']
    .map(slot_mean)
)

# Road type mean demand
road_mean = (
    train.groupby('RoadType')['demand']
    .mean()
)

train['road_mean_demand'] = (
    train['RoadType']
    .map(road_mean)
)

test['road_mean_demand'] = (
    test['RoadType']
    .map(road_mean)
)

# =========================================================
# DAY 48 LOOKUP FEATURE
# =========================================================

day48 = train[
    train['day'] == 48
]

lookup = (
    day48.groupby(
        ['geohash', 'slot']
    )['demand']
    .mean()
)

train['demand_day48'] = train.apply(
    lambda row: lookup.get(
        (row['geohash'], row['slot']),
        np.nan
    ),
    axis=1
)

test['demand_day48'] = test.apply(
    lambda row: lookup.get(
        (row['geohash'], row['slot']),
        np.nan
    ),
    axis=1
)

global_mean = train['demand'].mean()

train['demand_day48'] = (
    train['demand_day48']
    .fillna(global_mean)
)

test['demand_day48'] = (
    test['demand_day48']
    .fillna(global_mean)
)

# =========================================================
# EXTRA FEATURES
# =========================================================

train['lane_temp_ratio'] = (
    train['NumberofLanes']
    /
    (train['Temperature'] + 1)
)

test['lane_temp_ratio'] = (
    test['NumberofLanes']
    /
    (test['Temperature'] + 1)
)

# =========================================================
# CLEAN NaNs + INFs
# =========================================================

train = train.replace([np.inf, -np.inf], np.nan)
test = test.replace([np.inf, -np.inf], np.nan)

# ---------- TRAIN ----------

for col in train.columns:

    if col == 'demand':
        continue

    if train[col].dtype == 'object':

        train[col] = (
            train[col]
            .fillna("Unknown")
            .astype(str)
        )

    else:

        median_val = train[col].median()

        train[col] = (
            train[col]
            .fillna(median_val)
        )

# ---------- TEST ----------

for col in test.columns:

    if test[col].dtype == 'object':

        test[col] = (
            test[col]
            .fillna("Unknown")
            .astype(str)
        )

    else:

        median_val = test[col].median()

        test[col] = (
            test[col]
            .fillna(median_val)
        )

# =========================================================
# FEATURES + TARGET
# =========================================================

X = train.drop(
    columns=['demand', 'Index']
)

y = train['demand']

# =========================================================
# TRAIN VALIDATION SPLIT
# =========================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_test = test.drop(columns=['Index'])

print("\nX_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nX_valid:", X_valid.shape)
print("y_valid:", y_valid.shape)

# =========================================================
# CATEGORICAL FEATURES
# =========================================================

cat_features = [
    'geohash',
    'timestamp',
    'RoadType',
    'LargeVehicles',
    'Landmarks',
    'Weather',
    'geo_4',
    'geo_5'
]

# =========================================================
# FORCE STRING TYPE
# IMPORTANT FOR CATBOOST
# =========================================================

for col in cat_features:

    X_train[col] = X_train[col].astype(str)
    X_valid[col] = X_valid[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# =========================================================
# MODEL
# =========================================================

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=8,
    loss_function='RMSE',
    eval_metric='R2',
    random_seed=42,
    verbose=200
)

# =========================================================
# TRAIN MODEL
# =========================================================

print("\nTraining model...\n")

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

# =========================================================
# VALIDATION SCORE
# =========================================================

valid_pred = model.predict(X_valid)

score = r2_score(
    y_valid,
    valid_pred
)

print("\nValidation R2 Score:", score)
print("Approx Competition Score:", score * 100)

# =========================================================
# FINAL TRAINING
# =========================================================

X_full = train.drop(
    columns=['demand', 'Index']
)

y_full = train['demand']

for col in cat_features:

    X_full[col] = X_full[col].astype(str)

final_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=8,
    loss_function='RMSE',
    eval_metric='R2',
    random_seed=42,
    verbose=200
)

print("\nTraining final model...\n")

final_model.fit(
    X_full,
    y_full,
    cat_features=cat_features
)

# =========================================================
# TEST PREDICTION
# =========================================================

test_pred = final_model.predict(X_test)

# =========================================================
# CREATE SUBMISSION
# =========================================================

submission = pd.DataFrame({
    'Index': test['Index'],
    'demand': test_pred
})

submission.to_csv(
    "/Users/jasmon/Downloads/dataset/submission.csv",
    index=False
)

print("\nsubmission.csv generated successfully!")

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    'Feature': X_full.columns,
    'Importance': final_model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop 20 Features:\n")
print(importance.head(20))

# =========================================================
# DONE
# =========================================================

print("\nPipeline completed successfully.")