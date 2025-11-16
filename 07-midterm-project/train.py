#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

# -------------------------------------------------
# 1. Load the data
# -------------------------------------------------

df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
df.columns = df.columns.str.lower().str.replace(' ','_')

df = df.astype(int)

df['sex'] = df['sex'].map({0: 'female', 1: 'male'})

# -------------------------------------------------
# 2. Train/val/test split
# -------------------------------------------------

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.diabetes_binary.values
y_val = df_val.diabetes_binary.values
y_test = df_test.diabetes_binary.values

del df_train['diabetes_binary']
del df_val['diabetes_binary']

# -------------------------------------------------
# 3. DictVectorizer
# -------------------------------------------------

dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

# -------------------------------------------------
# 4. Train final Random Forest model
# -------------------------------------------------

print('training the final model')

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=1
)

rf.fit(X_train, y_train)

# evaluate on validation set
y_pred = rf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print(f"Validation AUC: {auc:.4f}")

# -------------------------------------------------
# 5. Save model + DictVectorizer
# -------------------------------------------------

output_file = "rf_model.bin"

with open(output_file, "wb") as f_out:
    pickle.dump((dv, rf), f_out)

print(f"Model saved to {output_file}")
