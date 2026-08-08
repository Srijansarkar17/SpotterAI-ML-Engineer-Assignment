import json
import warnings
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# 1. Load data
print("Loading data...")
df_train = pd.read_csv('data/train-test.csv', parse_dates=['date'])
df_val = pd.read_csv('data/validation.csv', parse_dates=['date'])

# Impute weight
weight_medians = df_train.groupby('equipment')['weight'].median()
for equip, med in weight_medians.items():
    df_train.loc[df_train['weight'].isnull() & (df_train['equipment'] == equip), 'weight'] = med
    df_val.loc[df_val['weight'].isnull() & (df_val['equipment'] == equip), 'weight'] = med

# Impute market index
daily_market = df_train.groupby('date')['market_index'].mean().reset_index().rename(columns={'market_index': 'market_daily_mean'})
daily_market = daily_market.sort_values('date').reset_index(drop=True)
daily_market['market_rolling7'] = daily_market['market_daily_mean'].rolling(7, min_periods=1).mean()
global_median_market = df_train['market_index'].median()

def fill_market_index(df, dm, fallback):
    df = df.copy()
    df = df.merge(dm[['date','market_rolling7']], on='date', how='left')
    df['market_rolling7'] = df['market_rolling7'].fillna(fallback)
    df['market_index'] = df['market_index'].fillna(df['market_rolling7'])
    return df.drop(columns=['market_rolling7'])

df_train = fill_market_index(df_train, daily_market, global_median_market)
df_val = fill_market_index(df_val, daily_market, global_median_market)

df_train['equipment_code'] = df_train['equipment'].map({'Dry Van':0,'Flatbed':1,'Reefer':2})
df_val['equipment_code'] = df_val['equipment'].map({'Dry Van':0,'Flatbed':1,'Reefer':2})
df_train['is_rate_outlier'] = (df_train['posted_rate'] > df_train['posted_rate'].quantile(0.99)).astype(int)
df_val['is_rate_outlier'] = 0

START_DATE = pd.Timestamp('2025-01-01')

def add_date_features(df):
    df = df.copy()
    df['year']             = df['date'].dt.year
    df['month']            = df['date'].dt.month
    df['day']              = df['date'].dt.day
    df['day_of_week']      = df['date'].dt.dayofweek
    df['is_weekend']       = (df['day_of_week'] >= 5).astype(int)
    df['week_of_year']     = df['date'].dt.isocalendar().week.astype(int)
    df['quarter']          = df['date'].dt.quarter
    df['day_of_year']      = df['date'].dt.dayofyear
    df['days_since_start'] = (df['date'] - START_DATE).dt.days
    return df

def haversine_miles_vec(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def assign_region(lat, lon):
    if lon >= -80:    return 'NE' if lat >= 38 else 'SE'
    elif lon >= -90:  return 'NE' if lat >= 39 else 'SE'
    elif lon >= -103: return 'MW'
    else:             return 'SW' if lat < 37 else 'W'

def add_geo_features(df):
    df = df.copy()
    df['haversine_dist']    = haversine_miles_vec(df['pickup_lat'], df['pickup_lon'],
                                                   df['delivery_lat'], df['delivery_lon'])
    df['distance_diff']     = df['distance'] - df['haversine_dist']
    df['dist_ratio']        = df['distance'] / df['haversine_dist']
    df['delta_lat']         = df['delivery_lat'] - df['pickup_lat']
    df['delta_lon']         = df['delivery_lon'] - df['pickup_lon']
    df['pickup_region']     = df.apply(lambda r: assign_region(r['pickup_lat'], r['pickup_lon']), axis=1)
    df['delivery_region']   = df.apply(lambda r: assign_region(r['delivery_lat'], r['delivery_lon']), axis=1)
    df['lane']              = df['pickup'] + '_' + df['delivery']
    df['is_cross_regional'] = (df['pickup_region'] != df['delivery_region']).astype(int)
    return df

def smoothed_target_encode(train_df, val_df, col, target='posted_rate', k=20, gm=None):
    if gm is None: gm = train_df[target].mean()
    stats = train_df.groupby(col)[target].agg(['mean','count'])
    stats['encoded'] = (stats['count'] * stats['mean'] + k * gm) / (stats['count'] + k)
    return train_df[col].map(stats['encoded']).fillna(gm), val_df[col].map(stats['encoded']).fillna(gm), stats

df_train = add_date_features(df_train)
df_val = add_date_features(df_val)
df_train = add_geo_features(df_train)
df_val = add_geo_features(df_val)

df_train['market_x_quote'] = df_train['market_index'] * df_train['quote_signal']
df_val['market_x_quote'] = df_val['market_index'] * df_val['quote_signal']
df_train['quote_over_market'] = df_train['quote_signal'] / df_train['market_index']
df_val['quote_over_market'] = df_val['quote_signal'] / df_val['market_index']

eq_tr = pd.get_dummies(df_train['equipment'], prefix='equip', drop_first=True).astype(int)
eq_v = pd.get_dummies(df_val['equipment'], prefix='equip', drop_first=True).astype(int)
for c in eq_tr.columns:
    if c not in eq_v.columns: eq_v[c] = 0
eq_v = eq_v[eq_tr.columns]
df_train = pd.concat([df_train, eq_tr], axis=1)
df_val = pd.concat([df_val, eq_v], axis=1)

global_mean = df_train['posted_rate'].mean()
df_train['pickup_enc'], df_val['pickup_enc'], _ = smoothed_target_encode(df_train, df_val, 'pickup', k=20, gm=global_mean)
df_train['delivery_enc'], df_val['delivery_enc'], _ = smoothed_target_encode(df_train, df_val, 'delivery', k=20, gm=global_mean)
df_train['lane_rate_enc'], df_val['lane_rate_enc'], _ = smoothed_target_encode(df_train, df_val, 'lane', k=10, gm=global_mean)
lane_counts = df_train['lane'].value_counts()
df_train['lane_count'] = df_train['lane'].map(lane_counts).fillna(1)
df_val['lane_count'] = df_val['lane'].map(lane_counts).fillna(1)

df_train['log_rate'] = np.log1p(df_train['posted_rate'])
df_train['weight'] = df_train['weight'].clip(lower=0)
df_val['weight'] = df_val['weight'].clip(lower=0)

FEATURE_COLS = [
    'distance','haversine_dist','distance_diff','dist_ratio','delta_lat','delta_lon',
    'weight','equipment_code','equip_Flatbed','equip_Reefer',
    'market_index','quote_signal','market_x_quote','quote_over_market',
    'month','day_of_week','is_weekend','week_of_year','quarter','day_of_year','days_since_start',
    'pickup_enc','delivery_enc','lane_rate_enc','lane_count',
    'is_cross_regional','is_rate_outlier',
]

X_train = df_train[FEATURE_COLS].copy()
y_train = df_train['log_rate'].copy()
X_val = df_val[FEATURE_COLS].copy()

HOLDOUT_START = pd.Timestamp('2025-09-01')
train_mask = df_train['date'] < HOLDOUT_START
holdout_mask = df_train['date'] >= HOLDOUT_START
X_tr = X_train[train_mask].reset_index(drop=True)
y_tr = y_train[train_mask].reset_index(drop=True)
X_hold = X_train[holdout_mask].reset_index(drop=True)
y_hold = y_train[holdout_mask].reset_index(drop=True)

print(f"Data split - X_tr: {X_tr.shape}, X_hold: {X_hold.shape}")

def evaluate(y_true_log, y_pred_log, label=''):
    yt = np.expm1(y_true_log)
    yp = np.expm1(y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    mape = np.mean(np.abs((yt - yp) / yt)) * 100
    print(f"[{label}] RMSE(log)={rmse_log:.4f}  RMSE($)=${rmse:,.2f}  MAE($)=${mae:,.2f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {'rmse_log': rmse_log, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

# Use TimeSeriesSplit
optuna.logging.set_verbosity(optuna.logging.WARNING)
tscv = TimeSeriesSplit(n_splits=3)

# 1. Tune LightGBM (with single thread to prevent OpenMP deadlocks on Mac)
def lgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
        'random_state': 42, 'verbosity': -1, 'n_jobs': 1
    }
    scores = []
    for tr_idx, val_idx in tscv.split(X_tr):
        Xf, Xv = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
        yf, yv = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]
        m = lgb.LGBMRegressor(**params)
        m.fit(Xf, yf, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(30, verbose=False)])
        scores.append(np.sqrt(np.mean((yv - m.predict(Xv))**2)))
    return np.mean(scores)

print("Tuning LightGBM (15 trials)...")
lgb_study = optuna.create_study(direction='minimize')
lgb_study.optimize(lgb_objective, n_trials=15)
print(f"LGBM Best Log RMSE: {lgb_study.best_value:.4f}")

# 2. Tune XGBoost (with n_jobs=1)
def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
        'random_state': 42, 'tree_method': 'hist', 'verbosity': 0, 'n_jobs': 1
    }
    scores = []
    for tr_idx, val_idx in tscv.split(X_tr):
        Xf, Xv = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
        yf, yv = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]
        m = xgb.XGBRegressor(**params, early_stopping_rounds=30)
        m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
        scores.append(np.sqrt(np.mean((yv - m.predict(Xv))**2)))
    return np.mean(scores)

print("Tuning XGBoost (15 trials)...")
xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(xgb_objective, n_trials=15)
print(f"XGBoost Best Log RMSE: {xgb_study.best_value:.4f}")

# 3. Tune CatBoost
def cat_objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
        'loss_function': 'RMSE', 'random_seed': 42, 'verbose': 0
    }
    scores = []
    for tr_idx, val_idx in tscv.split(X_tr):
        Xf, Xv = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
        yf, yv = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]
        m = CatBoostRegressor(**params)
        m.fit(Xf, yf, eval_set=(Xv, yv))
        scores.append(np.sqrt(np.mean((yv - m.predict(Xv))**2)))
    return np.mean(scores)

print("Tuning CatBoost (10 trials)...")
cat_study = optuna.create_study(direction='minimize')
cat_study.optimize(cat_objective, n_trials=10)
print(f"CatBoost Best Log RMSE: {cat_study.best_value:.4f}")

# Train the best models on X_tr, evaluate on X_hold
best_lgb = lgb.LGBMRegressor(**lgb_study.best_params, random_state=42, verbosity=-1, n_jobs=1)
best_lgb.fit(X_tr, y_tr)
lgb_m = evaluate(y_hold, best_lgb.predict(X_hold), 'LightGBM (tuned)')

best_xgb = xgb.XGBRegressor(**xgb_study.best_params, random_state=42, tree_method='hist', verbosity=0, n_jobs=1)
best_xgb.fit(X_tr, y_tr)
xgb_m = evaluate(y_hold, best_xgb.predict(X_hold), 'XGBoost (tuned)')

best_cat = CatBoostRegressor(**cat_study.best_params, loss_function='RMSE', random_seed=42, verbose=0)
best_cat.fit(X_tr, y_tr)
cat_m = evaluate(y_hold, best_cat.predict(X_hold), 'CatBoost (tuned)')

all_p8 = {
    'LightGBM (tuned)': lgb_m,
    'XGBoost (tuned)': xgb_m,
    'CatBoost (tuned)': cat_m
}
best_name = min(all_p8, key=lambda k: all_p8[k]['rmse'])
print(f"\nOverall Best Tuned Model on hold-out: {best_name}")

# Retrain best tuned model on full data
if best_name == 'LightGBM (tuned)':
    final_model = lgb.LGBMRegressor(**lgb_study.best_params, random_state=42, verbosity=-1, n_jobs=1)
elif best_name == 'XGBoost (tuned)':
    final_model = xgb.XGBRegressor(**xgb_study.best_params, random_state=42, tree_method='hist', verbosity=0, n_jobs=1)
else:
    final_model = CatBoostRegressor(**cat_study.best_params, loss_function='RMSE', random_seed=42, verbose=0)

final_model.fit(X_train, y_train)
val_preds = np.expm1(final_model.predict(X_val))

# Save results
summary = {
    'best_model': best_name,
    'lgb_best_params': lgb_study.best_params,
    'lgb_best_cv_rmse_log': lgb_study.best_value,
    'xgb_best_params': xgb_study.best_params,
    'xgb_best_cv_rmse_log': xgb_study.best_value,
    'cat_best_params': cat_study.best_params,
    'cat_best_cv_rmse_log': cat_study.best_value,
    'all_metrics': {k: {kk: float(vv) for kk,vv in v.items()} for k,v in all_p8.items()}
}

with open('phase8_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved phase8_results.json successfully!")
