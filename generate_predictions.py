import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

# 1. Load Datasets
print("Loading data...")
df_train = pd.read_csv('data/train-test.csv', parse_dates=['date'])
df_val = pd.read_csv('data/validation.csv', parse_dates=['date'])
df_dec = pd.read_csv('data/december-chart-inputs.csv', parse_dates=['date'])
df_template = pd.read_csv('data/validation-predictions-template.csv')

# 2. Rebuild Preprocessing & Imputation Pipelines
print("Imputing missing values...")

# Median weight from train
df_dec['weight'] = df_dec['weight'].astype(float)
weight_medians = df_train.groupby('equipment')['weight'].median()
for equip, med in weight_medians.items():
    df_train.loc[df_train['weight'].isnull() & (df_train['equipment'] == equip), 'weight'] = med
    df_val.loc[df_val['weight'].isnull() & (df_val['equipment'] == equip), 'weight'] = med
    df_dec.loc[df_dec['weight'].isnull() & (df_dec['equipment'] == equip), 'weight'] = med

# Market Index (rolling 7-day or global median)
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

# For December, we fill with the global median market index and median quote signal
df_dec['market_index'] = global_median_market
df_dec['quote_signal'] = df_train['quote_signal'].median()

# Standardize equipment types
df_train['equipment_code'] = df_train['equipment'].map({'Dry Van':0,'Flatbed':1,'Reefer':2})
df_val['equipment_code'] = df_val['equipment'].map({'Dry Van':0,'Flatbed':1,'Reefer':2})
df_dec['equipment_code'] = df_dec['equipment'].map({'Dry Van':0,'Flatbed':1,'Reefer':2})

# Outlier flag (top 1% rate outliers in training set)
df_train['is_rate_outlier'] = (df_train['posted_rate'] > df_train['posted_rate'].quantile(0.99)).astype(int)
df_val['is_rate_outlier'] = 0
df_dec['is_rate_outlier'] = 0

# 3. Feature Engineering
print("Engineering features...")
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

df_train = add_date_features(df_train)
df_val = add_date_features(df_val)
df_dec = add_date_features(df_dec)

# For December, we inject coordinates of Lexington (pickup) and Fort Wayne (delivery)
df_dec['pickup_lat'] = 36.99152
df_dec['pickup_lon'] = -84.99876
df_dec['delivery_lat'] = 41.31561
df_dec['delivery_lon'] = -85.36206

df_train = add_geo_features(df_train)
df_val = add_geo_features(df_val)
df_dec = add_geo_features(df_dec)

# Market signal interactions
df_train['market_x_quote'] = df_train['market_index'] * df_train['quote_signal']
df_val['market_x_quote'] = df_val['market_index'] * df_val['quote_signal']
df_dec['market_x_quote'] = df_dec['market_index'] * df_dec['quote_signal']

df_train['quote_over_market'] = df_train['quote_signal'] / df_train['market_index']
df_val['quote_over_market'] = df_val['quote_signal'] / df_val['market_index']
df_dec['quote_over_market'] = df_dec['quote_signal'] / df_dec['market_index']

# One-hot encoding of equipment dummies
eq_tr = pd.get_dummies(df_train['equipment'], prefix='equip', drop_first=True).astype(int)
eq_v = pd.get_dummies(df_val['equipment'], prefix='equip', drop_first=True).astype(int)
eq_d = pd.get_dummies(df_dec['equipment'], prefix='equip', drop_first=True).astype(int)

# Align dummies
for c in eq_tr.columns:
    if c not in eq_v.columns: eq_v[c] = 0
    if c not in eq_d.columns: eq_d[c] = 0
eq_v = eq_v[eq_tr.columns]
eq_d = eq_d[eq_tr.columns]

df_train = pd.concat([df_train, eq_tr], axis=1)
df_val = pd.concat([df_val, eq_v], axis=1)
df_dec = pd.concat([df_dec, eq_d], axis=1)

# Smoothed Target Encodings
global_mean = df_train['posted_rate'].mean()

def smoothed_target_encode(train_df, val_df, dec_df, col, target='posted_rate', k=20, gm=None):
    if gm is None: gm = train_df[target].mean()
    stats = train_df.groupby(col)[target].agg(['mean','count'])
    stats['encoded'] = (stats['count'] * stats['mean'] + k * gm) / (stats['count'] + k)
    train_enc = train_df[col].map(stats['encoded']).fillna(gm)
    val_enc   = val_df[col].map(stats['encoded']).fillna(gm)
    dec_enc   = dec_df[col].map(stats['encoded']).fillna(gm)
    return train_enc, val_enc, dec_enc, stats

df_train['pickup_enc'], df_val['pickup_enc'], df_dec['pickup_enc'], _ = \
    smoothed_target_encode(df_train, df_val, df_dec, 'pickup', k=20, gm=global_mean)

df_train['delivery_enc'], df_val['delivery_enc'], df_dec['delivery_enc'], _ = \
    smoothed_target_encode(df_train, df_val, df_dec, 'delivery', k=20, gm=global_mean)

df_train['lane_rate_enc'], df_val['lane_rate_enc'], df_dec['lane_rate_enc'], _ = \
    smoothed_target_encode(df_train, df_val, df_dec, 'lane', k=10, gm=global_mean)

# Lane counts
lane_counts = df_train['lane'].value_counts()
df_train['lane_count'] = df_train['lane'].map(lane_counts).fillna(1)
df_val['lane_count']   = df_val['lane'].map(lane_counts).fillna(1)
df_dec['lane_count']   = df_dec['lane'].map(lane_counts).fillna(1)

# Clip weights & transform target
df_train['log_rate'] = np.log1p(df_train['posted_rate'])
df_train['weight'] = df_train['weight'].clip(lower=0)
df_val['weight']   = df_val['weight'].clip(lower=0)
df_dec['weight']   = df_dec['weight'].clip(lower=0)

# Build feature matrices
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
X_val   = df_val[FEATURE_COLS].copy()
X_dec   = df_dec[FEATURE_COLS].copy()

# 4. Retrain Best Tuned Model on Full Training Set (Jan–Oct 2025)
print("Retraining final tuned CatBoost model on all 48,000 rows...")
with open('phase8_results.json') as f:
    p8_data = json.load(f)

final_params = p8_data['cat_best_params']
print(f"Optimal parameters: {final_params}")

final_model = CatBoostRegressor(**final_params, loss_function='RMSE', random_seed=42, verbose=0)
final_model.fit(X_train, y_train)

# 5. Generate validation_predictions.csv
print("Generating final validation set predictions...")
val_log_preds = final_model.predict(X_val)
val_preds = np.expm1(val_log_preds)

df_val_preds = pd.DataFrame({
    'load_id': df_val['load_id'],
    'predicted_rate': val_preds
})

# Verify ID alignment & values
assert len(df_val_preds) == 12000, "Validation set must contain exactly 12,000 predictions!"
assert (df_val_preds['predicted_rate'] > 0).all(), "All predicted rates must be strictly positive!"
df_val_preds.to_csv('validation_predictions.csv', index=False)
print("Saved validation_predictions.csv successfully!")

# 6. Fill predicted_rate in december-chart-inputs.csv
print("Generating December 2025 predictions...")
dec_log_preds = final_model.predict(X_dec)
dec_preds = np.expm1(dec_log_preds)

df_dec['predicted_rate'] = dec_preds
# Save back keeping the exact original columns
original_cols = ['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']
df_dec[original_cols].to_csv('data/december-chart-inputs.csv', index=False)
print("Updated data/december-chart-inputs.csv successfully!")

print("\nPhase 10 Code Execution Done!")
