"""
Phase 7 standalone execution script.
Reproduces the full pipeline up to Phase 7 efficiently and saves model results.
"""
import json, warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── 1. Load data ──────────────────────────────────────────────────────────────
df_train = pd.read_csv('data/train-test.csv', parse_dates=['date'])
df_val   = pd.read_csv('data/validation.csv', parse_dates=['date'])

# ── 2. Phase 3 cleaning ───────────────────────────────────────────────────────
weight_medians = df_train.groupby('equipment')['weight'].median()

def fill_weight(df, medians):
    df = df.copy()
    for equip, med in medians.items():
        df.loc[df['weight'].isnull() & (df['equipment'] == equip), 'weight'] = med
    return df

df_train = fill_weight(df_train, weight_medians)
df_val   = fill_weight(df_val, weight_medians)

daily_market = (df_train.groupby('date')['market_index'].mean()
                .reset_index().rename(columns={'market_index': 'market_daily_mean'}))
daily_market = daily_market.sort_values('date').reset_index(drop=True)
daily_market['market_rolling7'] = daily_market['market_daily_mean'].rolling(7, min_periods=1).mean()
global_median_market = df_train['market_index'].median()

def fill_market_index(df, dm, fallback):
    df = df.copy()
    df = df.merge(dm[['date','market_rolling7']], on='date', how='left')
    df['market_rolling7'] = df['market_rolling7'].fillna(fallback)
    df['market_index'] = df['market_index'].fillna(df['market_rolling7'])
    df = df.drop(columns=['market_rolling7'])
    return df

df_train = fill_market_index(df_train, daily_market, global_median_market)
df_val   = fill_market_index(df_val, daily_market, global_median_market)

df_train['equipment_code'] = df_train['equipment'].map({'Dry Van':0,'Flatbed':1,'Reefer':2})
df_val['equipment_code']   = df_val['equipment'].map({'Dry Van':0,'Flatbed':1,'Reefer':2})
df_train['is_rate_outlier'] = (df_train['posted_rate'] > df_train['posted_rate'].quantile(0.99)).astype(int)
df_val['is_rate_outlier']   = 0

# ── 3. Phase 4 feature engineering ───────────────────────────────────────────
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
    if lon >= -80:   return 'NE' if lat >= 38 else 'SE'
    elif lon >= -90: return 'NE' if lat >= 39 else 'SE'
    elif lon >= -103: return 'MW'
    else:            return 'SW' if lat < 37 else 'W'

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
df_val   = add_date_features(df_val)
df_train = add_geo_features(df_train)
df_val   = add_geo_features(df_val)

df_train['market_x_quote']    = df_train['market_index'] * df_train['quote_signal']
df_val['market_x_quote']      = df_val['market_index']   * df_val['quote_signal']
df_train['quote_over_market']  = df_train['quote_signal'] / df_train['market_index']
df_val['quote_over_market']    = df_val['quote_signal']   / df_val['market_index']

eq_tr = pd.get_dummies(df_train['equipment'], prefix='equip', drop_first=True).astype(int)
eq_v  = pd.get_dummies(df_val['equipment'],   prefix='equip', drop_first=True).astype(int)
for c in eq_tr.columns:
    if c not in eq_v.columns: eq_v[c] = 0
eq_v = eq_v[eq_tr.columns]
df_train = pd.concat([df_train, eq_tr], axis=1)
df_val   = pd.concat([df_val,   eq_v],  axis=1)

global_mean = df_train['posted_rate'].mean()
df_train['pickup_enc'],  df_val['pickup_enc'],  _ = smoothed_target_encode(df_train, df_val, 'pickup',   k=20, gm=global_mean)
df_train['delivery_enc'],df_val['delivery_enc'],_ = smoothed_target_encode(df_train, df_val, 'delivery', k=20, gm=global_mean)
df_train['lane_rate_enc'],df_val['lane_rate_enc'],_ = smoothed_target_encode(df_train, df_val, 'lane', k=10, gm=global_mean)
lane_counts = df_train['lane'].value_counts().rename('lane_count')
df_train['lane_count'] = df_train['lane'].map(lane_counts).fillna(1)
df_val['lane_count']   = df_val['lane'].map(lane_counts).fillna(1)

df_train['log_rate'] = np.log1p(df_train['posted_rate'])
df_train['weight']   = df_train['weight'].clip(lower=0)
df_val['weight']     = df_val['weight'].clip(lower=0)

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

# ── 4. Phase 5 split ──────────────────────────────────────────────────────────
HOLDOUT_START = pd.Timestamp('2025-09-01')
train_mask    = df_train['date'] < HOLDOUT_START
holdout_mask  = df_train['date'] >= HOLDOUT_START

X_tr    = X_train[train_mask].reset_index(drop=True)
y_tr    = y_train[train_mask].reset_index(drop=True)
X_hold  = X_train[holdout_mask].reset_index(drop=True)
y_hold  = y_train[holdout_mask].reset_index(drop=True)

# ── 5. Metrics ────────────────────────────────────────────────────────────────
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

def evaluate(y_true_log, y_pred_log, label=''):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    rmse     = np.sqrt(mean_squared_error(y_true, y_pred))
    mae      = mean_absolute_error(y_true, y_pred)
    r2       = r2_score(y_true, y_pred)
    mape     = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    prefix = f'[{label}] ' if label else ''
    print(f'{prefix}RMSE (log): {rmse_log:.4f}  RMSE($): ${rmse:,.2f}  MAE($): ${mae:,.2f}  R²: {r2:.4f}  MAPE: {mape:.2f}%')
    return {'rmse_log': rmse_log, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

model_results = {}

# ── 6. Phase 6 baselines ──────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

baseline_results = {}
mean_pred = np.full(len(y_hold), y_tr.mean())
baseline_results['Global Mean'] = evaluate(y_hold, mean_pred, 'Global Mean')
lr_dist = LinearRegression().fit(X_tr[['distance']], y_tr)
baseline_results['Linear (distance)'] = evaluate(y_hold, lr_dist.predict(X_hold[['distance']]), 'Linear (distance)')
ridge_pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=10.0))])
ridge_pipe.fit(X_tr, y_tr)
baseline_results['Ridge (all features)'] = evaluate(y_hold, ridge_pipe.predict(X_hold), 'Ridge (all features)')
naive_rmse = baseline_results['Global Mean']['rmse']

# ── 7a. XGBoost ───────────────────────────────────────────────────────────────
import xgboost as xgb, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def xgb_objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 300, 1200),
        'max_depth':        trial.suggest_int('max_depth', 3, 8),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state': 42, 'tree_method': 'hist', 'verbosity': 0,
    }
    scores = []
    for tr_idx, val_idx in TimeSeriesSplit(n_splits=3).split(X_tr):
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx],
              eval_set=[(X_tr.iloc[val_idx], y_tr.iloc[val_idx])], verbose=False)
        pred = m.predict(X_tr.iloc[val_idx])
        scores.append(np.sqrt(np.mean((y_tr.iloc[val_idx] - pred) ** 2)))
    return np.mean(scores)

print("Tuning XGBoost...")
xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(xgb_objective, n_trials=40, show_progress_bar=False)
print(f"XGBoost best CV RMSE (log): {xgb_study.best_value:.4f}")
print(f"XGBoost best params: {xgb_study.best_params}")

best_xgb = xgb.XGBRegressor(**xgb_study.best_params, random_state=42, tree_method='hist', verbosity=0)
best_xgb.fit(X_tr, y_tr)
xgb_pred = best_xgb.predict(X_hold)
model_results['XGBoost'] = evaluate(y_hold, xgb_pred, 'XGBoost')

# ── 7b. LightGBM ──────────────────────────────────────────────────────────────
import lightgbm as lgb

def lgb_objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 300, 1500),
        'max_depth':         trial.suggest_int('max_depth', 3, 10),
        'num_leaves':        trial.suggest_int('num_leaves', 20, 200),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state': 42, 'verbosity': -1,
    }
    scores = []
    for tr_idx, val_idx in TimeSeriesSplit(n_splits=3).split(X_tr):
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx],
              eval_set=[(X_tr.iloc[val_idx], y_tr.iloc[val_idx])])
        pred = m.predict(X_tr.iloc[val_idx])
        scores.append(np.sqrt(np.mean((y_tr.iloc[val_idx] - pred) ** 2)))
    return np.mean(scores)

print("Tuning LightGBM...")
lgb_study = optuna.create_study(direction='minimize')
lgb_study.optimize(lgb_objective, n_trials=40, show_progress_bar=False)
print(f"LightGBM best CV RMSE (log): {lgb_study.best_value:.4f}")
print(f"LightGBM best params: {lgb_study.best_params}")

best_lgb = lgb.LGBMRegressor(**lgb_study.best_params, random_state=42, verbosity=-1)
best_lgb.fit(X_tr, y_tr)
lgb_pred = best_lgb.predict(X_hold)
model_results['LightGBM'] = evaluate(y_hold, lgb_pred, 'LightGBM')

# ── 7c. CatBoost ──────────────────────────────────────────────────────────────
from catboost import CatBoostRegressor
print("Training CatBoost...")
cat_model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6,
                               l2_leaf_reg=3.0, loss_function='RMSE',
                               random_seed=42, verbose=0)
cat_model.fit(X_tr, y_tr, eval_set=(X_hold, y_hold))
cat_pred = cat_model.predict(X_hold)
model_results['CatBoost'] = evaluate(y_hold, cat_pred, 'CatBoost')

# ── 7d. Random Forest ─────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=300, max_depth=16, min_samples_split=5,
                                  min_samples_leaf=2, max_features=0.6,
                                  random_state=42, n_jobs=-1)
rf_model.fit(X_tr, y_tr)
rf_pred = rf_model.predict(X_hold)
model_results['Random Forest'] = evaluate(y_hold, rf_pred, 'Random Forest')

# ── 7e. Comparison ────────────────────────────────────────────────────────────
all_results = {**baseline_results, **model_results}
comparison = pd.DataFrame(all_results).T[['rmse_log','rmse','mae','r2','mape']]
comparison.columns = ['RMSE (log)', 'RMSE ($)', 'MAE ($)', 'R²', 'MAPE (%)']
comparison = comparison.sort_values('RMSE ($)')

print("\n" + "="*65)
print("Full model comparison (hold-out Sep–Oct):")
print(comparison.round(4).to_string())

models_sorted = comparison.index.tolist()
colors = ['#2ca02c','#1f77b4','#ff7f0e','#9467bd','#d62728','#8c564b','#17becf']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
rmse_v = [all_results[m]['rmse'] for m in models_sorted[::-1]]
axes[0].barh(models_sorted[::-1], rmse_v, color=colors[:len(models_sorted)])
axes[0].set_title('RMSE ($) — Lower is Better'); axes[0].set_xlabel('RMSE ($)')
for i, v in enumerate(rmse_v):
    axes[0].text(v+3, i, f'${v:,.0f}', va='center', fontsize=8)

mape_v = [all_results[m]['mape'] for m in models_sorted[::-1]]
axes[1].barh(models_sorted[::-1], mape_v, color=colors[:len(models_sorted)])
axes[1].set_title('MAPE (%) — Lower is Better'); axes[1].set_xlabel('MAPE (%)')
for i, v in enumerate(mape_v):
    axes[1].text(v+0.1, i, f'{v:.1f}%', va='center', fontsize=8)

r2_v = [all_results[m]['r2'] for m in models_sorted[::-1]]
axes[2].barh(models_sorted[::-1], r2_v, color=colors[:len(models_sorted)])
axes[2].set_title('R² — Higher is Better'); axes[2].set_xlabel('R²')
axes[2].axvline(0, color='black', lw=0.8, linestyle='--')
for i, v in enumerate(r2_v):
    axes[2].text(max(v+0.002, 0.002), i, f'{v:.3f}', va='center', fontsize=8)

plt.suptitle('Phase 7 — All Models Comparison (Hold-out: Sep–Oct 2025)', fontsize=13)
plt.tight_layout()
plt.savefig('eda_7e_model_comparison.png', dpi=130, bbox_inches='tight')
plt.close()
print("Saved eda_7e_model_comparison.png")

# ── 7f. Feature importance ────────────────────────────────────────────────────
best_name = comparison.index[0]
print(f"\nBest model: {best_name}")

if best_name == 'XGBoost':
    fi = pd.Series(best_xgb.feature_importances_, index=FEATURE_COLS)
elif best_name == 'LightGBM':
    fi = pd.Series(best_lgb.feature_importances_, index=FEATURE_COLS)
elif best_name == 'CatBoost':
    fi = pd.Series(cat_model.get_feature_importance(), index=FEATURE_COLS)
else:
    fi = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS)

fi = fi.sort_values(ascending=False)
print("Top 10 features:")
print(fi.head(10).round(4).to_string())

fig, ax = plt.subplots(figsize=(10, 8))
fi.head(20).plot(kind='barh', ax=ax, color='steelblue')
ax.invert_yaxis()
ax.set_title(f'Top 20 Feature Importances — {best_name}')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('eda_7f_feature_importance.png', dpi=130, bbox_inches='tight')
plt.close()
print("Saved eda_7f_feature_importance.png")

# ── 7g. Retrain on full data ──────────────────────────────────────────────────
print(f"\nRetraining {best_name} on full Jan–Oct data ({len(X_train):,} rows)...")
if best_name == 'XGBoost':
    final_model = xgb.XGBRegressor(**xgb_study.best_params, random_state=42, tree_method='hist', verbosity=0)
elif best_name == 'LightGBM':
    final_model = lgb.LGBMRegressor(**lgb_study.best_params, random_state=42, verbosity=-1)
elif best_name == 'CatBoost':
    final_model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6,
                                     l2_leaf_reg=3.0, loss_function='RMSE', random_seed=42, verbose=0)
else:
    final_model = RandomForestRegressor(n_estimators=300, max_depth=16, min_samples_split=5,
                                         min_samples_leaf=2, max_features=0.6, random_state=42, n_jobs=-1)

final_model.fit(X_train, y_train)

final_hold_pred = final_model.predict(X_hold)
print(f"\nFinal {best_name} — hold-out after full retrain:")
final_metrics = evaluate(y_hold, final_hold_pred, label=f'Final {best_name}')

val_predictions_log = final_model.predict(X_val)
val_predictions     = np.expm1(val_predictions_log)
print(f"\nValidation predictions (Nov–Dec):")
print(f"  Count: {len(val_predictions):,}")
print(f"  Min  : ${val_predictions.min():,.2f}")
print(f"  Max  : ${val_predictions.max():,.2f}")
print(f"  Mean : ${val_predictions.mean():,.2f}")
print(f"  Std  : ${val_predictions.std():,.2f}")

# ── 7h. Diagnostics chart ─────────────────────────────────────────────────────
true_hold  = np.expm1(y_hold)
pred_hold  = np.expm1(final_hold_pred)
residuals  = pred_hold - true_hold

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(true_hold, pred_hold, alpha=0.15, s=6, color='steelblue')
mv = max(true_hold.max(), pred_hold.max())
axes[0].plot([0,mv],[0,mv],'r--',lw=1.5,label='Perfect')
axes[0].set_title(f'{best_name}: Predicted vs Actual (Hold-out)')
axes[0].set_xlabel('Actual posted_rate ($)'); axes[0].set_ylabel('Predicted posted_rate ($)')
axes[0].legend()

axes[1].hist(residuals, bins=80, color='darkorange', edgecolor='white', lw=0.3)
axes[1].axvline(0, color='red', lw=1.5, linestyle='--')
axes[1].set_title(f'{best_name}: Residuals Distribution')
axes[1].set_xlabel('Residual ($)'); axes[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig('eda_7h_best_model_diagnostics.png', dpi=130, bbox_inches='tight')
plt.close()
print("Saved eda_7h_best_model_diagnostics.png")

print(f"\nResiduals — mean: ${residuals.mean():,.2f}, std: ${residuals.std():,.2f}")
print(f"Within $200: {(residuals.abs()<200).mean()*100:.1f}%")
print(f"Within $500: {(residuals.abs()<500).mean()*100:.1f}%")

# ── 7i. Summary ───────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(" Phase 7 — Advanced Modeling Summary")
print("="*65)
print()
print(f"{'Model':<25} {'RMSE ($)':>10} {'MAE ($)':>9} {'R²':>7} {'MAPE':>8}")
print("  " + "-"*57)
for name in comparison.index:
    m = all_results[name]
    print(f"  {name:<25} ${m['rmse']:>8,.0f} ${m['mae']:>7,.0f} {m['r2']:>7.3f} {m['mape']:>7.1f}%")
print()
print(f"Best model : {best_name}")
print(f"  RMSE ($) : ${all_results[best_name]['rmse']:,.2f}")
print(f"  MAPE     : {all_results[best_name]['mape']:.2f}%")
print(f"  R²       : {all_results[best_name]['r2']:.4f}")
print()
best_rmse  = all_results[best_name]['rmse']
ridge_rmse = baseline_results['Ridge (all features)']['rmse']
print("Improvement over baselines:")
print(f"  vs Naive mean  : {(naive_rmse - best_rmse)/naive_rmse*100:+.1f}% RMSE")
print(f"  vs Ridge       : {(ridge_rmse - best_rmse)/ridge_rmse*100:+.1f}% RMSE")
print()
print(f"Validation predictions generated: {len(val_predictions):,} rows")

# Save best model params and results for reference
import json as _json
summary = {
    'best_model': best_name,
    'best_params': xgb_study.best_params if best_name == 'XGBoost' else
                   lgb_study.best_params if best_name == 'LightGBM' else {},
    'hold_out_metrics': all_results[best_name],
    'all_metrics': {k: {kk: float(vv) for kk,vv in v.items()} for k,v in all_results.items()},
    'feature_importance_top10': fi.head(10).to_dict(),
}
with open('phase7_results.json', 'w') as f:
    _json.dump(summary, f, indent=2)
print("Saved phase7_results.json")
