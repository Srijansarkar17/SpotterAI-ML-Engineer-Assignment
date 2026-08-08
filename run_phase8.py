"""
Phase 8 — Hyperparameter Tuning via Optuna (100 trials each)
Tunes: LightGBM, XGBoost, and CatBoost
Uses: TimeSeriesSplit(n_splits=5) + early stopping per fold
Saves: phase8_results.json + charts
"""
import json, warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── 1. Rebuild full pipeline (same as Phase 7) ────────────────────────────────
print("Loading and preparing data...")

df_train = pd.read_csv('data/train-test.csv', parse_dates=['date'])
df_val   = pd.read_csv('data/validation.csv', parse_dates=['date'])

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
    return df.drop(columns=['market_rolling7'])

df_train = fill_market_index(df_train, daily_market, global_median_market)
df_val   = fill_market_index(df_val, daily_market, global_median_market)

df_train['equipment_code'] = df_train['equipment'].map({'Dry Van':0,'Flatbed':1,'Reefer':2})
df_val['equipment_code']   = df_val['equipment'].map({'Dry Van':0,'Flatbed':1,'Reefer':2})
df_train['is_rate_outlier'] = (df_train['posted_rate'] > df_train['posted_rate'].quantile(0.99)).astype(int)
df_val['is_rate_outlier']   = 0

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
df_val   = add_date_features(df_val)
df_train = add_geo_features(df_train)
df_val   = add_geo_features(df_val)

df_train['market_x_quote']   = df_train['market_index'] * df_train['quote_signal']
df_val['market_x_quote']     = df_val['market_index']   * df_val['quote_signal']
df_train['quote_over_market'] = df_train['quote_signal'] / df_train['market_index']
df_val['quote_over_market']   = df_val['quote_signal']   / df_val['market_index']

eq_tr = pd.get_dummies(df_train['equipment'], prefix='equip', drop_first=True).astype(int)
eq_v  = pd.get_dummies(df_val['equipment'],   prefix='equip', drop_first=True).astype(int)
for c in eq_tr.columns:
    if c not in eq_v.columns: eq_v[c] = 0
eq_v = eq_v[eq_tr.columns]
df_train = pd.concat([df_train, eq_tr], axis=1)
df_val   = pd.concat([df_val,   eq_v],  axis=1)

global_mean = df_train['posted_rate'].mean()
df_train['pickup_enc'],   df_val['pickup_enc'],   _ = smoothed_target_encode(df_train, df_val, 'pickup',   k=20, gm=global_mean)
df_train['delivery_enc'], df_val['delivery_enc'], _ = smoothed_target_encode(df_train, df_val, 'delivery', k=20, gm=global_mean)
df_train['lane_rate_enc'],df_val['lane_rate_enc'],_ = smoothed_target_encode(df_train, df_val, 'lane',     k=10, gm=global_mean)
lane_counts = df_train['lane'].value_counts()
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

HOLDOUT_START = pd.Timestamp('2025-09-01')
train_mask    = df_train['date'] < HOLDOUT_START
holdout_mask  = df_train['date'] >= HOLDOUT_START
X_tr   = X_train[train_mask].reset_index(drop=True)
y_tr   = y_train[train_mask].reset_index(drop=True)
X_hold = X_train[holdout_mask].reset_index(drop=True)
y_hold = y_train[holdout_mask].reset_index(drop=True)

print(f"Data ready — X_tr: {X_tr.shape}, X_hold: {X_hold.shape}")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

def evaluate(y_true_log, y_pred_log, label=''):
    yt = np.expm1(y_true_log)
    yp = np.expm1(y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    rmse     = np.sqrt(mean_squared_error(yt, yp))
    mae      = mean_absolute_error(yt, yp)
    r2       = r2_score(yt, yp)
    mape     = np.mean(np.abs((yt - yp) / yt)) * 100
    print(f"[{label}]  RMSE(log)={rmse_log:.4f}  RMSE($)=${rmse:,.2f}  MAE($)=${mae:,.2f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {'rmse_log': rmse_log, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

# Phase 7 CatBoost reference
from catboost import CatBoostRegressor
print("\nPhase 7 reference: training CatBoost (fixed defaults)...")
cat_ref = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6,
                             l2_leaf_reg=3.0, loss_function='RMSE', random_seed=42, verbose=0)
cat_ref.fit(X_tr, y_tr, eval_set=(X_hold, y_hold))
cat_ref_metrics = evaluate(y_hold, cat_ref.predict(X_hold), 'CatBoost (Phase7)')

p8_results = {}
tscv = TimeSeriesSplit(n_splits=5)

# ── 8a. Tune LightGBM — 100 trials ───────────────────────────────────────────
import lightgbm as lgb, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def lgb_objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 2000),
        'learning_rate':     trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth':         trial.suggest_int('max_depth', 3, 12),
        'num_leaves':        trial.suggest_int('num_leaves', 16, 256),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbosity': -1,
        'n_jobs': -1,
    }
    scores = []
    for tr_idx, val_idx in tscv.split(X_tr):
        Xf, Xv = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
        yf, yv = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]
        m = lgb.LGBMRegressor(**params)
        m.fit(Xf, yf,
              eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
        scores.append(np.sqrt(np.mean((yv - m.predict(Xv)) ** 2)))
    return np.mean(scores)

print("\n[8a] Tuning LightGBM — 100 Optuna trials, 5-fold TimeSeriesSplit with early stopping...")
lgb_study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
lgb_study.optimize(lgb_objective, n_trials=100, show_progress_bar=True)

print(f"  Best CV RMSE (log): {lgb_study.best_value:.5f}")
print(f"  Best params: {lgb_study.best_params}")

best_lgb = lgb.LGBMRegressor(**lgb_study.best_params, random_state=42, verbosity=-1, n_jobs=-1)
best_lgb.fit(X_tr, y_tr)
p8_results['LightGBM (tuned)'] = evaluate(y_hold, best_lgb.predict(X_hold), 'LightGBM (tuned)')

# ── 8b. Tune XGBoost — 100 trials ────────────────────────────────────────────
import xgboost as xgb

def xgb_objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 200, 2000),
        'learning_rate':    trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth':        trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'tree_method': 'hist',
        'verbosity': 0,
        'n_jobs': -1,
    }
    scores = []
    for tr_idx, val_idx in tscv.split(X_tr):
        Xf, Xv = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
        yf, yv = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]
        m = xgb.XGBRegressor(**params, early_stopping_rounds=50)
        m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
        scores.append(np.sqrt(np.mean((yv - m.predict(Xv)) ** 2)))
    return np.mean(scores)

print("\n[8b] Tuning XGBoost — 100 Optuna trials, 5-fold TimeSeriesSplit with early stopping...")
xgb_study = optuna.create_study(direction='minimize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=100, show_progress_bar=True)

print(f"  Best CV RMSE (log): {xgb_study.best_value:.5f}")
print(f"  Best params: {xgb_study.best_params}")

best_xgb = xgb.XGBRegressor(**xgb_study.best_params, random_state=42,
                              tree_method='hist', verbosity=0, n_jobs=-1)
best_xgb.fit(X_tr, y_tr)
p8_results['XGBoost (tuned)'] = evaluate(y_hold, best_xgb.predict(X_hold), 'XGBoost (tuned)')

# ── 8c. Tune CatBoost — 100 trials ───────────────────────────────────────────
def cat_objective(trial):
    params = {
        'iterations':    trial.suggest_int('iterations', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'depth':         trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg':   trial.suggest_float('l2_leaf_reg', 1e-2, 20.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'loss_function': 'RMSE',
        'random_seed': 42,
        'verbose': 0,
    }
    scores = []
    for tr_idx, val_idx in tscv.split(X_tr):
        Xf, Xv = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
        yf, yv = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]
        m = CatBoostRegressor(**params)
        m.fit(Xf, yf, eval_set=(Xv, yv))
        scores.append(np.sqrt(np.mean((yv - m.predict(Xv)) ** 2)))
    return np.mean(scores)

print("\n[8c] Tuning CatBoost — 100 Optuna trials, 5-fold TimeSeriesSplit...")
cat_study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
cat_study.optimize(cat_objective, n_trials=100, show_progress_bar=True)

print(f"  Best CV RMSE (log): {cat_study.best_value:.5f}")
print(f"  Best params: {cat_study.best_params}")

best_cat = CatBoostRegressor(**cat_study.best_params, loss_function='RMSE',
                               random_seed=42, verbose=0)
best_cat.fit(X_tr, y_tr)
p8_results['CatBoost (tuned)'] = evaluate(y_hold, best_cat.predict(X_hold), 'CatBoost (tuned)')

# ── 8d. Pick overall best & retrain on full data ──────────────────────────────
all_p8 = {'CatBoost (Phase7)': cat_ref_metrics, **p8_results}
comparison = pd.DataFrame(all_p8).T.sort_values('rmse')
print("\n" + "="*70)
print("Phase 8 — Tuned model comparison:")
print(comparison[['rmse','mae','r2','mape']].round(4).to_string())

best_name = comparison.index[0]
print(f"\nBest model: {best_name}")

if best_name == 'LightGBM (tuned)':
    final_model = lgb.LGBMRegressor(**lgb_study.best_params, random_state=42, verbosity=-1, n_jobs=-1)
elif best_name == 'XGBoost (tuned)':
    final_model = xgb.XGBRegressor(**xgb_study.best_params, random_state=42, tree_method='hist', verbosity=0, n_jobs=-1)
else:
    final_model = CatBoostRegressor(**cat_study.best_params, loss_function='RMSE', random_seed=42, verbose=0)

print(f"Retraining {best_name} on full Jan–Oct ({len(X_train):,} rows)...")
final_model.fit(X_train, y_train)

val_log_preds = final_model.predict(X_val)
val_preds     = np.expm1(val_log_preds)
print(f"Validation predictions: count={len(val_preds):,}  mean=${val_preds.mean():,.2f}  "
      f"min=${val_preds.min():,.2f}  max=${val_preds.max():,.2f}")

# ── 8e. Optuna convergence plot ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
studies = [('LightGBM', lgb_study), ('XGBoost', xgb_study), ('CatBoost', cat_study)]

for ax, (name, study) in zip(axes, studies):
    vals = [t.value for t in study.trials if t.value is not None]
    best_so_far = np.minimum.accumulate(vals)
    ax.plot(range(1, len(vals)+1), vals, alpha=0.4, color='steelblue', lw=0.8, label='Trial RMSE')
    ax.plot(range(1, len(best_so_far)+1), best_so_far, color='red', lw=2, label='Best so far')
    ax.set_title(f'{name} — Optuna Convergence\nBest RMSE (log)={min(vals):.4f}')
    ax.set_xlabel('Trial'); ax.set_ylabel('CV RMSE (log scale)')
    ax.legend(fontsize=8)

plt.suptitle('Phase 8 — Optuna Convergence (100 trials each, 5-fold TimeSeriesSplit)', fontsize=13)
plt.tight_layout()
plt.savefig('eda_8e_optuna_convergence.png', dpi=130, bbox_inches='tight')
plt.close()
print("Saved eda_8e_optuna_convergence.png")

# ── 8f. Model comparison bar chart ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
models_sorted = comparison.index.tolist()
colors = ['#2ca02c','#1f77b4','#ff7f0e','#9467bd']

rmse_v = [all_p8[m]['rmse'] for m in models_sorted[::-1]]
axes[0].barh(models_sorted[::-1], rmse_v, color=colors[:len(models_sorted)])
axes[0].set_title('RMSE ($) — Lower is Better')
for i, v in enumerate(rmse_v):
    axes[0].text(v+1, i, f'${v:,.0f}', va='center', fontsize=9)

mape_v = [all_p8[m]['mape'] for m in models_sorted[::-1]]
axes[1].barh(models_sorted[::-1], mape_v, color=colors[:len(models_sorted)])
axes[1].set_title('MAPE (%) — Lower is Better')
for i, v in enumerate(mape_v):
    axes[1].text(v+0.05, i, f'{v:.2f}%', va='center', fontsize=9)

r2_v = [all_p8[m]['r2'] for m in models_sorted[::-1]]
axes[2].barh(models_sorted[::-1], r2_v, color=colors[:len(models_sorted)])
axes[2].set_title('R² — Higher is Better')
for i, v in enumerate(r2_v):
    axes[2].text(max(v+0.001, 0.001), i, f'{v:.4f}', va='center', fontsize=9)

plt.suptitle('Phase 8 — Tuned Models vs Phase 7 Reference', fontsize=13)
plt.tight_layout()
plt.savefig('eda_8f_tuned_comparison.png', dpi=130, bbox_inches='tight')
plt.close()
print("Saved eda_8f_tuned_comparison.png")

# ── 8g. Predicted vs actual (best tuned model) ───────────────────────────────
best_hold_pred = final_model.predict(X_hold)
true_hold  = np.expm1(y_hold)
pred_hold  = np.expm1(best_hold_pred)
residuals  = pred_hold - true_hold

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(true_hold, pred_hold, alpha=0.15, s=6, color='steelblue')
mv = max(true_hold.max(), pred_hold.max())
axes[0].plot([0,mv],[0,mv],'r--',lw=1.5,label='Perfect')
axes[0].set_title(f'{best_name}: Predicted vs Actual (Hold-out)')
axes[0].set_xlabel('Actual ($)'); axes[0].set_ylabel('Predicted ($)')
axes[0].legend()

axes[1].hist(residuals, bins=80, color='darkorange', edgecolor='white', lw=0.3)
axes[1].axvline(0, color='red', lw=1.5, linestyle='--')
axes[1].set_title(f'{best_name}: Residuals Distribution')
axes[1].set_xlabel('Residual ($)'); axes[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig('eda_8g_final_diagnostics.png', dpi=130, bbox_inches='tight')
plt.close()
print("Saved eda_8g_final_diagnostics.png")

print(f"\nDiagnostics — mean: ${residuals.mean():,.2f}, std: ${residuals.std():,.2f}")
print(f"Within $200: {(residuals.abs()<200).mean()*100:.1f}%")
print(f"Within $500: {(residuals.abs()<500).mean()*100:.1f}%")

# ── Save results ──────────────────────────────────────────────────────────────
summary = {
    'best_model': best_name,
    'lgb_best_params': lgb_study.best_params,
    'lgb_best_cv_rmse_log': lgb_study.best_value,
    'xgb_best_params': xgb_study.best_params,
    'xgb_best_cv_rmse_log': xgb_study.best_value,
    'cat_best_params': cat_study.best_params,
    'cat_best_cv_rmse_log': cat_study.best_value,
    'phase7_catboost_rmse': cat_ref_metrics['rmse'],
    'all_metrics': {k: {kk: float(vv) for kk,vv in v.items()} for k,v in all_p8.items()},
    'residuals': {
        'mean': float(residuals.mean()),
        'std': float(residuals.std()),
        'pct_within_200': float((residuals.abs()<200).mean()*100),
        'pct_within_500': float((residuals.abs()<500).mean()*100),
    }
}

with open('phase8_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*70)
print(" Phase 8 COMPLETE")
print("="*70)
print(f"Best model     : {best_name}")
print(f"RMSE ($)       : ${all_p8[best_name]['rmse']:,.2f}")
print(f"MAPE           : {all_p8[best_name]['mape']:.2f}%")
print(f"R²             : {all_p8[best_name]['r2']:.4f}")
print(f"Val predictions: {len(val_preds):,} rows")
print("\nSaved phase8_results.json")
print("Saved eda_8e_optuna_convergence.png")
print("Saved eda_8f_tuned_comparison.png")
print("Saved eda_8g_final_diagnostics.png")
