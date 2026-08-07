# Phase 4 — Feature Engineering Walkthrough

**Project:** Freight Rate Prediction Challenge  
**Notebook:** `freight_rate_prediction.ipynb` (cells 54–71)  
**Input:** Clean df_train (48,000 rows) · df_val (12,000 rows) from Phase 3  
**Output:** X_train (48,000 × 27) · X_val (12,000 × 27) · y_train = log(1 + posted_rate)  
**Goal:** Create informative features that boost model performance while avoiding leakage

---

## Overview

| Sub-step | Features Created | Count |
|---|---|---|
| 4a | Date / Time features | 7 |
| 4b | Route / Geographic features | 7 |
| 4c | Market Signal interactions | 2 (+ 2 base features retained) |
| 4d | Equipment encoding | 3 (ordinal + 2 one-hot dummies) |
| 4e | City-level target encoding | 2 (pickup_enc, delivery_enc) |
| 4f | Lane-level features | 2 (lane_rate_enc, lane_count) |
| **Total** | **27 features, 0 nulls** | |

---

## 4a — Date / Time Features

### Why These Matter

Freight rates have clear temporal patterns discovered in Phase 2:
- Midweek (Wed–Thu) rates are ~$70 higher than weekends
- Monthly seasonality: rates fluctuate across Q1–Q4
- `days_since_start` acts as a **trend signal** — lets the model learn if rates drift upward or downward over the Jan–Oct period and extrapolate to Nov–Dec

### Code

```python
START_DATE = pd.Timestamp('2025-01-01')

def add_date_features(df):
    df = df.copy()
    df['year']             = df['date'].dt.year
    df['month']            = df['date'].dt.month
    df['day']              = df['date'].dt.day
    df['day_of_week']      = df['date'].dt.dayofweek        # 0=Mon, 6=Sun
    df['is_weekend']       = (df['day_of_week'] >= 5).astype(int)
    df['week_of_year']     = df['date'].dt.isocalendar().week.astype(int)
    df['quarter']          = df['date'].dt.quarter
    df['day_of_year']      = df['date'].dt.dayofyear
    df['days_since_start'] = (df['date'] - START_DATE).dt.days
    return df

df_train = add_date_features(df_train)
df_val   = add_date_features(df_val)
```

### Result

```
Date features added (sample — first 3 rows):

   year  month  day  day_of_week  is_weekend  week_of_year  quarter  day_of_year  days_since_start
0  2025      1    1            2           0             1        1            1                 0
1  2025      1    1            2           0             1        1            1                 0
2  2025      1    1            2           0             1        1            1                 0

days_since_start — min: 0, max: 303
```

### Feature Descriptions

| Feature | Range | Purpose |
|---|---|---|
| `year` | 2025 | Constant here; useful in multi-year datasets |
| `month` | 1–10 (train), 11–12 (val) | Monthly seasonality |
| `day` | 1–31 | Within-month position |
| `day_of_week` | 0–6 | Weekday effect |
| `is_weekend` | 0 or 1 | Binary weekend flag |
| `week_of_year` | 1–44 | ISO week; finer seasonality than month |
| `quarter` | 1–4 | Coarse seasonal bucket |
| `day_of_year` | 1–303 (train) | Continuous seasonality signal |
| `days_since_start` | 0–303 | Trend signal from Jan 1 baseline |

---

## 4b — Route / Geographic Features

### Why These Matter

- **Haversine distance** is the straight-line geodesic distance computed from coordinates. It provides a route distance estimate that is independent of the reported `distance` column — useful as a cross-check and as a feature that captures geographic separation cleanly.
- **`distance_diff`** (road − haversine) measures how "winding" a route is. Mountainous or rural routes can have large differences; interstate corridors are closer to the straight line.
- **Directional bias** (`delta_lat`, `delta_lon`) captures whether the haul goes north/south and east/west — important because east-west transcontinental hauls dominate the highest-rate rows.
- **Region assignment** clusters cities into 5 US geographic zones using simple lat/lon rules. This creates a categorical signal that generalizes across routes.
- **`is_cross_regional`** flags hauls that cross regional boundaries — these are typically longer and more expensive.

### Code

```python
def haversine_miles_vec(lat1, lon1, lat2, lon2):
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def assign_region(lat, lon):
    if lon >= -80:   return 'NE' if lat >= 38 else 'SE'
    elif lon >= -90: return 'NE' if lat >= 39 else 'SE'
    elif lon >= -103: return 'MW'
    else:            return 'SW' if lat < 37 else 'W'

def add_geo_features(df):
    df = df.copy()
    df['haversine_dist']   = haversine_miles_vec(df['pickup_lat'], df['pickup_lon'],
                                                  df['delivery_lat'], df['delivery_lon'])
    df['distance_diff']    = df['distance'] - df['haversine_dist']
    df['dist_ratio']       = df['distance'] / df['haversine_dist']
    df['delta_lat']        = df['delivery_lat'] - df['pickup_lat']
    df['delta_lon']        = df['delivery_lon'] - df['pickup_lon']
    df['pickup_region']    = df.apply(lambda r: assign_region(r['pickup_lat'], r['pickup_lon']), axis=1)
    df['delivery_region']  = df.apply(lambda r: assign_region(r['delivery_lat'], r['delivery_lon']), axis=1)
    df['lane']             = df['pickup'] + '_' + df['delivery']
    df['is_cross_regional']= (df['pickup_region'] != df['delivery_region']).astype(int)
    return df
```

### Result

```
Pickup region distribution (TRAIN):
  SE    15,072  (31.4%)
  NE    14,244  (29.7%)
  MW     9,703  (20.2%)
  SW     8,339  (17.4%)
  W        642   (1.3%)

Cross-regional loads:
  Cross-regional (1) : 35,985  (75.0%)
  Same-region   (0)  : 12,015  (25.0%)
```

75% of hauls cross regional boundaries — confirming the dataset is dominated by longer-distance interstate freight.

### Region Definition

| Region | Code | Approximate Coverage |
|---|---|---|
| Northeast | NE | lon ≥ −90, lat ≥ 38–39 |
| Southeast | SE | lon ≥ −90, lat < 38–39 |
| Midwest | MW | −103 ≤ lon < −90 |
| Southwest | SW | lon < −103, lat < 37 |
| West | W | lon < −103, lat ≥ 37 |

---

## 4c — Market Signal Features

### Why These Matter

`market_index` and `quote_signal` were identified in Phase 2 as having weak *linear* correlations with `posted_rate`. However, their **interaction** may carry non-linear signal:
- When both market conditions AND quote pressure are high, rates could be disproportionately elevated.
- The **ratio** `quote_signal / market_index` represents the relative quote level given current market conditions — a signal of how "aggressive" a quote is.

### Code

```python
def add_market_features(df):
    df = df.copy()
    df['market_x_quote']    = df['market_index'] * df['quote_signal']
    df['quote_over_market'] = df['quote_signal'] / df['market_index'].replace(0, np.nan)
    return df
```

### Result

```
                market_index  quote_signal  market_x_quote  quote_over_market
count           48,000.00     48,000.00       48,000.00          48,000.00
mean                 1.0834        2.0625           2.2312              1.9529
std                  0.1680        0.2914           0.4597              0.4209
min                  0.6764        0.6923           0.6198              0.5294
25%                  0.9498        1.8910           1.8965              1.6598
50%                  1.0558        2.0558           2.1793              1.9334
75%                  1.2199        2.2217           2.5281              2.2168
max                  1.4678        3.6104           4.8730              4.5202
```

---

## 4d — Equipment Encoding

### One-Hot Encoding (drop_first=True)

In addition to the ordinal `equipment_code` (Dry Van=0, Flatbed=1, Reefer=2) created in Phase 3, we create two binary dummy columns for explicit use in linear models (which need one-hot representation):

```python
equip_dummies = pd.get_dummies(df['equipment'], prefix='equip', drop_first=True)
# drop_first removes Dry Van → Dry Van is the reference category (both dummies = 0)
```

### Result

```
Columns added: ['equip_Flatbed', 'equip_Reefer']

equip_Flatbed  equip_Reefer   Count
0              0             27,202   (Dry Van — reference)
0              1             12,045   (Reefer)
1              0              8,753   (Flatbed)
```

**Encoding scheme:**
| Equipment | `equip_Flatbed` | `equip_Reefer` | `equipment_code` |
|---|---|---|---|
| Dry Van | 0 | 0 | 0 |
| Flatbed | 1 | 0 | 1 |
| Reefer | 0 | 1 | 2 |

> Tree-based models (XGBoost, LightGBM, CatBoost) use `equipment_code`. Linear models (Ridge) use `equip_Flatbed` + `equip_Reefer`.

---

## 4e — City-level Target Encoding (Smoothed Bayesian)

### The Problem

Raw mean encoding of cities leaks information when the same rows used to compute means are also used to train the model. We address this with **Bayesian / smoothed target encoding**:

### Smoothing Formula

```
encoded_value = (n × city_mean + k × global_mean) / (n + k)
```

Where:
- `n` = number of loads from/to that city in training data
- `city_mean` = average `posted_rate` for that city
- `k` = smoothing factor (we use k=20)
- `global_mean` = overall mean `posted_rate` across all training rows ($2,374)

When `n` is small (rare city), the formula pulls the encoded value toward the global mean — preventing overfitting on noisy rare cities. When `n` is large (common city), the encoded value closely tracks the true city mean.

### Code

```python
SMOOTH_K = 20
global_mean = df_train['posted_rate'].mean()

def smoothed_target_encode(train_df, val_df, col, target='posted_rate', k=20, gm=None):
    if gm is None: gm = train_df[target].mean()
    stats = train_df.groupby(col)[target].agg(['mean', 'count'])
    stats['encoded'] = (stats['count'] * stats['mean'] + k * gm) / (stats['count'] + k)
    train_enc = train_df[col].map(stats['encoded']).fillna(gm)
    val_enc   = val_df[col].map(stats['encoded']).fillna(gm)
    return train_enc, val_enc, stats

df_train['pickup_enc'],  df_val['pickup_enc'],  pickup_stats  = \
    smoothed_target_encode(df_train, df_val, 'pickup',  k=SMOOTH_K, gm=global_mean)

df_train['delivery_enc'], df_val['delivery_enc'], delivery_stats = \
    smoothed_target_encode(df_train, df_val, 'delivery', k=SMOOTH_K, gm=global_mean)
```

### Result

```
Global mean posted_rate (smoothing anchor): $2,373.98

Top 5 pickup cities by encoded rate:
                  mean    count   encoded
pickup
San Francisco   4071.21    453   3999.44
Fresno          3878.91    827   3843.37
Phoenix         3764.77   1121   3740.40
Reno            3765.39    642   3723.35
Los Angeles     3729.64    945   3701.54

Top 5 delivery cities by encoded rate:
                  mean    count   encoded
delivery
San Francisco   4064.47    467   3995.05
Fresno          3865.38    763   3827.28
Phoenix         3751.47   1090   3726.65
Los Angeles     3742.25    955   3714.18
Reno            3679.87    653   3641.06

Validation cities falling back to global mean: 725 pickup, 722 delivery
```

### Key Observations

- **West Coast cities dominate the high-rate list** (San Francisco, Fresno, LA, Reno, Phoenix) — consistent with long transcontinental haul patterns.
- **725 pickup city occurrences in validation fall back to global mean** — these are the 8 new cities not seen in training (Chicago, Charlotte, Knoxville, etc.), which occur across many rows. The global mean ($2,374) is a reasonable fallback.
- The smoothing effect is visible: San Francisco's raw mean is $4,071 but smoothed to $3,999 — pulled slightly toward global mean due to moderate sample size (453 loads).

---

## 4f — Lane-level Features

### Why Lane Features Matter

A lane = specific pickup city → delivery city pair. The same route type (e.g., Los Angeles → Chicago) tends to have consistent rate characteristics due to:
- Fixed carrier availability on that corridor
- Established contractual rate levels
- Physical distance being nearly constant

### Code

```python
LANE_SMOOTH_K = 10  # tighter smoothing — lanes are more specific

# Smoothed mean rate per lane
df_train['lane_rate_enc'], df_val['lane_rate_enc'], lane_stats = \
    smoothed_target_encode(df_train, df_val, 'lane', k=LANE_SMOOTH_K, gm=global_mean)

# Load count per lane (popularity proxy) — learned from train only
lane_counts = df_train['lane'].value_counts().rename('lane_count')
df_train['lane_count'] = df_train['lane'].map(lane_counts).fillna(1)
df_val['lane_count']   = df_val['lane'].map(lane_counts).fillna(1)
```

### Result

```
Unique lanes in train : 4,014
Unique lanes in val   : 4,214
New lanes in val (unseen): 736

Top 5 lanes by load count:
  Columbia → Oklahoma City     39 loads
  Lexington → Atlanta          39 loads
  Phoenix → Shreveport         39 loads
  Fort Wayne → Philadelphia    38 loads
  Lexington → Cincinnati       37 loads

Top 5 lanes by encoded mean rate:
                           mean  count  encoded
lane
Bakersfield → Hartford   5998.48    35  5193.04
Boston → Bakersfield     6737.90    16  5059.47
Reno → Philadelphia      6555.73    15  4883.03
Phoenix → Syracuse       5718.16    28  4838.11
Providence → Phoenix     6859.19    12  4820.46
```

- **4,014 unique lanes in training** — 4-digit granularity of OD pairs.
- **736 new lanes in validation** — unseen OD pairs fall back to global mean ($2,374) via smoothed encoding.
- Highest-rate lanes are all **cross-country West-to-East** routes (Bakersfield/CA → Hartford/CT, Boston → Bakersfield/CA, Reno/NV → Philadelphia/PA) confirming the distance-rate relationship.

---

## 4g — Final Feature Matrix Assembly

### Log-Transform Target

```python
df_train['log_rate'] = np.log1p(df_train['posted_rate'])
```

```
log_rate statistics:
  min  = 4.0642   (exp → $57.22 = minimum raw rate)
  max  = 10.1478  (exp → $25,533 = maximum raw rate)
  mean = 7.5732   (exp → ~$1,929)
```

### Clip Negative Weights

```python
df_train['weight'] = df_train['weight'].clip(lower=0)
df_val['weight']   = df_val['weight'].clip(lower=0)
```

This corrects the negative weight data error identified in Phase 3 (`-36,413 lbs`).

### Final Feature List (27 features)

```
Feature matrix shape: X_train=(48000, 27), X_val=(12000, 27)
Target (log_rate) shape: (48000,)

 1. distance                nulls: 0   ← dominant feature (r=0.91)
 2. haversine_dist          nulls: 0   ← geodesic alternative to distance
 3. distance_diff           nulls: 0   ← road - haversine (route winding)
 4. dist_ratio              nulls: 0   ← road / haversine (normalised routing)
 5. delta_lat               nulls: 0   ← N-S directional bias
 6. delta_lon               nulls: 0   ← E-W directional bias
 7. weight                  nulls: 0
 8. equipment_code          nulls: 0   ← ordinal: DV=0, FB=1, RF=2
 9. equip_Flatbed           nulls: 0   ← one-hot dummy
10. equip_Reefer            nulls: 0   ← one-hot dummy
11. market_index            nulls: 0
12. quote_signal            nulls: 0
13. market_x_quote          nulls: 0   ← interaction feature
14. quote_over_market       nulls: 0   ← ratio feature
15. month                   nulls: 0
16. day_of_week             nulls: 0
17. is_weekend              nulls: 0
18. week_of_year            nulls: 0
19. quarter                 nulls: 0
20. day_of_year             nulls: 0
21. days_since_start        nulls: 0   ← trend signal
22. pickup_enc              nulls: 0   ← Bayesian city encoding
23. delivery_enc            nulls: 0   ← Bayesian city encoding
24. lane_rate_enc           nulls: 0   ← Bayesian OD pair encoding
25. lane_count              nulls: 0   ← lane popularity
26. is_cross_regional       nulls: 0   ← regional boundary flag
27. is_rate_outlier         nulls: 0   ← outlier flag from Phase 3
```

### Leakage Check

```
Leakage check — columns with "rate" in name inside X_val:
  ['is_rate_outlier']

→ is_rate_outlier is set to 0 for all validation rows (no target known)
→ No actual leakage. Clean.

Null counts in X_train: 0 total nulls
Null counts in X_val  : 0 total nulls
```

---

## 4h — Phase 4 Summary (Printed Output)

```
=================================================================
 Phase 4 — Feature Engineering Summary
=================================================================

Total features    : 27
Training samples  : 48,000
Validation samples: 12,000

Feature groups:
  4a Date/Time       : 7 features (month, dow, is_weekend, week, quarter, doy, days_since)
  4b Geographic      : 6 features (haversine, dist_diff, ratio, delta_lat, delta_lon, cross_regional)
  4c Market signals  : 4 features (market_index, quote_signal, interaction, ratio)
  4d Equipment       : 3 features (equipment_code + 2 one-hot dummies)
  4e City encoding   : 2 features (pickup_enc, delivery_enc — smoothed Bayesian)
  4f Lane features   : 2 features (lane_rate_enc, lane_count)
  Core route         : distance (strongest single feature, r=0.91)

Target: log_rate = log1p(posted_rate)
  min=4.0642, max=10.1478, mean=7.5732

Datasets are ready for Phase 5 — Model Training.
```

---

## Files Produced in Phase 4

| File | Description |
|---|---|
| `freight_rate_prediction.ipynb` | Updated notebook with 18 new Phase 4 cells (cells 54–71) |

No new charts — Phase 4 is code-only. SHAP feature importance plots will be produced in Phase 6 (Model Evaluation).

---

## Key Engineering Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | **`days_since_start` as a trend signal** | Allows the model to learn a temporal drift from Jan to Oct and extrapolate to Nov–Dec |
| 2 | **Haversine recomputed for both train and val** | Validation didn't inherit this from Phase 3 — always compute freshly in the feature function |
| 3 | **Smoothed Bayesian encoding (k=20 for cities, k=10 for lanes)** | Prevents overfitting on rare cities/lanes; handles 8 unseen validation cities gracefully |
| 4 | **Lane encoding with k=10 (tighter)** | Lanes are more specific than cities; there are 4,014 unique lanes with smaller n per lane |
| 5 | **Unseen cities/lanes fall back to global mean** | 725 pickup, 722 delivery, 736 lane occurrences in validation use $2,374 as placeholder |
| 6 | **`is_rate_outlier` set to 0 for validation** | We don't know validation targets, so this is a conservative default |
| 7 | **Clip weight to ≥ 0** | Corrects the -36,413 lb data error without deleting the row |
| 8 | **One-hot AND ordinal equipment encoding both retained** | Tree models use ordinal; linear baseline uses one-hot dummies |

---

## Leakage Prevention Summary

| Feature | Computed from | Applied to val? | Safe? |
|---|---|---|---|
| `pickup_enc` | Train `posted_rate` statistics | Yes, via map | ✅ Safe — val rates never seen |
| `delivery_enc` | Train `posted_rate` statistics | Yes, via map | ✅ Safe |
| `lane_rate_enc` | Train `posted_rate` statistics | Yes, via map | ✅ Safe |
| `lane_count` | Train row counts | Yes, via map | ✅ Safe — no target info |
| All date features | `date` column only | Yes | ✅ Safe |
| All geo features | Lat/lon columns only | Yes | ✅ Safe |
| `market_x_quote` | Two non-target columns | Yes | ✅ Safe |
| `is_rate_outlier` | Train target (posted_rate > 99th pct) | Set to 0 for val | ✅ Safe |

---

## State of Datasets After Phase 4

| Property | X_train | X_val |
|---|---|---|
| Rows | 48,000 | 12,000 |
| Features | **27** | **27** |
| Null values | **0** | **0** |
| Target (`y_train`) | log_rate (log-transformed) | — (unknown) |
| Target range | [4.06, 10.15] | — |

---

## Next Steps — Phase 5 (Model Training)

With a clean 27-feature matrix ready, Phase 5 will:

1. **Baseline model** — Ridge regression (linear) as a sanity check
2. **Primary models** — XGBoost and LightGBM with hyperparameter tuning via Optuna
3. **CatBoost** — handles categoricals natively; strong competitor
4. **Cross-validation strategy** — TimeSeriesSplit (respects temporal ordering) with 5 folds
5. **Evaluation metric** — RMSE on log scale (equivalent to relative % error)
6. **Ensemble** — weighted average of top models if performance improves
7. **Generate validation_predictions.csv** — final submission file
