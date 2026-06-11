"""
train_model.py
--------------
Reads project2.csv, engineers features, trains a Random Forest Classifier,
and saves the model + feature columns to disk so the app can load them.

Run once before starting the web app:
    python train_model.py
"""

import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# ── 1. Load data ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'project2.csv')
df = pd.read_csv(DATA_PATH)

# ── 2. Clean ──────────────────────────────────────────────────────────────────
df = df.dropna(subset=['Score'])
df = df.drop(columns=['Attendance', 'Time'])

# ── 3. Parse goals from Score column ─────────────────────────────────────────
# Score column uses an en-dash (–), try both common separators
try:
    df[['home_goals', 'away_goals']] = (
        df['Score'].str.split('–', expand=True).astype(int)
    )
except Exception:
    df[['home_goals', 'away_goals']] = (
        df['Score'].str.split('-', expand=True).astype(int)
    )

# ── 4. Result label ───────────────────────────────────────────────────────────
def det_res(row):
    if row['home_goals'] > row['away_goals']:
        return 'Home win'
    elif row['home_goals'] < row['away_goals']:
        return 'Away win'
    else:
        return 'Draw'

df['result'] = df.apply(det_res, axis=1)

# ── 5. Date features ──────────────────────────────────────────────────────────
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day_name()
df['season_start'] = df['Date'].apply(
    lambda x: x.year - 1 if x.month < 8 else x.year
)

# ── 6. One-hot encode Day ─────────────────────────────────────────────────────
df = pd.get_dummies(df, columns=['Day'])
df.reset_index(drop=True, inplace=True)

# ── 7. Rolling averages (last 5 games per team) ───────────────────────────────
df['home_rolling_avg_goals'] = None
df['away_rolling_avg_goals'] = None
df['home_rolling_avg_xG']    = None
df['away_rolling_avg_xG']    = None

for team in df.Home.unique():
    team_mask = (df['Home'] == team) | (df['Away'] == team)
    temp = df[team_mask].sort_values('Date')

    temp['goal_val'] = temp.apply(
        lambda r: r['home_goals'] if r['Home'] == team else r['away_goals'], axis=1
    )
    temp['xG_val'] = temp.apply(
        lambda r: r['xG'] if r['Home'] == team else r['xG.1'], axis=1
    )
    rolling_goals = temp['goal_val'].rolling(window=5, closed='left', min_periods=1).mean()
    rolling_xG    = temp['xG_val'].rolling(window=5, closed='left', min_periods=1).mean()

    for idx, row in temp.iterrows():
        if row['Home'] == team:
            df.at[idx, 'home_rolling_avg_goals'] = rolling_goals[idx]
            df.at[idx, 'home_rolling_avg_xG']    = rolling_xG[idx]
        else:
            df.at[idx, 'away_rolling_avg_goals'] = rolling_goals[idx]
            df.at[idx, 'away_rolling_avg_xG']    = rolling_xG[idx]

df = df.dropna(subset=['home_rolling_avg_goals', 'away_rolling_avg_goals',
                        'home_rolling_avg_xG', 'away_rolling_avg_xG'])

# ── 8. One-hot encode team / venue / referee ──────────────────────────────────
df = pd.get_dummies(df, columns=['Home', 'Away', 'Referee', 'Venue'])

# ── 9. Feature list ───────────────────────────────────────────────────────────
drop_cols = ['Date', 'xG', 'xG.1', 'Score', 'result',
             'home_goals', 'away_goals', 'season_start']
features = [c for c in df.columns if c not in drop_cols]

# ── 10. Train / test split ────────────────────────────────────────────────────
train_df = df[df['season_start'] <= 2022]  # Note: season_start already dropped above
# Re-add season_start temporarily for split
df_with_season = df.copy()
df_with_season['season_start'] = df['season_start'] if 'season_start' in df.columns else None

# Redo with season column kept for splitting
df2 = pd.read_csv(DATA_PATH)
df2 = df2.dropna(subset=['Score'])
df2 = df2.drop(columns=['Attendance', 'Time'])
try:
    df2[['home_goals', 'away_goals']] = df2['Score'].str.split('–', expand=True).astype(int)
except Exception:
    df2[['home_goals', 'away_goals']] = df2['Score'].str.split('-', expand=True).astype(int)
df2['result'] = df2.apply(det_res, axis=1)
df2['Date'] = pd.to_datetime(df2['Date'])
df2['Day'] = df2['Date'].dt.day_name()
df2['season_start'] = df2['Date'].apply(lambda x: x.year - 1 if x.month < 8 else x.year)
df2 = pd.get_dummies(df2, columns=['Day'])
df2.reset_index(drop=True, inplace=True)

df2['home_rolling_avg_goals'] = None
df2['away_rolling_avg_goals'] = None
df2['home_rolling_avg_xG']    = None
df2['away_rolling_avg_xG']    = None

for team in df2.Home.unique():
    team_mask = (df2['Home'] == team) | (df2['Away'] == team)
    temp = df2[team_mask].sort_values('Date')
    temp['goal_val'] = temp.apply(lambda r: r['home_goals'] if r['Home'] == team else r['away_goals'], axis=1)
    temp['xG_val']   = temp.apply(lambda r: r['xG'] if r['Home'] == team else r['xG.1'], axis=1)
    rolling_goals = temp['goal_val'].rolling(window=5, closed='left', min_periods=1).mean()
    rolling_xG    = temp['xG_val'].rolling(window=5, closed='left', min_periods=1).mean()
    for idx, row in temp.iterrows():
        if row['Home'] == team:
            df2.at[idx, 'home_rolling_avg_goals'] = rolling_goals[idx]
            df2.at[idx, 'home_rolling_avg_xG']    = rolling_xG[idx]
        else:
            df2.at[idx, 'away_rolling_avg_goals'] = rolling_goals[idx]
            df2.at[idx, 'away_rolling_avg_xG']    = rolling_xG[idx]

df2 = df2.dropna(subset=['home_rolling_avg_goals', 'away_rolling_avg_goals',
                           'home_rolling_avg_xG', 'away_rolling_avg_xG'])
df2 = pd.get_dummies(df2, columns=['Home', 'Away', 'Referee', 'Venue'])

drop_cols = ['Date', 'xG', 'xG.1', 'Score', 'result', 'home_goals', 'away_goals', 'season_start']
features = [c for c in df2.columns if c not in drop_cols]

train_df = df2[df2['season_start'] <= 2022]
x_train  = train_df[features]
y_train  = train_df['result']

# ── 11. Train model ───────────────────────────────────────────────────────────
print("Training Random Forest model...")
clf = RandomForestClassifier(random_state=42, max_depth=20, n_estimators=200)
clf.fit(x_train, y_train)
print("Training complete!")

# ── 12. Save model & feature list ────────────────────────────────────────────
os.makedirs(os.path.join(os.path.dirname(__file__), 'model'), exist_ok=True)
model_path    = os.path.join(os.path.dirname(__file__), 'model', 'rf_model.pkl')
features_path = os.path.join(os.path.dirname(__file__), 'model', 'features.pkl')

joblib.dump(clf, model_path)
joblib.dump(features, features_path)

print(f"Model saved    -> {model_path}")
print(f"Features saved -> {features_path}")
print(f"Number of features: {len(features)}")
