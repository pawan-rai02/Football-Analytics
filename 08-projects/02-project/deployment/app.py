"""
app.py
------
Simple Flask web app that loads the trained Random Forest model
and lets you predict a football match result via a web form.

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model and feature list once at startup
MODEL_DIR    = os.path.join(os.path.dirname(__file__), 'model')
clf          = joblib.load(os.path.join(MODEL_DIR, 'rf_model.pkl'))
FEATURES     = joblib.load(os.path.join(MODEL_DIR, 'features.pkl'))

# Build lists of teams, referees, venues from feature column names
def extract_options(prefix):
    return sorted([
        col.replace(f'{prefix}_', '', 1)
        for col in FEATURES if col.startswith(f'{prefix}_')
    ])

HOME_TEAMS = extract_options('Home')
AWAY_TEAMS = extract_options('Away')
REFEREES   = extract_options('Referee')
VENUES     = extract_options('Venue')
DAYS       = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html',
                           home_teams=HOME_TEAMS,
                           away_teams=AWAY_TEAMS,
                           referees=REFEREES,
                           venues=VENUES,
                           days=DAYS)


@app.route('/predict', methods=['POST'])
def predict():
    # Collect form inputs
    home_team              = request.form.get('home_team')
    away_team              = request.form.get('away_team')
    referee                = request.form.get('referee')
    venue                  = request.form.get('venue')
    day                    = request.form.get('day')
    wk                     = float(request.form.get('week', 20))
    home_rolling_goals     = float(request.form.get('home_rolling_goals', 1.5))
    away_rolling_goals     = float(request.form.get('away_rolling_goals', 1.2))
    home_rolling_xg        = float(request.form.get('home_rolling_xg', 1.5))
    away_rolling_xg        = float(request.form.get('away_rolling_xg', 1.2))

    # Build a single-row DataFrame with all features set to 0.0 (float)
    # Using 0.0 instead of 0 so pandas uses float64 dtype (allows decimal values)
    match = pd.DataFrame(0.0, index=[0], columns=FEATURES)

    # Fill numeric features
    if 'Wk' in FEATURES:
        match.at[0, 'Wk'] = wk
    if 'home_rolling_avg_goals' in FEATURES:
        match.at[0, 'home_rolling_avg_goals'] = home_rolling_goals
    if 'away_rolling_avg_goals' in FEATURES:
        match.at[0, 'away_rolling_avg_goals'] = away_rolling_goals
    if 'home_rolling_avg_xG' in FEATURES:
        match.at[0, 'home_rolling_avg_xG'] = home_rolling_xg
    if 'away_rolling_avg_xG' in FEATURES:
        match.at[0, 'away_rolling_avg_xG'] = away_rolling_xg

    # Fill one-hot features
    home_col = f'Home_{home_team}'
    away_col = f'Away_{away_team}'
    ref_col  = f'Referee_{referee}'
    ven_col  = f'Venue_{venue}'
    day_col  = f'Day_{day}'

    for col in [home_col, away_col, ref_col, ven_col, day_col]:
        if col in match.columns:
            match.at[0, col] = 1

    # Predict
    prediction    = clf.predict(match)[0]
    probabilities = clf.predict_proba(match)[0]
    class_labels  = clf.classes_

    proba_dict = {label: round(prob * 100, 1)
                  for label, prob in zip(class_labels, probabilities)}

    return render_template('result.html',
                           prediction=prediction,
                           proba=proba_dict,
                           home_team=home_team,
                           away_team=away_team)


if __name__ == '__main__':
    app.run(debug=True)
