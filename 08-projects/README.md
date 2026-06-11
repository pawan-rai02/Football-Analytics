# ⚽ 08-Projects — Football Data Science Projects

Two end-to-end football data projects covering **event data visualization** and **match outcome prediction with machine learning**.

---

## 📁 Projects Overview

### [Project 1 — FIFA World Cup Final 2022: Argentina vs France](./01-project/)

> *Sports Analytics · Data Visualization · Event Data*

A match report dashboard for the **2022 FIFA World Cup Final** built using StatsBomb open event data.

**What it does:**
- Fetches live event-level data (passes, shots, tactics) via `statsbombpy`
- Builds a multi-panel matplotlib figure containing:
  - **Pass networks** for both teams (player positions + pass connections)
  - **Shot maps** on a half-pitch for each team
  - **xG flow chart** showing cumulative xG over 90+ minutes
  - **Match stats table** (xG, shots, shots on target, passes, pass completion %)

**Key facts from the data:**
- Match result: Argentina 3–3 France (Argentina win on penalties)
- Data source: StatsBomb open event data — match ID `3869685`
- Includes all periods (normal time + extra time, excludes penalty shootout)

---

### [Project 2 — La Liga Match Result Predictor](./02-project/)

> *Machine Learning · Classification · Feature Engineering · Flask Deployment*

A Random Forest classifier that predicts football match results (**Home Win / Away Win / Draw**) trained on La Liga seasons 2018–2023.

**What it does:**
- Cleans and engineers features from raw match data (`project2.csv` — 4,560 rows, 12 columns)
- Computes rolling 5-game averages of goals and xG per team
- One-hot encodes team names, referee, venue, and day of week
- Trains a Random Forest with GridSearchCV hyperparameter tuning
- Deploys as a simple Flask web app

**Key metrics from notebook outputs:**

| Model Version | Features Used | Accuracy |
|---|---|---|
| Baseline RF (default params) | Rolling stats + Day | 43.6% |
| Tuned RF (`max_depth=5, n=50`) | Rolling stats + Day | 48.7% |
| **Final RF (`max_depth=20, n=200`)** | **+ Team / Venue / Referee (OHE)** | **51.9%** |

**Class distribution (baseline):**

| Result | Share |
|---|---|
| Home Win | 44.6% |
| Away Win | 30.4% |
| Draw | 25.1% |

> The model beats a naïve "always predict Home Win" baseline (44.6%) by ~7 percentage points.

**Train / Test split:**
- Train: Seasons up to 2022 → 4,024 matches
- Test: Season 2023 → 374 matches

---

## 🛠️ Tools & Technologies Used (Both Projects)

| Category | Tools |
|---|---|
| Data | pandas, numpy |
| Football data | statsbombpy, project2.csv (FBRef/scraped) |
| Visualization | matplotlib, mplsoccer, seaborn |
| Machine Learning | scikit-learn (RandomForestClassifier, GridSearchCV) |
| Spatial analysis | scipy (ConvexHull) |
| Deployment | Flask, joblib |
| Environment | Python 3.14, Jupyter Notebook |

---

## 📄 Resume Section

### Bullet Points (for CV / LinkedIn)

- **Built a match report dashboard** for the 2022 FIFA World Cup Final using StatsBomb event data, visualizing pass networks, shot maps, and xG flow charts with mplsoccer and matplotlib
- **Engineered rolling-window features** (5-game avg goals & xG per team) from 4,500+ La Liga match records across 5 seasons
- **Trained and tuned a Random Forest classifier** to predict football match outcomes (Home Win / Away Win / Draw), achieving **51.9% accuracy** — outperforming the majority-class baseline by ~7%
- **Performed hyperparameter tuning** using 5-fold GridSearchCV across 12 parameter combinations (`n_estimators`, `max_depth`)
- **Deployed the ML model** as a Flask web application with a clean prediction form and probability output
- **Applied one-hot encoding** on high-cardinality categorical features (teams, referees, venues — 255 total features)

---

### Interview Q&A

**Q: Why use Random Forest for this problem?**
> It handles mixed feature types well (numeric rolling stats + one-hot encoded categories), is robust to overfitting with enough trees, and gives probability estimates for all 3 classes — useful for showing confidence in the web app.

**Q: How did you handle class imbalance?**
> The dataset is slightly imbalanced (Home Win 44.6%, Away Win 30.4%, Draw 25.1%), but no resampling was applied. A future improvement would be using `class_weight='balanced'` or SMOTE for the minority Draw class.

**Q: What do the rolling averages capture?**
> A team's form — the average goals scored/conceded and xG created/conceded over the last 5 games. This is a common football analytics feature that captures momentum better than season-long averages.

**Q: How did you validate the model?**
> A time-based train/test split: trained on seasons ≤ 2022 (4,024 matches), tested on season 2023 (374 matches). This avoids data leakage since a match's rolling features are calculated from past games only (`closed='left'` in `.rolling()`).

**Q: What would you do to improve accuracy?**
> Add Elo ratings per team, include player-level features (lineup strength, injuries), use more seasons of data, try XGBoost or LightGBM, and add home/away form separately rather than combined.

**Q: What is xG and why is it useful?**
> Expected Goals (xG) measures the quality of a shot based on historical data — position, angle, assist type, etc. It's a better predictor of future performance than actual goals because it removes luck from finishing.
