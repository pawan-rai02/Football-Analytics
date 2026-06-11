# ⚽ Project 2 — La Liga Match Result Predictor

A machine learning project that predicts the outcome of a La Liga football match (**Home Win / Away Win / Draw**) using a **Random Forest Classifier**, with a deployed Flask web app for live predictions.

---

## 📊 Dataset

- **Source:** `project2.csv` — La Liga match data (2018–2023)
- **Raw shape:** 4,560 rows × 12 columns
- **After cleaning:** 4,433 rows (dropped rows with missing Score)
- **Columns:** `Wk, Day, Date, Home, Away, xG, xG.1, Score, Attendance, Venue, Referee`

---

## 🔄 Feature Engineering Pipeline

| Step | What was done |
|---|---|
| Drop nulls | Removed rows with missing `Score` (127 rows) |
| Parse score | Split `Score` column into `home_goals` / `away_goals` |
| Create target | `result` → Home win / Away win / Draw |
| Date features | Extracted `Day` (day name), `season_start` (year) |
| One-hot encode | `Day` → 7 binary columns |
| Rolling averages | 5-game rolling mean of goals & xG per team (home & away separately) |
| One-hot encode | `Home`, `Away`, `Referee`, `Venue` → final 255 features |

**Rolling average logic:** For each team, sorted by date, computed a left-closed rolling window (uses only past games, no data leakage).

---

## 🤖 Modelling

- **Algorithm:** `RandomForestClassifier` (scikit-learn)
- **Train set:** Seasons ≤ 2022 → **4,024 matches**
- **Test set:** Season 2023 → **374 matches**

### Results

| Model | Accuracy |
|---|---|
| Baseline RF (default params, rolling stats only) | **43.6%** |
| Tuned RF — `max_depth=5, n_estimators=50` | **48.7%** |
| **Final RF — `max_depth=20, n_estimators=200` + OHE teams** | **51.9%** |
| Naïve baseline (always predict Home Win) | 44.6% |

**Hyperparameter tuning:** `GridSearchCV` with 5-fold CV over `n_estimators ∈ {50,100,200}` and `max_depth ∈ {5,10,15,20}`.

### Class Distribution

| Result | Proportion |
|---|---|
| Home Win | 44.6% |
| Away Win | 30.4% |
| Draw | 25.1% |

---

## 🚀 Deployment

A simple **Flask web app** allows you to predict any match result by selecting teams, referee, venue, day, week number, and rolling averages.

```
deployment/
├── train_model.py     ← Train & save the model (run once)
├── app.py             ← Flask app
├── requirements.txt   ← flask, pandas, scikit-learn, joblib
├── model/
│   ├── rf_model.pkl   ← Saved model
│   └── features.pkl   ← Saved feature list
└── templates/
    ├── index.html     ← Input form
    └── result.html    ← Prediction + probability bars
```

See [deployment/README.md](./deployment/README.md) for step-by-step run instructions.

---

## 🛠️ Tools & Libraries

| Category | Library |
|---|---|
| Data wrangling | pandas, numpy |
| Machine Learning | scikit-learn (RF, GridSearchCV, accuracy_score) |
| Deployment | Flask, joblib |
| Notebook | Jupyter / VS Code |

---

## 📄 Resume Section

### Bullet Points

- **Cleaned and engineered features** from 4,500+ La Liga match records — parsed scorelines, engineered rolling 5-game averages for goals and xG per team
- **Built a 3-class classifier** (Home Win / Away Win / Draw) using Random Forest, achieving **51.9% test accuracy** on the 2023 La Liga season
- **Applied GridSearchCV** with 5-fold cross-validation across 12 hyperparameter combinations, improving accuracy from 43.6% → 51.9%
- **Applied one-hot encoding** to high-cardinality features (teams, referees, venues), expanding the feature space to 255 dimensions
- **Deployed the model** as a Flask web application serving real-time predictions with probability outputs

### Interview Q&A

**Q: What features had the most impact?**
> The one-hot encoded team identity (Home/Away) columns had the most impact — adding them jumped accuracy from 48.7% to 51.9%. Rolling average goals and xG per team are the most meaningful numeric features.

**Q: Why time-based train/test split instead of random?**
> Football data is temporal — using future matches to predict past ones would cause data leakage. Training on ≤2022 and testing on 2023 simulates a real-world deployment scenario.

**Q: What does 51.9% accuracy mean in context?**
> Since there are 3 classes, random guessing gives ~33%. Always predicting the most common class (Home Win) gives 44.6%. The model beats that by 7+ points, showing it has learned signal from the features.

**Q: What would improve the model?**
> Elo ratings per team, player availability/injury data, head-to-head historical records, and separate home/away form features. XGBoost or LightGBM may also outperform Random Forest here.
