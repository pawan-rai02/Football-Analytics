# ⚽ Football Match Result Predictor — Deployment

This is a simple web app that uses a **Random Forest** machine learning model to predict the result of a football match (Home Win / Away Win / Draw), trained on La Liga data.

---

## 📁 Folder Structure

```
08-projects/02-project/
│
├── project2.csv                  ← Raw match data (used for training)
├── match_predictions.ipynb       ← Original analysis notebook
│
└── deployment/
    ├── train_model.py            ← Step 1: Train & save the model
    ├── app.py                    ← Step 2: Run the web app
    ├── requirements.txt          ← Python packages needed
    ├── model/
    │   ├── rf_model.pkl          ← Saved trained model (auto-generated)
    │   └── features.pkl          ← Saved feature list  (auto-generated)
    └── templates/
        ├── index.html            ← Prediction form page
        └── result.html           ← Result display page
```

---

## 🚀 How to Run — Step by Step

### Step 1 — Make sure your virtual environment is active

Open a terminal inside the project root:
```
d:\Pawan\projects\Football-Analytics\
```

Activate the `analysis` virtual environment:
```powershell
.\analysis\Scripts\activate
```

You should now see `(analysis)` at the start of your terminal prompt.

---

### Step 2 — Install required packages (first time only)

```powershell
pip install -r 08-projects\02-project\deployment\requirements.txt
```

---

### Step 3 — Train and save the model

This reads `project2.csv`, processes the data, trains the Random Forest model, and saves it to the `model/` folder.

```powershell
python 08-projects\02-project\deployment\train_model.py
```

**Expected output:**
```
Training Random Forest model...
Training complete!
Model saved    -> ...deployment\model\rf_model.pkl
Features saved -> ...deployment\model\features.pkl
Number of features: 300+
```

> ✅ You only need to run this **once**. After that, the model is saved and the app loads it directly.

---

### Step 4 — Start the web app

```powershell
python 08-projects\02-project\deployment\app.py
```

**Expected output:**
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

---

### Step 5 — Open in your browser

Go to: **http://127.0.0.1:5000**

You will see a form where you can:
1. Select **Home Team** and **Away Team**
2. Select **Referee** and **Venue**
3. Select the **Day of the week**
4. Enter the **Week number** (matchweek)
5. Enter **rolling average goals and xG** from the last 5 games for each team
6. Click **Predict Result** ⚡

The result page will show:
- The predicted outcome (Home Win / Away Win / Draw)
- Probability bars for each outcome

---

## 🛑 How to Stop the App

Press `Ctrl + C` in the terminal where the app is running.

---

## ❓ Common Issues

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: flask` | Run `pip install flask` or activate the `analysis` venv first |
| `FileNotFoundError: rf_model.pkl` | Run `train_model.py` first (Step 3) |
| `project2.csv not found` | Make sure you run the script from the project root directory |
| Port 5000 already in use | Change `app.run(port=5001)` in `app.py` |

---

## 🧠 About the Model

- **Algorithm:** Random Forest Classifier (scikit-learn)
- **Best params:** `max_depth=20`, `n_estimators=200`
- **Features:** Week number, rolling avg goals (last 5), rolling avg xG (last 5), day of week (one-hot), home team, away team, referee, venue
- **Target:** Match result → `Home win`, `Away win`, `Draw`
- **Training data:** La Liga seasons up to 2022
- **Test data:** La Liga season 2023
