# ⚽ Football Analytics

A structured learning repository covering football data science — from **web scraping** and **data visualization** to **machine learning** and **model deployment**. All projects use real football data.

---

## 📂 Repository Structure

| Folder | Topic | Key Libraries |
|---|---|---|
| `01-webscrapping` | Scraping match data from websites | requests, BeautifulSoup, Selenium |
| `02-stats-bomb` | Exploring StatsBomb open event data | statsbombpy, pandas |
| `03-fbref` | Scraping FBRef player/team stats | Selenium, pandas |
| `04-sofascore` | Pulling data from Sofascore | requests, pandas |
| `05-visualizations` | Football-specific plots | mplsoccer, matplotlib, seaborn |
| `06-data-analysis` | EDA and statistics | pandas, numpy, scipy |
| `07-machine-learning` | Classification & regression models | scikit-learn |
| `08-projects` | Full end-to-end projects | All of the above + Flask |

---

## 🏆 Projects (`08-projects`)

### [Project 1 — FIFA World Cup Final 2022: Argentina vs France](./08-projects/01-project/)
Match report dashboard using StatsBomb event data. Includes pass networks, shot maps, xG flow chart, and match stats table for the 2022 World Cup Final (Argentina 3–3 France).

### [Project 2 — Match Result Predictor](./08-projects/02-project/)
Random Forest classifier predicting La Liga match outcomes (Home Win / Away Win / Draw) across 4,500+ matches. Deployed as a Flask web app. Final accuracy: **51.9%** on the 2023 season test set.

---

## 🛠️ Tech Stack

- **Language:** Python 3.14
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn, mplsoccer, plotly
- **Football Data:** statsbombpy, FBRef (via Selenium), Sofascore
- **Machine Learning:** scikit-learn
- **Deployment:** Flask, joblib
- **Scraping:** requests, BeautifulSoup, Selenium, lxml
- **Environment:** Jupyter Notebooks, VS Code, virtualenv (`analysis`)

---

## ⚙️ Setup

```powershell
# Activate the virtual environment
.\analysis\Scripts\activate

# All dependencies are already installed in the 'analysis' venv
# If starting fresh:
pip install -r 08-projects/02-project/deployment/requirements.txt
```

> See individual project READMEs for specific run instructions.
