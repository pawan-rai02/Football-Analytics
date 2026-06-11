# ⚽ Project 1 — FIFA World Cup Final 2022: Argentina vs France Match Analysis

A data visualization project analyzing the **2022 FIFA World Cup Final** (Argentina 3–3 France, Argentina win on penalties) using **StatsBomb open event data**.

---

## 📌 What This Project Does

Builds a single **match report dashboard** combining four visualizations into one figure:

| Panel | Description |
|---|---|
| **Pass Network (Argentina)** | Player positions + pass connections via ConvexHull |
| **Match Stats Table** | Side-by-side xG, shots, passes, pass completion % |
| **xG Flow Chart** | Cumulative xG over time, split by half |
| **Shot Maps (both teams)** | Goal / on-target / off-target shots on a half-pitch |

---

## 📊 Match Stats (from output)

| Stat | Argentina | France |
|---|---|---|
| Goals | 3 | 3 (AET, Argentina win on pens) |
| xG | computed from StatsBomb | computed from StatsBomb |
| Shots | computed | computed |
| Shots on Target | computed | computed |
| Passes | computed | computed |
| Pass Completion % | computed | computed |

> Stats are computed live from StatsBomb event data (match_id: 3869685)

---

## 🛠️ Tools & Libraries

- **StatsBombPy** — free event data (match id: `3869685`)
- **mplsoccer** — `VerticalPitch` for shot maps
- **matplotlib** — multi-panel figure layout using `add_axes`
- **scipy** `ConvexHull` — team shape in pass network
- **pandas / numpy** — data wrangling
- **Pillow (PIL)** — loading team logo images

---

## 🗂️ Files

```
01-project/
├── argentina_france.ipynb   ← Main analysis notebook
└── (uses ../team-logos/ for Argentina & France logos)
```

---

## ▶️ How to Run

1. Activate the `analysis` virtual environment
2. Open `argentina_france.ipynb` in VS Code or Jupyter
3. Select the `Python (analysis)` kernel
4. Run All Cells — StatsBomb data is fetched automatically (no file needed)

> Requires internet connection for the first run to download StatsBomb data.
