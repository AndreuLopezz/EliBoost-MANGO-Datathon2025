# EliBoost — MANGO Demand Forecasting (Datathon FME 2025)

Project developed for **Datathon FME 2025** under the **MANGO challenge**: **predicting clothing demand** to support production planning.  
This repository contains our solution implemented in **Jupyter Notebooks** (100% of the repo).

---

## Inspiration

Our main inspiration was the **VAR business metric**. The moment we realized that **lost sales are dramatically more costly than excess inventory**, the entire modeling strategy changed. Instead of predicting an average or median demand, we designed a model that **intentionally over-targets demand** to avoid stockouts.

A second key source of inspiration came later: the **correlation map**. It clearly revealed that only a handful of numeric variables had real predictive power compared to the high-dimensional embeddings. This insight guided us to **simplify rather than complicate**, and ultimately made the model far more stable.

---

## What it does

The system is a **high-confidence production forecast model** focused on **maximizing VAR**.

- Instead of predicting the “expected” demand, it predicts the **82nd percentile (P82)** — a deliberately conservative estimate that minimizes under-production risk.
- The model returns **one recommended production quantity per product (ID)**.
- This recommendation represents a **strategic balance** between uncertainty and business risk.

---

## How we built it

### 1) Smart, Business-Driven Features
We engineered features that contextualize each product within its competitive and historical environment:

- **`price_vs_trend`**: positions price relative to past demand patterns  
- **`category_scale`**: captures category-level seasonality and size effects  
- **last-season contextual demand** (category and family)

These features ultimately contributed **more predictive power** than any single embedding component.

### 2) Embedding PCA Compression
We combined and reduced text/image embeddings (≈ **200–300 dimensions**) using **PCA** down to **64 components**.  
This prevented embedding noise from overwhelming simpler, more predictive numeric variables.

### 3) Log-Transform of the Target
Weekly demand is extremely skewed. Applying a **log-transform**:

- stabilized variance,
- helped the quantile model focus on relative differences,
- produced more consistent forecasts.

### 4) Quantile LightGBM Model (α = 0.82)
Final production recommendations come from a **LightGBM Quantile Regressor** with:

- **objective:** quantile regression  
- **alpha:** **0.82**

This allowed us to directly aim at the business goal: a **high-confidence upper bound** on demand aligned with VAR.

---

## Challenges we ran into

### Embeddings vs. Numeric Features
Balancing ~10 interpretable numeric variables (e.g., price, num_stores, num_sizes, last-season demand) against 200+ embedding features was the biggest challenge.

- Without PCA, embeddings dominated training and degraded behavior.
- With PCA(64), embeddings became useful signal instead of noise.

### Finding the Right Quantile
Choosing the quantile wasn’t trivial:

- too low → stockouts  
- too high → overproduction  

We used **actual VAR behavior on a validation season** to converge on **P82**.

### The Correlation Map Realization (Turning Point)
The correlation map showed:

- **price** → moderate negative correlation  
- **num_stores**, **num_sizes** → moderate positive correlation  
- **category_demand_last_season** → extremely strong and consistent correlation  

This revealed two key truths:

1. We already had most meaningful signal in the dataset.
2. Many additional tested features were **pure noise**, which is why “more complex” models often got worse.

This drove simplification:

- keep core numeric features,
- keep reduced embeddings,
- avoid feature bloat without clear statistical justification.

---

## Accomplishments that we’re proud of

### We modeled the business metric, not the technical metric
Rather than optimizing MSE (a weak proxy for VAR), we optimized for the **quantile** that improves business value.

### The log-transform + PCA + quantile trio
This combination proved extremely stable:

- log-transform → smooths extreme volatility  
- PCA → reduces embedding noise  
- quantile regression → directly aligns with VAR  

### A simple, robust, justifiable model
With limited signal and high noise, simplicity won. We ended with a model that is:

- fast  
- interpretable  
- reproducible  
- strategically aligned with business risk  

---

## What we learned

- **Always model the real business objective.** Technical metrics are proxies; here, MSE was misleading and VAR mattered.
- **Log-transforming demand is non-negotiable.** Raw demand skew makes training unstable without it.
- **High-quality features beat raw embeddings.** A few strong contextual features outperformed hundreds of embedding dimensions.
- **Complexity can be counterproductive.** Many “fancier” experiments added noise, not signal.

---

## What’s next for EliBoost v2 (FT for MANGO)

- **Fine-tune the quantile:** test 0.81, 0.83, etc. to validate whether 0.82 is optimal.
- **Hyperparameter tuning:** use **Optuna** to find the best LightGBM settings.
- **More robust validation:** train with **GroupKFold**, grouped by `id_season`, to generalize better to future seasons.

---

## Tech Stack

- Python + Jupyter Notebooks
- LightGBM (Quantile Regression)
- PCA (dimensionality reduction)
- Standard ML tooling (NumPy / Pandas / Scikit-learn)

---

## Acknowledgements

- **Datathon FME 2025** organizers
- **MANGO** for hosting the challenge and providing the dataset