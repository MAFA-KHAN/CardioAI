# CardioAI — Master System Guide & Defense Roadmap

This document serves as a unified, single-source summary detailing the mathematical mechanisms, clinical origins, page interactions, and study roadmap for the CardioAI Intelligent Clinical Decision Support System.

---

## 1. The Assessment Engine (Page 1)

The **Assessment Page** is designed for the initial intake and multi-stage evaluation of a patient. It operates through a three-step sequential pipeline:

```
[13 Patient Inputs] ──> [21 Features & StandardScaler] ──> [XGBoost Classifier (prob >= 0.35)]
                                                                     │
                                          ┌──────────────────────────┴──────────────────────────┐
                                          ▼                                                     ▼
                             [A* Search Pathway Planner]                           [KB Forward Chaining Engine]
                 (Dynamic path from risk state to Goal State)             (Fires 24 logical rules to a fixed point)
```

### A. How the Code Processes Inputs
1. **Intake**: The frontend UI collects 13 clinical variables (e.g., age, sex, chest pain type, resting BP, cholesterol, resting ECG, max heart rate, exercise-induced angina, and oldpeak/ST depression).
2. **Feature Alignment**: In `server.py` (Online) or `model.js` (Offline), these 13 variables are converted into the 21 columns expected by the classifier:
   * Categorical values are converted to one-hot columns (e.g., `cp` splits into `cp_typical angina`, `cp_atypical angina`, and `cp_non-anginal`; `asymptomatic` is the reference category represented when all three are $0$).
   * The four engineered variables are computed:
     * `age_risk_group` ($0$ if age $< 40$, $1$ if $< 55$, $2$ if $< 65$, $3$ if $\ge 65$).
     * `bp_chol_interaction` = $\text{trestbps} \times \text{chol}$.
     * `exercise_stress_score` = $\text{exang} + \text{oldpeak}$.
     * `thalch_age_ratio` = $\text{thalch} / (220 - \text{age})$.
3. **Scaling**: The 21 features are standardized using the mean and standard deviation matrices fitted on the training set:
   $$z = \frac{x - \mu}{\sigma}$$
4. **Classification**: The scaled features are evaluated by XGBoost. A clinical screening threshold of **`0.35`** is applied to the output probability. If the probability is $\ge 0.35$, the prediction is set to `1` (Cardiac Risk Detected).

### B. A* Search Care Planner
* The output risk probability determines the initial state: `High Risk` ($\ge 80\%$), `Medium Risk` ($40\% - 80\%$), or `Low Risk` ($< 40\%$).
* Edge paths are generated based on thresholds (e.g., a node for `Blood Pressure Management` is added if resting blood pressure $\ge 140$).
* A* evaluates states using $f(n) = g(n) + h(n)$, prioritizing nodes in a min-heap. Because the heuristic $h(n)$ represents the minimum steps remaining to the goal state, it is admissible ($h(n) \le h^*(n)$), ensuring the optimal, lowest-cost care pathway is found.

### C. Forward Chaining Rule Engine
* The system evaluates **24 medical rules** (R01-R24) across 3 tiers.
* The engine checks if the conditions of a rule are satisfied by the current patient facts (e.g., `hypertension` and `hyperlipidemia` present $\rightarrow$ infer `elevated_cardiovascular_risk`).
* Fired conclusions are added to the fact set, and the loop repeats until no new rules can fire (reaching a logical fixed point).

---

## 2. The R-Simulator Engine (Page 2)

The **Simulator (R-Simulator)** allows clinicians to run real-time virtual intervention analyses (sensitivity analysis) for an active patient profile.

### A. How the Simulator Page Works
1. The user selects a baseline patient profile or enters custom values.
2. The simulator automatically duplicates this baseline profile and generates **four counterfactual scenarios**, modifying one variable at a time to represent clinical treatments:
   * **Scenario 1 (BP Control)**: Sets resting blood pressure `trestbps` to $120$ mmHg (simulating successful antihypertensive medication).
   * **Scenario 2 (Cholesterol Control)**: Sets serum cholesterol `chol` to $180$ mg/dL (simulating statin treatment).
   * **Scenario 3 (Exercise Capacity)**: Increases max heart rate `thalch` by $+10$ bpm (simulating cardiovascular training).
   * **Scenario 4 (ECG Resolution)**: Sets ST depression `oldpeak` to $0.0$ mm (simulating resolved ischemia).

### B. Backend and Local Computation of Prediction values
For each of the four scenarios:
* **Online Mode**: The frontend sends a JSON payload to Flask (`server.py` at `/simulate`). The server rebuilds the 21 features for each scenario, scales them, and runs `model.predict_proba()` to compute a new risk probability.
* **Offline Fallback Mode**: The frontend calls `runOfflineSimulate()` in `model.js`. It performs the same feature alignment and scales the variables. It then runs a local logistic approximation of the XGBoost decision boundaries:
  $$P(\text{disease}) = \frac{1}{1 + e^{-(\sum (w_i \cdot x_i) + b)}}$$
* **Delta Output**: The simulator subtracts the new scenario probability from the patient's baseline probability:
  $$\Delta = P_{\text{baseline}} - P_{\text{scenario}}$$
* **UI Rendering**: The results are returned to the client and updated on the dashboard:
  * **Bar Chart**: A horizontal bar chart renders the risk change, showing clinicians which modifiable risk factor yields the largest drop in disease probability for this specific patient.
  * **Physiological Visualizer**: An animated color-coded human SVG figure dynamically shifts from red (Critical) to green (Low Risk) based on the simulated risk level.

---

## 3. The Clinical KB & Search Engine (Page 3)

The **Clinical KB** page acts as an information retrieval tool to ground the AI's predictions in established clinical guidelines.

### A. TF-IDF & Cosine Similarity NLP Engine
1. **Document Corpus**: Stored in `algorithms.js`, this comprises 10 summaries of professional clinical guidelines (covering hypertension, hyperlipidemia, ECG interpretations, and the A*/KB algorithms).
2. **Indexing**: The documents are processed to build a TF-IDF matrix. Common words are downweighted, while specific medical terms (e.g., "ischemia", "thalassemia") are assigned high weight.
3. **Query Vectorization**: When a user inputs a query, it is converted into a vector in the same word space.
4. **Similarity Metric**: The engine calculates the cosine angle between the query vector $q$ and each document vector $d$:
   $$\text{Cosine Similarity}(q, d) = \frac{q \cdot d}{\|q\| \|d\|}$$
   The document with the highest similarity score (closest to $1.0$) is displayed.

### B. Purpose of the Selected Sources
* **AHA (American Heart Association) & ACC (American College of Cardiology)**: Used to define Stage 2 hypertension ($\ge 140$ mmHg) and age-related baseline cardiovascular risk.
* **NHLBI (National Heart, Lung, and Blood Institute)**: Sourced for hyperlipidemia thresholds ($\ge 240$ mg/dL).
* **Mayo Clinic Guidelines**: Used to define exercise-induced ischemia indicators (ST depression $>2.0$ mm and reduced cardiac reserve).
* *Why Mention These?* In medicine, symbolic reasoning must be grounded in peer-reviewed clinical guidelines to earn the trust of practitioners.

---

## 4. Student Study Roadmap for Project Defense

To justify this project as an advanced, high-quality implementation during your viva, you should focus on the following key areas:

### A. Code Modules to Master

1. **`backend/server.py` (`prepare_features` & `/predict`)**
   * *What to know*: Explain how this function converts the 13 raw user inputs into the 21 columns in the exact sequence expected by the scaler, ensuring consistency between training and deployment.
2. **`frontend/app.js` (`try/catch` fallbacks)**
   * *What to know*: Explain how the connection monitoring works. The script attempts to connect to Flask on port 5000. If a fetch fails, the catch block updates the status badge to `Offline (Edge AI)` and routes the request to local JS handlers.
3. **`frontend/model.js` (Offline Sigmoid Math)**
   * *What to know*: Explain that this file houses the model weights and intercepts data during offline mode, running a sigmoid function to approximate the XGBoost probability boundaries.
4. **`frontend/algorithms.js` (A* and Forward Chaining)**
   * *What to know*: Explain how this file replicates the Python search queues and the 24 logical rules in Javascript, allowing the planning and reasoning components to run offline.

### B. Core Concept Questions

1. **Why is the decision threshold set to 0.35 instead of 0.50?**
   * *Answer*: In clinical screening, false negatives (missing heart disease) are far more dangerous than false positives. Setting the threshold to 0.35 prioritizes **Recall (Sensitivity)**, reaching 93.1% on the test set.
2. **Why choose XGBoost over a Neural Network (MLP)?**
   * *Answer*: Deep learning neural networks require tens of thousands of samples to learn patterns. For a 920-row clinical tabular dataset, XGBoost is more effective because it uses regularized gradient boosted trees, which generalize better without overfitting.
3. **What makes your A* heuristic admissible?**
   * *Answer*: A heuristic is admissible if it never overestimates the cost to the goal ($h(n) \le h^*(n)$). Our heuristic values represent the minimum physical steps required to transition through the clinical pipeline (e.g., Cardiology Consultation is 2 steps from the Goal State), guaranteeing that A* finds the optimal care pathway.
4. **Why use Forward Chaining instead of Backward Chaining?**
   * *Answer*: Forward chaining is a data-driven reasoning method that starts with known facts (patient lab values) and derives new conclusions. Backward chaining is goal-driven (starting with a hypothesis and working backward). Since we start with patient data and want to generate a diagnostic report, forward chaining is the natural choice.
