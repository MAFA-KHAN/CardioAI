# CardioAI — Technical & Clinical Justifications

This document provides the clinical and technical rationale behind the architectural choices, algorithm selections, and parameter thresholds implemented in CardioAI.

---

## 1. Machine Learning Model: XGBoost
* **Decision**: We chose **XGBoost (Extreme Gradient Boosting)** over Logistic Regression and Random Forests.
* **Technical Justification**: Clinical tabular datasets are characterized by non-linear relationships and high-dimensional interactions (e.g., how blood pressure interacts with cholesterol at different ages). XGBoost handles tabular data exceptionally well by building sequential decision trees that optimize residuals, resulting in a superior ROC-AUC ($0.898$).
* **Clinical Justification**: XGBoost handles missing values naturally and is robust to outliers, which is common in clinical datasets due to recording variability across different hospitals.

---

## 2. Decision Threshold: Youden's J of 0.35
* **Decision**: The probability classification boundary was set to **`0.35`** instead of the standard `0.50` default.
* **Technical Justification**: Youden's J statistic ($J = \text{Sensitivity} + \text{Specificity} - 1$) was calculated on the validation set, locating the optimal trade-off point that balances detection rates while maintaining specificity.
* **Clinical Justification**: In cardiac screening, a **False Negative** (failing to diagnose a patient who actually has coronary artery disease) is far more dangerous than a **False Positive** (referring a healthy patient for further screening). Lowering the threshold to `0.35` increases clinical sensitivity (recall) to **`0.931` (93.1%)**, ensuring that nearly all patients at risk are safely routed into the care pathway.

---

## 3. Care Pathway Planner: A* Search
* **Decision**: We implemented the **A* Search algorithm** for pathway routing on a dynamic graph.
* **Technical Justification**: Graph pathfinding algorithms like Dijkstra or Breadth-First Search do not utilize heuristic estimates, leading to broader node exploration. By utilizing an **admissible heuristic** ($h(n) \le h^*(n)$), A* guarantees finding the mathematical minimum-cost pathway while exploring fewer nodes, rendering it highly efficient.
* **Clinical Justification**: Hospitals operate under strict resource constraints. Mapped to step costs (where invasive or expensive tests have higher weights), A* ensures the patient is routed through the most clinical-resource-efficient pathway to reach the treatment goal state, preventing redundant diagnostics.

---

## 4. Reasoning Engine: Forward-Chaining Knowledge Base
* **Decision**: We selected a **Forward-Chaining inference engine** for the expert rule base.
* **Technical Justification**: Forward chaining starts from a set of known facts (symptoms, lab results, ML predictions) and applies IF-THEN rules to derive new conclusions (data-driven). Backward chaining starts from a hypothesis and works backward to see if data supports it (goal-driven). Since our objective is to generate a comprehensive diagnostic report from a set of baseline measurements, forward chaining is the natural fit.
* **Clinical Justification**: Forward chaining mimics the cognitive workflow of a clinician during an initial examination, starting with general observations and chaining them to formulate diagnoses.

---

## 5. Medical Search: TF-IDF with Cosine Similarity
* **Decision**: We built a client-side **TF-IDF + Cosine Similarity search engine** for clinical guideline retrieval.
* **Technical Justification**: Large Language Models (LLMs) require heavy runtime resources, introduce licensing costs, and are prone to **hallucinations**, which is unacceptable in medical applications. TF-IDF provides a deterministic, transparent, and lightweight indexing mechanism that runs instantly in standard browser runtimes without external network requests.
* **Clinical Justification**: Search results are fully grounded in a curated, verified corpus of ACC/AHA guidelines, ensuring 100% factual accuracy and complete explainability.
