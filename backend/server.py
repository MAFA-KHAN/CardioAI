from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import heapq
import traceback

app = Flask(__name__)
CORS(app)

MODEL_PATH = "models/heart_disease_model.pkl"
SCALER_PATH = "models/heart_scaler.pkl"

model = None
scaler = None

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[OK] Real XGBoost models loaded successfully.")
except Exception as e:
    print(f"[WARN] Model loading failed: {e}")
    print("[INFO] Running in Demo mode with fallback model.")

    class DemoModel:
        def predict_proba(self, X):
            arr = np.array(X)
            val = float(arr[0,0]) * 0.1 + float(arr[0,5]) * 0.3
            prob = np.clip(1 / (1 + np.exp(-val)), 0.05, 0.95)
            return [[1 - prob, prob]]
        def predict(self, X):
            return [1 if self.predict_proba(X)[0][1] > 0.35 else 0]

    class DemoScaler:
        def transform(self, X):
            return np.array(X) * 0.1

    model = DemoModel()
    scaler = DemoScaler()

# ─────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (matches phase_1.ipynb pipeline exactly)
# ─────────────────────────────────────────────────────────────────
def build_feature_df(data):
    age = float(data.get('age', 50))
    sex = float(data.get('sex', 1))
    cp_choice = float(data.get('cp', 4))
    trestbps = float(data.get('trestbps', 130))
    chol = float(data.get('chol', 200))
    fbs = float(data.get('fbs', 0))
    thalch = float(data.get('thalch', 150))
    exang = float(data.get('exang', 0))
    oldpeak = float(data.get('oldpeak', 1.0))
    slope_choice = float(data.get('slope', 2))
    thal_choice = float(data.get('thal', 1))
    restecg_choice = float(data.get('restecg', 0))

    cp_typical = 1 if cp_choice == 1 else 0
    cp_atypical = 1 if cp_choice == 2 else 0
    cp_non_anginal = 1 if cp_choice == 3 else 0

    restecg_normal = 1 if restecg_choice == 0 else 0
    restecg_st = 1 if restecg_choice == 1 else 0

    slope_flat = 1 if slope_choice == 2 else 0
    slope_up = 1 if slope_choice == 1 else 0

    thal_normal = 1 if thal_choice == 1 else 0
    thal_rev = 1 if thal_choice == 3 else 0

    age_group = 0 if age < 40 else 1 if age < 55 else 2 if age < 65 else 3
    bp_chol = trestbps * chol
    ex_stress = exang + oldpeak
    pred_hr = 220 - age
    hr_ratio = thalch / pred_hr if pred_hr != 0 else 0

    return pd.DataFrame({
        'age': [age], 'sex': [sex], 'trestbps': [trestbps], 'chol': [chol], 'fbs': [fbs],
        'thalch': [thalch], 'exang': [exang], 'oldpeak': [oldpeak],
        'cp_atypical angina': [cp_atypical], 'cp_non-anginal': [cp_non_anginal],
        'cp_typical angina': [cp_typical], 'restecg_normal': [restecg_normal],
        'restecg_st-t abnormality': [restecg_st], 'slope_flat': [slope_flat],
        'slope_upsloping': [slope_up], 'thal_normal': [thal_normal],
        'thal_reversable defect': [thal_rev], 'age_risk_group': [age_group],
        'bp_chol_interaction': [bp_chol], 'exercise_stress_score': [ex_stress],
        'thalch_age_ratio': [hr_ratio]
    })

# ─────────────────────────────────────────────────────────────────
# RISK DETERMINATION
# ─────────────────────────────────────────────────────────────────
def determine_risk(prob):
    if prob >= 0.80: return "High Risk"
    elif prob >= 0.40: return "Medium Risk"
    else: return "Low Risk"

# ─────────────────────────────────────────────────────────────────
# A* SEARCH ENGINE (from phase_2_(2).ipynb)
# ─────────────────────────────────────────────────────────────────
heuristic = {
    'High Risk': 5, 'Medium Risk': 3, 'Low Risk': 1,
    "Blood Pressure Management": 4, "Cholesterol Management": 4,
    "ECG Evaluation": 3, "Stress Evaluation": 3,
    "Cardiology Consultation": 2, "Risk Stratification": 1,
    "Treatment Planning": 1, "Goal State": 0
}

def a_star_search(graph, start, goal):
    queue = []
    heapq.heappush(queue, (0, start, [start], 0))
    visited = set()
    while queue:
        f, node, path, g = heapq.heappop(queue)
        if node == goal:
            return path, g
        if node in visited:
            continue
        visited.add(node)
        for neighbor, cost in graph.get(node, {}).items():
            if neighbor in heuristic:
                new_g = g + cost
                new_f = new_g + heuristic.get(neighbor, 2)
                heapq.heappush(queue, (new_f, neighbor, path + [neighbor], new_g))
    return [start, goal], 1

def build_medical_graph(risk_state, trestbps, chol, exang, oldpeak, age):
    g = {risk_state: {}}
    if trestbps >= 140: g[risk_state]["Blood Pressure Management"] = 2
    if chol >= 240: g[risk_state]["Cholesterol Management"] = 2
    if exang == 1: g[risk_state]["ECG Evaluation"] = 1
    if oldpeak > 2: g[risk_state]["Stress Evaluation"] = 1
    if age >= 60 or risk_state == "High Risk": g[risk_state]["Cardiology Consultation"] = 1
    if risk_state == "Low Risk" and not g[risk_state]:
        g[risk_state]["Goal State"] = 1
    g.update({
        "Blood Pressure Management": {"Risk Stratification": 2},
        "Cholesterol Management": {"Risk Stratification": 2},
        "ECG Evaluation": {"Cardiology Consultation": 1},
        "Stress Evaluation": {"Cardiology Consultation": 1},
        "Cardiology Consultation": {"Risk Stratification": 1},
        "Risk Stratification": {"Treatment Planning": 1},
        "Treatment Planning": {"Goal State": 1},
        "Goal State": {}
    })
    return g

# ─────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE — Forward Chaining (from phase_2_(2).ipynb)
# ─────────────────────────────────────────────────────────────────
knowledge_base = [
    {"id":"R01","if":["elderly_patient"],"then":"age_related_cardiac_risk","source":"AHA"},
    {"id":"R02","if":["high_bp"],"then":"hypertension","source":"AHA/ACC"},
    {"id":"R03","if":["high_cholesterol"],"then":"hyperlipidemia","source":"NHLBI"},
    {"id":"R04","if":["low_max_hr"],"then":"reduced_cardiac_reserve","source":"Mayo Clinic"},
    {"id":"R05","if":["exercise_angina"],"then":"possible_ischemia","source":"NHLBI"},
    {"id":"R06","if":["high_oldpeak"],"then":"abnormal_stress_response","source":"Cardiology guidelines"},
    {"id":"R07","if":["reversible_thal"],"then":"reversible_perfusion_defect","source":"Nuclear Cardiology"},
    {"id":"R08","if":["ecg_abnormal"],"then":"ecg_detected_abnormality","source":"AHA"},
    {"id":"R09","if":["silent_ischemia_risk"],"then":"asymptomatic_cp","source":"Dataset insight"},
    {"id":"R10","if":["hypertension","hyperlipidemia"],"then":"elevated_cardiovascular_risk","source":"AHA"},
    {"id":"R11","if":["possible_ischemia","high_risk"],"then":"suspected_coronary_artery_disease","source":"ACC"},
    {"id":"R12","if":["abnormal_stress_response","possible_ischemia"],"then":"requires_ecg","source":"Mayo Clinic"},
    {"id":"R13","if":["elderly_patient","silent_ischemia_risk"],"then":"critical_screening_needed","source":"AHA"},
    {"id":"R14","if":["reduced_cardiac_reserve","reversible_perfusion_defect"],"then":"exercise_cardiac_failure_risk","source":"Cardiology"},
    {"id":"R15","if":["requires_ecg"],"then":"diagnostic_testing","source":"Clinical pathway"},
    {"id":"R16","if":["diagnostic_testing"],"then":"cardiology_consultation","source":"Clinical pathway"},
    {"id":"R17","if":["high_risk"],"then":"close_monitoring","source":"AHA"},
    {"id":"R18","if":["close_monitoring","diagnostic_testing"],"then":"specialist_followup","source":"Clinical pathway"},
    {"id":"R19","if":["specialist_followup"],"then":"treatment_planning","source":"Clinical pathway"},
    {"id":"R20","if":["low_risk"],"then":"preventive_education","source":"WHO"},
]

def generate_initial_facts(age, trestbps, chol, exang, oldpeak, risk_state,
                            thalch=150, thal_choice=1, restecg_choice=0, cp_choice=4):
    facts = set()
    if risk_state == "High Risk": facts.add("high_risk")
    elif risk_state == "Medium Risk": facts.add("medium_risk")
    else: facts.add("low_risk")
    if age >= 60: facts.add("elderly_patient")
    if trestbps >= 140: facts.add("high_bp")
    if chol >= 240: facts.add("high_cholesterol")
    if thalch < 120: facts.add("low_max_hr")
    if exang == 1: facts.add("exercise_angina")
    if oldpeak > 2: facts.add("high_oldpeak")
    if thal_choice == 3: facts.add("reversible_thal")
    if restecg_choice in [1, 2]: facts.add("ecg_abnormal")
    if cp_choice == 4: facts.add("silent_ischemia_risk")
    return facts

def forward_chaining(facts, rules):
    inferred = set(facts)
    trace = []
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if all(c in inferred for c in rule["if"]) and rule["then"] not in inferred:
                inferred.add(rule["then"])
                trace.append((rule["if"], rule["then"], rule.get("source", "")))
                changed = True
    return inferred, trace

def generate_recommendations(trestbps, chol, exang, oldpeak, age, thalch, risk_state):
    recs = []
    if trestbps >= 140: recs.append("Blood pressure management — antihypertensive therapy recommended")
    if chol >= 240: recs.append("Cholesterol management — statin therapy and dietary intervention")
    if exang == 1: recs.append("ECG evaluation — exercise-induced angina warrants 12-lead ECG")
    if oldpeak > 2.0: recs.append("Stress test evaluation — ST depression indicates cardiac stress")
    if age >= 60: recs.append("Cardiology consultation — age-related risk requires specialist review")
    if thalch < 120: recs.append("Cardiac reserve assessment — low max heart rate indicates reduced function")
    if risk_state == "High Risk": recs.append("URGENT: Schedule cardiologist appointment within 7 days")
    elif risk_state == "Medium Risk": recs.append("Schedule follow-up within 4 to 6 weeks")
    else: recs.append("Continue preventive care — annual cardiac check-up recommended")
    return recs

# ─────────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # --- ML Prediction ---
        df = build_feature_df(data)
        scaled = scaler.transform(df)
        prob = float(model.predict_proba(scaled)[0][1])
        prediction = 1 if prob >= 0.35 else 0
        risk_state = determine_risk(prob)

        age = float(data.get('age', 50))
        trestbps = float(data.get('trestbps', 130))
        chol = float(data.get('chol', 200))
        exang = float(data.get('exang', 0))
        oldpeak = float(data.get('oldpeak', 1.0))
        thalch = float(data.get('thalch', 150))
        thal_choice = float(data.get('thal', 1))
        restecg_choice = float(data.get('restecg', 0))
        cp_choice = float(data.get('cp', 4))

        # --- A* Search ---
        graph = build_medical_graph(risk_state, trestbps, chol, exang, oldpeak, age)
        pathway, cost = a_star_search(graph, risk_state, "Goal State")

        # --- Knowledge Base Forward Chaining ---
        init_facts = generate_initial_facts(
            age, trestbps, chol, exang, oldpeak, risk_state,
            thalch, thal_choice, restecg_choice, cp_choice
        )
        final_facts, trace = forward_chaining(init_facts, knowledge_base)
        new_facts = list(final_facts - init_facts)

        # --- Clinical Recommendations ---
        recs = generate_recommendations(trestbps, chol, exang, oldpeak, age, thalch, risk_state)

        # Build trace for frontend
        trace_list = []
        for conditions, conclusion, source in trace:
            trace_list.append({
                "conditions": conditions,
                "conclusion": conclusion,
                "source": source
            })

        return jsonify({
            'success': True,
            'probability': prob,
            'prediction': prediction,
            'risk_state': risk_state,
            'pathway': pathway,
            'pathway_cost': cost,
            'initial_facts': list(init_facts),
            'derived_facts': new_facts,
            'kb_trace': trace_list,
            'recommendations': recs
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        base_df = build_feature_df(data)
        base_prob = float(model.predict_proba(scaler.transform(base_df))[0][1])

        sensitivity = {}

        def probe(label, key, new_val):
            test_data = dict(data)
            test_data[key] = new_val
            df = build_feature_df(test_data)
            new_prob = float(model.predict_proba(scaler.transform(df))[0][1])
            sensitivity[label] = round((new_prob - base_prob) * 100, 1)

        if float(data.get('trestbps', 130)) > 120:
            probe('Lower BP to 120 mmHg', 'trestbps', 120)
        if float(data.get('chol', 200)) > 180:
            probe('Lower Cholesterol to 180', 'chol', 180)
        current_hr = float(data.get('thalch', 150))
        if current_hr < 180:
            probe('Increase Max HR by 15', 'thalch', min(202, current_hr + 15))
        if float(data.get('oldpeak', 1.0)) > 0:
            probe('Reduce ST Depression to 0', 'oldpeak', 0)
        if float(data.get('exang', 0)) == 1:
            probe('Remove Exercise Angina', 'exang', 0)
        if float(data.get('thal', 1)) != 1:
            probe('Normal Thalassemia', 'thal', 1)

        risk_state = determine_risk(base_prob)

        return jsonify({
            'success': True,
            'base_probability': base_prob,
            'risk_state': risk_state,
            'sensitivity': sensitivity
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
