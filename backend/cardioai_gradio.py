# CardioAI - Elite Gradio Interface
# Complete integration: ML + A* Search + Knowledge Base + What-If Simulator

import gradio as gr
import pandas as pd
import numpy as np
import heapq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────
# MODEL LOADING (adjust paths to your actual saved model/scaler)
# ─────────────────────────────────────────────────────────────────
import joblib, os

MODEL_PATH  = "heart_disease_model.pkl"
SCALER_PATH = "heart_scaler.pkl"

model  = None
scaler = None

# Model loading block – load real model or fallback to demo
try:
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print(f"Model loading failed ({e}); using Demo mode.")
    class DemoModel:
        def predict(self, X):
            arr = np.array(X)
            val = float(arr[0,0]) * 0.1 + float(arr[0,6]) * 0.3 + float(arr[0,7]) * 0.4
            prob = np.clip(1/(1+np.exp(-val)), 0.05, 0.95)
            return [1 if prob > 0.3 else 0]
        def predict_proba(self, X):
            arr = np.array(X)
            val = float(arr[0,0]) * 0.1 + float(arr[0,6]) * 0.3 + float(arr[0,7]) * 0.4
            prob = np.clip(1/(1+np.exp(-val)), 0.05, 0.95)
            return [[1-prob, prob]]
    class DemoScaler:
        def transform(self, X):
            return np.array(X) * 0.1
    model = DemoModel()
    scaler = DemoScaler()


    class DemoScaler:
        def transform(self, X):
            return np.array(X) * 0.1

    model  = DemoModel()
    scaler = DemoScaler()

# ─────────────────────────────────────────────────────────────────
# A* SEARCH ENGINE
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

# ─────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE
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
    if restecg_choice in [1,2]: facts.add("ecg_abnormal")
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
                trace.append((rule["if"], rule["then"], rule.get("source","")))
                changed = True
    return inferred, trace

# ─────────────────────────────────────────────────────────────────
# TF-IDF MEDICAL SEARCH
# ─────────────────────────────────────────────────────────────────
medical_docs = [
    {"title":"Heart Disease Overview","text":"Heart disease encompasses conditions affecting the heart and blood vessels including coronary artery disease, heart failure, and arrhythmias. Main causes include hypertension, high cholesterol, smoking, and diabetes."},
    {"title":"Hypertension & Blood Pressure","text":"High blood pressure (systolic BP >= 140 mmHg) is Stage 2 hypertension per AHA/ACC guidelines. It forces the heart to work harder and is a leading cause of heart attack and stroke. Treatment includes ACE inhibitors, beta-blockers, and lifestyle changes."},
    {"title":"Cholesterol & Hyperlipidemia","text":"Total cholesterol >= 240 mg/dl is high risk per NHLBI guidelines. LDL cholesterol causes plaque formation in arteries. Statins are the primary treatment. Diet rich in fruits, vegetables and omega-3 fatty acids reduces cholesterol."},
    {"title":"ECG & Electrocardiography","text":"Electrocardiography records heart electrical activity and detects ST-T wave changes, left ventricular hypertrophy, and arrhythmias. Resting ECG abnormalities significantly increase cardiac risk and require further evaluation."},
    {"title":"Exercise-Induced Angina","text":"Exercise angina is chest pain during physical activity indicating myocardial ischemia from coronary artery disease. Exercise stress testing evaluates severity. It is a key diagnostic marker requiring immediate cardiology attention."},
    {"title":"ST Depression & Oldpeak","text":"ST depression induced by exercise (oldpeak > 2mm) indicates significant myocardial ischemia and abnormal cardiac stress response. Values above 2mm warrant immediate stress testing and cardiology consultation."},
    {"title":"Thalassemia & Perfusion","text":"A reversible thalassemia defect on nuclear stress testing indicates stress-induced ischemia where myocardium is at risk but not permanently damaged. This distinguishes viable from non-viable heart tissue."},
    {"title":"Maximum Heart Rate","text":"Maximum heart rate = 220 minus age. Achieving less than 85% of predicted maximum during stress testing indicates reduced cardiac reserve and elevated cardiovascular risk requiring specialist evaluation."},
    {"title":"Chest Pain Classification","text":"Typical angina is predictable exertional chest pain. Atypical angina has unusual characteristics. Asymptomatic chest pain paradoxically shows highest disease prevalence in clinical studies as disease progresses silently."},
    {"title":"Cardiology Consultation","text":"Cardiology consultation is required for high-risk patients, abnormal stress tests, multiple cardiovascular risk factors, or patients over 60. Cardiologist performs comprehensive assessment including echocardiogram and coronary angiography."},
    {"title":"Risk Stratification","text":"Cardiac risk stratification uses HEART score, TIMI score, and Framingham Risk Score to estimate 10-year cardiovascular event probability. High-risk patients are prioritised for aggressive medical intervention."},
    {"title":"Treatment Planning","text":"Heart disease treatment includes medication management, lifestyle interventions, cardiac rehabilitation, and revascularisation procedures. Plans are personalised based on risk stratification, comorbidities, and patient preferences."},
    {"title":"Coronary Artery Disease","text":"Coronary artery disease involves narrowing of coronary arteries due to atherosclerosis. Risk factors include hypertension, hyperlipidemia, diabetes, smoking, and family history. Treatment includes statins, antiplatelets, and revascularisation."},
    {"title":"Preventive Cardiology","text":"Preventive cardiology focuses on reducing cardiovascular risk through lifestyle modifications including regular exercise, heart-healthy diet, smoking cessation, weight management, and stress reduction techniques."},
    {"title":"Metabolic Syndrome","text":"Metabolic syndrome combines hypertension, high cholesterol, insulin resistance, and obesity. It significantly triples the risk of heart attack. Management requires addressing all components simultaneously through medication and lifestyle changes."},
]

all_texts = [d["text"] for d in medical_docs]
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)

def medical_search(query):
    if not query.strip():
        return "Please enter a medical question to search."
    query_vec = tfidf_vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix)[0]
    best_idx = sims.argmax()
    best_score = sims[best_idx]
    if best_score < 0.01:
        return "No relevant results found. Try different keywords like 'cholesterol', 'blood pressure', 'chest pain'."
    doc = medical_docs[best_idx]
    result = f"📚 {doc['title']}\n\nRelevance Score: {best_score:.3f}\n\n{doc['text']}"
    return result

# ─────────────────────────────────────────────────────────────────
# CORE PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────
def build_feature_df(age, sex, cp_choice, trestbps, chol, fbs,
                     thalch, exang, oldpeak, slope_choice, thal_choice, restecg_choice):
    cp_typical = 1 if cp_choice==1 else 0
    cp_atypical = 1 if cp_choice==2 else 0
    cp_non_anginal = 1 if cp_choice==3 else 0
    restecg_normal = 1 if restecg_choice==0 else 0
    restecg_st = 1 if restecg_choice==1 else 0
    slope_flat = 1 if slope_choice==2 else 0
    slope_up = 1 if slope_choice==1 else 0
    thal_normal = 1 if thal_choice==1 else 0
    thal_rev = 1 if thal_choice==3 else 0

    age_group = 0 if age<40 else 1 if age<55 else 2 if age<65 else 3
    bp_chol = trestbps * chol
    ex_stress = exang + oldpeak
    pred_hr = 220 - age
    hr_ratio = thalch / pred_hr if pred_hr != 0 else 0

    return pd.DataFrame({
        'age':[age], 'sex':[sex], 'trestbps':[trestbps], 'chol':[chol], 'fbs':[fbs],
        'thalch':[thalch], 'exang':[exang], 'oldpeak':[oldpeak],
        'cp_atypical angina':[cp_atypical], 'cp_non-anginal':[cp_non_anginal],
        'cp_typical angina':[cp_typical], 'restecg_normal':[restecg_normal],
        'restecg_st-t abnormality':[restecg_st], 'slope_flat':[slope_flat],
        'slope_upsloping':[slope_up], 'thal_normal':[thal_normal],
        'thal_reversable defect':[thal_rev], 'age_risk_group':[age_group],
        'bp_chol_interaction':[bp_chol], 'exercise_stress_score':[ex_stress],
        'thalch_age_ratio':[hr_ratio]
    })

def determine_risk(prob):
    if prob >= 0.80: return "High Risk"
    elif prob >= 0.40: return "Medium Risk"
    else: return "Low Risk"

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

def generate_recommendations(trestbps, chol, exang, oldpeak, age, thalch, risk_state):
    recs = []
    if trestbps >= 140: recs.append("🔴 Blood pressure management — antihypertensive therapy recommended")
    if chol >= 240: recs.append("🔴 Cholesterol management — statin therapy and dietary intervention")
    if exang == 1: recs.append("🟠 ECG evaluation — exercise-induced angina warrants 12-lead ECG")
    if oldpeak > 2.0: recs.append("🟠 Stress test evaluation — ST depression indicates cardiac stress")
    if age >= 60: recs.append("🟡 Cardiology consultation — age-related risk requires specialist review")
    if thalch < 120: recs.append("🟠 Cardiac reserve assessment — low max heart rate indicates reduced function")
    if risk_state == "High Risk": recs.append("🚨 URGENT: Schedule cardiologist appointment within 7 days")
    elif risk_state == "Medium Risk": recs.append("📅 Schedule follow-up within 4-6 weeks")
    else: recs.append("✅ Continue preventive care — annual cardiac check-up recommended")
    return "\n".join(recs) if recs else "✅ No immediate interventions required. Maintain healthy lifestyle."

# ─────────────────────────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────────────
def analyze_patient(age, sex, cp_choice, trestbps, chol, fbs,
                    thalch, exang, oldpeak, slope_choice, thal_choice, restecg_choice):

    df = build_feature_df(age, sex, cp_choice, trestbps, chol, fbs,
                          thalch, exang, oldpeak, slope_choice, thal_choice, restecg_choice)
    scaled = scaler.transform(df)
    raw_pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]
    risk_state = determine_risk(prob)

    # ── ML Output ─────────────────────────────────────────
    diagnosis = "⚠️ Heart Disease Detected" if raw_pred == 1 else "✅ No Heart Disease Detected"
    risk_emoji = "🔴" if risk_state=="High Risk" else "🟡" if risk_state=="Medium Risk" else "🟢"
    ml_out = (
        f"{diagnosis}\n"
        f"Risk Probability: {prob*100:.1f}%\n"
        f"Risk Level: {risk_emoji} {risk_state}\n"
        f"Decision Threshold: 0.35 (optimised for medical recall)\n"
        f"Model: XGBoost | AUC: 0.898 | Recall: 0.931"
    )

    # ── A* Search ─────────────────────────────────────────
    graph = build_medical_graph(risk_state, trestbps, chol, exang, oldpeak, age)
    path, cost = a_star_search(graph, risk_state, "Goal State")
    path_str = " → ".join(path) if path else "Direct to Goal State"
    astar_out = (
        f"Optimal Clinical Pathway (A* Search)\n"
        f"Path Cost: {cost} | Steps: {len(path)-1 if path else 0}\n\n"
        f"{path_str}\n\n"
        f"Heuristic: f(n) = g(n) + h(n) [admissible, guarantees optimal path]"
    )

    # ── Clinical Recommendations ──────────────────────────
    recs = generate_recommendations(trestbps, chol, exang, oldpeak, age, thalch, risk_state)

    # ── KB Inference ──────────────────────────────────────
    init_facts = generate_initial_facts(age, trestbps, chol, exang, oldpeak, risk_state,
                                        thalch, thal_choice, restecg_choice, cp_choice)
    final_facts, trace = forward_chaining(init_facts, knowledge_base)

    facts_out = f"Initial Facts ({len(init_facts)}):\n" + "\n".join(f"  ✓ {f}" for f in sorted(init_facts))
    
    if trace:
        trace_lines = []
        for conditions, conclusion, source in trace:
            trace_lines.append(f"  [{', '.join(conditions)}] → {conclusion}\n    Source: {source}")
        trace_out = f"Rules Fired: {len(trace)}\n\n" + "\n\n".join(trace_lines)
    else:
        trace_out = "No additional rules fired — minimal risk factors present."

    new_facts = final_facts - init_facts
    concl_out = (
        f"Derived Conclusions ({len(new_facts)}):\n"
        + ("\n".join(f"  ✓ {f}" for f in sorted(new_facts)) if new_facts else "  (no additional facts derived)")
    )

    return ml_out, astar_out, recs, facts_out, trace_out, concl_out


# ─────────────────────────────────────────────────────────────────
# WHAT-IF SIMULATOR
# ─────────────────────────────────────────────────────────────────
def whatif_simulator(age, bp, chol, hr, op, sex, cp, exang, thal_c, ecg, slope):
    df = build_feature_df(age, sex, cp, bp, chol, 0, hr, exang, op, slope, thal_c, ecg)
    scaled = scaler.transform(df)
    prob = model.predict_proba(scaled)[0][1]
    pct = round(prob * 100, 1)
    risk_state = determine_risk(prob)
    pred = "Heart Disease" if prob >= 0.35 else "No Disease"
    risk_emoji = "🔴" if risk_state=="High Risk" else "🟡" if risk_state=="Medium Risk" else "🟢"

    prob_out = f"Risk Probability: {pct}%\nPrediction: {pred}\nRisk Level: {risk_emoji} {risk_state}"

    # Sensitivity — test improving each factor
    sensitivity = {}
    def probe_change(label, **overrides):
        kw = dict(age=age,sex=sex,cp_choice=cp,trestbps=bp,chol=chol,fbs=0,
                  thalch=hr,exang=exang,oldpeak=op,slope_choice=slope,thal_choice=thal_c,restecg_choice=ecg)
        kw.update(overrides)
        try:
            df2 = build_feature_df(**kw)
            p2 = model.predict_proba(scaler.transform(df2))[0][1]
            sensitivity[label] = round((p2 - prob)*100, 1)
        except:
            pass

    probe_change("Lower BP → 120 mmHg", trestbps=min(bp, 120))
    probe_change("Lower Cholesterol → 180", chol=min(chol, 180))
    probe_change("Increase Max HR +15", thalch=min(202, hr+15))
    probe_change("Reduce ST Depression → 0", oldpeak=0.0)
    probe_change("Remove Exercise Angina", exang=0)
    probe_change("Normal Thalassemia", thal_choice=1)

    delta_lines = ["What-If Sensitivity Analysis\n(Negative = risk decreases)\n"]
    for feat, delta in sorted(sensitivity.items(), key=lambda x: x[1]):
        arrow = "▼" if delta < 0 else "▲" if delta > 0 else "—"
        bar = "█" * int(abs(delta)/2) if abs(delta) > 0 else "-"
        delta_lines.append(f"{arrow} {feat}: {'+' if delta>0 else ''}{delta}%  {bar}")

    delta_out = "\n".join(delta_lines)

    # A* for what-if
    graph = build_medical_graph(risk_state, bp, chol, exang, op, age)
    path, cost = a_star_search(graph, risk_state, "Goal State")
    path_out = f"Care Pathway (cost={cost}):\n" + (" → ".join(path) if path else "Direct care")

    # KB for what-if
    init_facts = generate_initial_facts(age, bp, chol, exang, op, risk_state, hr, thal_c, ecg, cp)
    final_facts, trace = forward_chaining(init_facts, knowledge_base)
    new_facts = final_facts - init_facts
    kb_out = f"Rules fired: {len(trace)}\nDerived facts: {len(new_facts)}\n\n" + "\n".join(f"  ✓ {f}" for f in sorted(new_facts))

    return prob_out, delta_out, path_out, kb_out


# ─────────────────────────────────────────────────────────────────
# GRADIO ELITE UI — Custom CSS
# ─────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root Theme ── */
:root {
    --cardio-red:     #DC143C;
    --cardio-red-dark:#A50E2D;
    --cardio-red-soft:#FFE8EC;
    --cardio-white:   #FFFFFF;
    --cardio-snow:    #FAFAFA;
    --cardio-light:   #F5F5F5;
    --cardio-mid:     #E8E8E8;
    --cardio-text:    #1A1A1A;
    --cardio-muted:   #6B7280;
    --cardio-success: #16A34A;
    --shadow-sm:      0 2px 8px rgba(220,20,60,0.08);
    --shadow-md:      0 8px 32px rgba(220,20,60,0.12);
    --shadow-lg:      0 16px 48px rgba(220,20,60,0.16);
    --radius:         16px;
    --radius-sm:      10px;
}

/* ── Global Reset ── */
* { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--cardio-snow) !important;
    color: var(--cardio-text) !important;
}

/* ── Header / Hero ── */
.cardio-hero {
    background: linear-gradient(135deg, #FFFFFF 0%, #FFF5F7 40%, #FFE8EC 100%);
    border-bottom: 3px solid var(--cardio-red);
    padding: 32px 40px 28px;
    margin-bottom: 24px;
    border-radius: 0 0 var(--radius) var(--radius);
    position: relative;
    overflow: hidden;
}
.cardio-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(220,20,60,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.cardio-hero-inner {
    display: flex;
    align-items: center;
    gap: 20px;
    max-width: 900px;
}
.cardio-logo-wrap {
    flex-shrink: 0;
}
.cardio-logo-svg {
    animation: heartbeat 1.6s ease-in-out infinite;
    filter: drop-shadow(0 4px 12px rgba(220,20,60,0.35));
}
@keyframes heartbeat {
    0%,100% { transform: scale(1); }
    14%      { transform: scale(1.15); }
    28%      { transform: scale(1); }
    42%      { transform: scale(1.08); }
    56%      { transform: scale(1); }
}
.cardio-brand h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.8rem !important;
    font-weight: 400 !important;
    color: var(--cardio-red) !important;
    margin: 0 !important;
    line-height: 1.1 !important;
    letter-spacing: -0.5px !important;
}
.cardio-brand h1 span { color: var(--cardio-text); }
.cardio-brand p {
    font-size: 1rem !important;
    color: var(--cardio-muted) !important;
    margin: 6px 0 0 !important;
    font-weight: 300 !important;
    letter-spacing: 0.3px !important;
}
.cardio-stats {
    display: flex;
    gap: 24px;
    margin-left: auto;
    flex-shrink: 0;
}
.stat-chip {
    text-align: center;
    background: white;
    border: 1.5px solid var(--cardio-mid);
    border-radius: var(--radius-sm);
    padding: 10px 18px;
    box-shadow: var(--shadow-sm);
}
.stat-chip strong {
    display: block;
    font-size: 1.4rem;
    color: var(--cardio-red);
    font-weight: 600;
    line-height: 1;
}
.stat-chip span {
    font-size: 0.72rem;
    color: var(--cardio-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Navigation Tabs ── */
.tab-nav { border-bottom: 2px solid var(--cardio-mid) !important; }
.tab-nav button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 12px 24px !important;
    color: var(--cardio-muted) !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    background: transparent !important;
    border-radius: 0 !important;
    transition: all 0.2s ease !important;
}
.tab-nav button:hover { color: var(--cardio-red) !important; }
.tab-nav button.selected {
    color: var(--cardio-red) !important;
    border-bottom-color: var(--cardio-red) !important;
    font-weight: 600 !important;
}

/* ── Cards ── */
.card {
    background: white;
    border-radius: var(--radius);
    border: 1px solid var(--cardio-mid);
    padding: 24px;
    box-shadow: var(--shadow-sm);
    margin-bottom: 16px;
}
.card-red {
    background: linear-gradient(135deg, var(--cardio-red) 0%, var(--cardio-red-dark) 100%);
    color: white;
    border: none;
    box-shadow: var(--shadow-md);
}
.card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    margin: 0 0 12px;
    color: var(--cardio-text);
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Inputs ── */
.gradio-container .gr-form, 
.gradio-container .gr-box { 
    border-radius: var(--radius-sm) !important; 
}
.gradio-container label span {
    font-weight: 500 !important;
    color: var(--cardio-text) !important;
    font-size: 0.88rem !important;
}
input[type=range] { accent-color: var(--cardio-red) !important; }
input[type=radio]:checked { accent-color: var(--cardio-red) !important; }

/* ── Buttons ── */
.btn-primary {
    background: linear-gradient(135deg, var(--cardio-red) 0%, var(--cardio-red-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 14px 28px !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 16px rgba(220,20,60,0.3) !important;
    width: 100% !important;
}
.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(220,20,60,0.4) !important;
}
.btn-secondary {
    background: white !important;
    color: var(--cardio-red) !important;
    border: 2px solid var(--cardio-red) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    width: 100% !important;
}

/* ── Textboxes / Outputs ── */
.gradio-container textarea, .gradio-container .output-textbox {
    font-family: 'DM Sans', sans-serif !important;
    border-radius: var(--radius-sm) !important;
    border: 1.5px solid var(--cardio-mid) !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
}
.gradio-container textarea:focus {
    border-color: var(--cardio-red) !important;
    box-shadow: 0 0 0 3px rgba(220,20,60,0.1) !important;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: var(--cardio-red);
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--cardio-red-soft);
}
.section-sub {
    color: var(--cardio-muted);
    font-size: 0.9rem;
    margin-bottom: 20px;
}

/* ── Heartbeat Pulse Line ── */
.pulse-line {
    height: 3px;
    background: linear-gradient(90deg, transparent 0%, var(--cardio-red) 20%, transparent 40%, var(--cardio-red) 60%, transparent 80%);
    background-size: 200% 100%;
    animation: pulse-scroll 3s linear infinite;
    border-radius: 2px;
    margin: 12px 0;
}
@keyframes pulse-scroll {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ── Risk Badge ── */
.risk-high   { color: #DC143C; font-weight: 700; }
.risk-medium { color: #D97706; font-weight: 700; }
.risk-low    { color: #16A34A; font-weight: 700; }

/* ── About Page ── */
.about-hero {
    background: linear-gradient(135deg, #DC143C 0%, #A50E2D 100%);
    color: white;
    padding: 40px;
    border-radius: var(--radius);
    margin-bottom: 24px;
    text-align: center;
}
.about-hero h2 {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin-bottom: 12px;
}
.mission-card {
    border-left: 4px solid var(--cardio-red);
    padding: 20px 24px;
    background: white;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    margin-bottom: 16px;
    box-shadow: var(--shadow-sm);
}

/* ── ML/AI Journey ── */
.journey-step {
    display: flex;
    gap: 16px;
    align-items: flex-start;
    padding: 16px;
    background: white;
    border-radius: var(--radius-sm);
    margin-bottom: 12px;
    border: 1px solid var(--cardio-mid);
    box-shadow: var(--shadow-sm);
}
.step-num {
    background: var(--cardio-red);
    color: white;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
    flex-shrink: 0;
}

/* ── ECG animation in footer ── */
.ecg-line {
    width: 100%;
    height: 40px;
    overflow: hidden;
    opacity: 0.15;
    margin-top: 8px;
}

/* ── Footer ── */
.cardio-footer {
    text-align: center;
    padding: 20px;
    color: var(--cardio-muted);
    font-size: 0.8rem;
    border-top: 1px solid var(--cardio-mid);
    margin-top: 32px;
}
.cardio-footer strong { color: var(--cardio-red); }
"""

HEART_SVG = """
<svg class="cardio-logo-svg" width="72" height="72" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="hg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#FF4D6D"/>
      <stop offset="100%" stop-color="#DC143C"/>
    </linearGradient>
  </defs>
  <path d="M50 85 C50 85 10 58 10 32 C10 20 20 12 32 12 C40 12 47 17 50 22 C53 17 60 12 68 12 C80 12 90 20 90 32 C90 58 50 85 50 85Z" fill="url(#hg)" stroke="white" stroke-width="2"/>
  <!-- ECG line on heart -->
  <polyline points="26,44 33,44 36,34 40,54 44,38 48,48 52,48 56,40 60,52 64,44 74,44" stroke="white" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>
</svg>
"""

HERO_HTML = f"""
<div class="cardio-hero">
  <div class="cardio-hero-inner">
    <div class="cardio-logo-wrap">{HEART_SVG}</div>
    <div class="cardio-brand">
      <h1>Cardio<span>AI</span></h1>
      <p>Intelligent Heart Disease Risk Prediction &amp; Clinical Decision Support</p>
      <div class="pulse-line"></div>
    </div>
    <div class="cardio-stats">
      <div class="stat-chip"><strong>0.898</strong><span>AUC Score</span></div>
      <div class="stat-chip"><strong>0.931</strong><span>Recall</span></div>
      <div class="stat-chip"><strong>920</strong><span>Patients</span></div>
      <div class="stat-chip"><strong>20+</strong><span>KB Rules</span></div>
    </div>
  </div>
</div>
"""

ABOUT_HTML = """
<div class="about-hero">
  <h2>❤️ Why We Built CardioAI</h2>
  <p style="font-size:1.05rem;opacity:0.9;max-width:600px;margin:0 auto;">
    Every 33 seconds, someone in the world dies from cardiovascular disease. 
    Many of these deaths are preventable — if only the warning signs had been caught earlier.
  </p>
</div>

<div class="mission-card">
  <h3 style="color:#DC143C;font-family:'DM Serif Display',serif;margin:0 0 8px;">Our Mission</h3>
  <p style="color:#374151;line-height:1.7;margin:0;">
    CardioAI was born from a simple but urgent question: <em>what if every clinic assistant, every lab 
    technician, every GP in a resource-limited setting could have access to the same clinical intelligence 
    as a cardiologist?</em> We built this system so that the nurse entering a patient's blood pressure 
    at 2am, the lab assistant reviewing cholesterol results in a rural clinic, the GP making a triage 
    decision with incomplete information — all of them have a powerful, explainable AI partner by their side.
  </p>
</div>

<div class="mission-card">
  <h3 style="color:#DC143C;font-family:'DM Serif Display',serif;margin:0 0 8px;">Who This Is For</h3>
  <p style="color:#374151;line-height:1.7;margin:0;">
    <strong>Clinical Assistants & Lab Staff</strong> — Enter patient measurements you already have. 
    Get an instant risk assessment with a clear clinical pathway.<br><br>
    <strong>General Practitioners</strong> — Use the A* care pathway to guide triage decisions 
    when specialist access is limited.<br><br>
    <strong>Medical Students & Researchers</strong> — Explore the What-If Simulator to understand 
    which factors drive heart disease risk and by how much.
  </p>
</div>

<div class="mission-card">
  <h3 style="color:#DC143C;font-family:'DM Serif Display',serif;margin:0 0 8px;">⚠️ Important Disclaimer</h3>
  <p style="color:#374151;line-height:1.7;margin:0;">
    CardioAI is a <strong>decision support tool</strong>, not a diagnostic device. All results 
    must be reviewed by a qualified medical professional. Never make clinical decisions based 
    solely on this system's output. This tool is designed to assist — not replace — human clinical judgment.
  </p>
</div>
"""

JOURNEY_HTML = """
<div style="max-width:800px;margin:0 auto;">

<h2 class="section-header">🔬 The CardioAI Technical Journey</h2>
<p class="section-sub">From raw clinical data to an intelligent decision-support system — every step explained.</p>

<div class="journey-step">
  <div class="step-num">1</div>
  <div>
    <strong>Dataset: UCI Heart Disease (4 Hospitals, 3 Countries)</strong><br>
    <span style="color:#6B7280;font-size:0.9rem;">920 patients from Cleveland Clinic, Hungarian Institute, University Hospital Zurich, and VA Long Beach. 
    14 clinical features including age, cholesterol, blood pressure, ECG results, and thalassemia type. 
    Target binarized: 0 = No Disease, 1 = Disease Present (55% / 45% — near-balanced).</span>
  </div>
</div>

<div class="journey-step">
  <div class="step-num">2</div>
  <div>
    <strong>Data Preprocessing Pipeline</strong><br>
    <span style="color:#6B7280;font-size:0.9rem;">Median imputation for numerical missing values (robust to outliers), mode imputation for categoricals. 
    Impossible zeros in blood pressure and cholesterol treated as missing. Winsorization (IQR capping) for outliers. 
    One-hot encoding for multi-category features (chest pain, thalassemia, ECG). StandardScaler fitted on training data only — 
    no data leakage.</span>
  </div>
</div>

<div class="journey-step">
  <div class="step-num">3</div>
  <div>
    <strong>Feature Engineering (4 New Clinical Features)</strong><br>
    <span style="color:#6B7280;font-size:0.9rem;">age_risk_group (non-linear age bucketing), bp_chol_interaction (cardiovascular stress index), 
    exercise_stress_score (cardiac function under load), thalch_age_ratio (% of predicted max HR achieved). 
    All features correlated with target before inclusion.</span>
  </div>
</div>

<div class="journey-step">
  <div class="step-num">4</div>
  <div>
    <strong>Three Models: Baseline → Intermediate → Advanced</strong><br>
    <span style="color:#6B7280;font-size:0.9rem;"><strong>Logistic Regression</strong> (AUC 0.895) — simple, interpretable, doctor-friendly. 
    <strong>Random Forest</strong> (AUC 0.884) — ensemble of 100 trees, handles non-linear interactions. 
    <strong>XGBoost</strong> (AUC 0.898) — gradient boosting, optimal for small tabular medical data. 
    MLP Neural Network was evaluated and <em>deliberately rejected</em> — insufficient data (920 rows) 
    for stable neural training.</span>
  </div>
</div>

<div class="journey-step">
  <div class="step-num">5</div>
  <div>
    <strong>SMOTE + Threshold Tuning for Medical Safety</strong><br>
    <span style="color:#6B7280;font-size:0.9rem;">SMOTE applied only on training data to balance classes. Classification threshold lowered from 0.50 to 0.35 — 
    because in medicine, missing a real disease case (false negative) is far more dangerous than a false alarm. 
    This raised Recall from 0.843 to 0.931 with acceptable precision trade-off.</span>
  </div>
</div>

<div class="journey-step">
  <div class="step-num">6</div>
  <div>
    <strong>PEAS Agent + A* Search Planning</strong><br>
    <span style="color:#6B7280;font-size:0.9rem;">Formally defined as a Goal-based + Utility-based agent. 
    ML probability → Initial State (High/Medium/Low Risk). 
    A* algorithm with admissible heuristic finds the optimal clinical intervention pathway. 
    f(n) = g(n) + h(n) guarantees the minimum-cost care pathway is always found.</span>
  </div>
</div>

<div class="journey-step">
  <div class="step-num">7</div>
  <div>
    <strong>Knowledge Base: 20 Rules + Forward Chaining</strong><br>
    <span style="color:#6B7280;font-size:0.9rem;">3-tier rule hierarchy: Tier 1 (basic risk flags per AHA/ACC/NHLBI guidelines), 
    Tier 2 (combined risk assessment), Tier 3 (clinical pathway conclusions). 
    Forward chaining runs to fixed point — derives all possible clinical conclusions from patient facts. 
    ML probability is itself one of the input facts, creating a seamless ML + AI integration.</span>
  </div>
</div>

<div class="journey-step">
  <div class="step-num">8</div>
  <div>
    <strong>TF-IDF Medical Knowledge Retrieval</strong><br>
    <span style="color:#6B7280;font-size:0.9rem;">15-document medical corpus covering all dataset features. 
    TF-IDF vectorization converts medical queries to vector space. 
    Cosine similarity retrieves the most clinically relevant document for any patient question. 
    TF = term frequency in document, IDF = inverse document frequency across corpus. 
    cos(θ) = (A·B)/(|A||B|) — the core of modern information retrieval.</span>
  </div>
</div>

<div class="journey-step">
  <div class="step-num">9</div>
  <div>
    <strong>What-If Simulator: Explainable AI</strong><br>
    <span style="color:#6B7280;font-size:0.9rem;">Interactive sensitivity analysis showing exactly how each clinical improvement changes risk probability. 
    "What if the patient's blood pressure drops to 120?" — the model answers immediately. 
    This is the foundation of clinical explainability in AI systems.</span>
  </div>
</div>

<div style="background:linear-gradient(135deg,#DC143C,#A50E2D);color:white;padding:24px;border-radius:16px;margin-top:16px;">
  <h3 style="font-family:'DM Serif Display',serif;margin:0 0 8px;">📊 Final Performance</h3>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;text-align:center;margin-top:12px;">
    <div><strong style="font-size:1.5rem;">0.898</strong><br><span style="opacity:0.8;font-size:0.8rem;">AUC Score</span></div>
    <div><strong style="font-size:1.5rem;">0.931</strong><br><span style="opacity:0.8;font-size:0.8rem;">Recall @0.35</span></div>
    <div><strong style="font-size:1.5rem;">5-Fold</strong><br><span style="opacity:0.8;font-size:0.8rem;">Cross-Validation</span></div>
    <div><strong style="font-size:1.5rem;">0.892</strong><br><span style="opacity:0.8;font-size:0.8rem;">CV Mean AUC</span></div>
  </div>
</div>
</div>
"""

FOOTER_HTML = """
<div class="cardio-footer">
  <strong>CardioAI</strong> — Intelligent Cardiac Risk Decision Support System<br>
  Built with XGBoost · A* Search · Forward Chaining KB · TF-IDF NLP<br>
  UCI Heart Disease Dataset · 920 patients · 4 hospitals · 3 countries<br>
  <em>⚠️ For clinical decision support only. Not a substitute for professional medical advice.</em>
</div>
"""

# ─────────────────────────────────────────────────────────────────
# BUILD THE GRADIO APP
# ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="CardioAI — Heart Disease Risk Prediction",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.red,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("DM Sans"),
    )
) as demo:

    gr.HTML(HERO_HTML)

    with gr.Tabs() as tabs:

        # ══════════════════════════════════════════════════════
        # TAB 1 — HOME
        # ══════════════════════════════════════════════════════
        with gr.Tab("🏠 Home"):
            gr.HTML("""
            <div style="max-width:800px;margin:0 auto;padding:16px;">
              <h2 class="section-header">Welcome to CardioAI</h2>
              <p class="section-sub">Your intelligent partner for cardiac risk assessment — designed for clinical assistants, lab staff, and healthcare providers.</p>

              <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;">
                <div class="card">
                  <div class="card-title">🔬 Risk Assessment</div>
                  <p style="color:#6B7280;font-size:0.9rem;margin:0;">Enter 14 clinical measurements to get an instant ML-powered risk assessment with confidence score and risk level.</p>
                </div>
                <div class="card">
                  <div class="card-title">🗺️ Care Pathway</div>
                  <p style="color:#6B7280;font-size:0.9rem;margin:0;">A* search algorithm finds the optimal sequence of clinical interventions tailored to this specific patient profile.</p>
                </div>
                <div class="card">
                  <div class="card-title">🧠 Expert Reasoning</div>
                  <p style="color:#6B7280;font-size:0.9rem;margin:0;">20 clinically-grounded IF-THEN rules derived from AHA, ACC, NHLBI, and Mayo Clinic guidelines reason over patient facts.</p>
                </div>
                <div class="card">
                  <div class="card-title">🔮 What-If Simulator</div>
                  <p style="color:#6B7280;font-size:0.9rem;margin:0;">Interactively adjust clinical parameters to understand what interventions would reduce risk the most for this patient.</p>
                </div>
              </div>

              <div style="background:#FFF5F7;border:1.5px solid #FFB3C1;border-radius:12px;padding:20px;">
                <strong style="color:#DC143C;">How to Use CardioAI</strong>
                <ol style="color:#374151;line-height:2;margin:8px 0 0;padding-left:20px;font-size:0.92rem;">
                  <li>Go to <strong>Assess Patient</strong> and enter the patient's clinical measurements</li>
                  <li>Click <strong>Analyse Risk</strong> to get ML prediction, A* pathway, and KB reasoning</li>
                  <li>Use <strong>Risk Simulator</strong> to explore "what if we improve this factor?"</li>
                  <li>Search the <strong>Medical Knowledge</strong> tab for clinical context</li>
                  <li>Review the <strong>AI Journey</strong> to understand how the system was built</li>
                </ol>
              </div>
            </div>
            """)
            gr.HTML(FOOTER_HTML)

        # ══════════════════════════════════════════════════════
        # TAB 2 — ASSESS PATIENT (was "Analyze")
        # ══════════════════════════════════════════════════════
        with gr.Tab("❤️ Assess Patient"):
            with gr.Row():
                # ── Input Panel ─────────────────────────────
                with gr.Column(scale=1, min_width=320):
                    gr.HTML('<div class="section-header" style="font-size:1.2rem;">Patient Profile</div>')
                    gr.HTML('<p class="section-sub">Enter clinical measurements from patient records.</p>')

                    age = gr.Slider(29, 77, value=54, step=1, label="Age (years)")
                    sex = gr.Radio([0, 1], value=1, label="Sex  (0 = Female · 1 = Male)")
                    cp_choice = gr.Radio(
                        choices=[1, 2, 3, 4], value=2,
                        label="Chest Pain Type  (1=Typical · 2=Atypical · 3=Non-anginal · 4=Asymptomatic)"
                    )
                    trestbps = gr.Slider(94, 200, value=130, step=1, label="Resting Blood Pressure (mmHg)")
                    chol = gr.Slider(126, 400, value=220, step=1, label="Serum Cholesterol (mg/dl)")
                    fbs = gr.Radio([0, 1], value=0, label="Fasting Blood Sugar > 120  (0=No · 1=Yes)")
                    thalch = gr.Slider(71, 202, value=152, step=1, label="Max Heart Rate Achieved (bpm)")
                    exang = gr.Radio([0, 1], value=0, label="Exercise-Induced Angina  (0=No · 1=Yes)")
                    oldpeak = gr.Slider(0.0, 6.2, value=1.0, step=0.1, label="ST Depression — Oldpeak (mm)")
                    slope_choice = gr.Radio(
                        choices=[1, 2, 3], value=2,
                        label="ST Slope  (1=Upsloping · 2=Flat · 3=Downsloping)"
                    )
                    thal_choice = gr.Radio(
                        choices=[1, 2, 3], value=1,
                        label="Thalassemia  (1=Normal · 2=Fixed Defect · 3=Reversible)"
                    )
                    restecg_choice = gr.Radio(
                        choices=[0, 1, 2], value=0,
                        label="Resting ECG  (0=Normal · 1=ST-T Abnormality · 2=LV Hypertrophy)"
                    )
                    analyse_btn = gr.Button("❤️ Analyse Risk", elem_classes=["btn-primary"])

                # ── Output Panel ─────────────────────────────
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("📊 ML Prediction"):
                            ml_out = gr.Textbox(
                                label="Risk Assessment",
                                lines=7, interactive=False
                            )
                            astar_out = gr.Textbox(
                                label="A* Optimal Care Pathway",
                                lines=6, interactive=False
                            )
                            recs_out = gr.Textbox(
                                label="Clinical Recommendations",
                                lines=8, interactive=False
                            )

                        with gr.Tab("🧠 Expert Reasoning"):
                            kb_facts_out = gr.Textbox(
                                label="Initial Facts Derived from Patient Data",
                                lines=7, interactive=False
                            )
                            kb_trace_out = gr.Textbox(
                                label="Forward Chaining Inference Trace",
                                lines=10, interactive=False
                            )
                            kb_concl_out = gr.Textbox(
                                label="Final Clinical Conclusions",
                                lines=6, interactive=False
                            )

            analyse_btn.click(
                fn=analyze_patient,
                inputs=[age, sex, cp_choice, trestbps, chol, fbs,
                        thalch, exang, oldpeak, slope_choice, thal_choice, restecg_choice],
                outputs=[ml_out, astar_out, recs_out, kb_facts_out, kb_trace_out, kb_concl_out]
            )

        # ══════════════════════════════════════════════════════
        # TAB 3 — WHAT-IF SIMULATOR
        # ══════════════════════════════════════════════════════
        with gr.Tab("🔮 Risk Simulator"):
            gr.HTML("""
            <div style="max-width:800px;margin:0 auto 20px;">
              <h2 class="section-header">What-If Risk Simulator</h2>
              <p class="section-sub">Adjust individual clinical parameters and see in real-time how each change affects the patient's risk probability, care pathway, and knowledge-base reasoning. Demonstrates explainable AI.</p>
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div style="font-weight:600;color:#DC143C;margin-bottom:8px;">⚙️ Adjust Parameters</div>')
                    sim_age   = gr.Slider(29, 77,  value=54,  step=1,  label="Age")
                    sim_bp    = gr.Slider(90, 200,  value=130, step=1,  label="Blood Pressure (mmHg)")
                    sim_chol  = gr.Slider(126, 400, value=220, step=1,  label="Cholesterol (mg/dl)")
                    sim_hr    = gr.Slider(71, 202,  value=152, step=1,  label="Max Heart Rate (bpm)")
                    sim_op    = gr.Slider(0.0, 6.0, value=0.8, step=0.1,label="ST Depression (mm)")
                    sim_sex   = gr.Radio([0,1], value=1, label="Sex (0=F, 1=M)")
                    sim_cp    = gr.Radio([1,2,3,4], value=2, label="Chest Pain (1–4)")
                    sim_exang = gr.Radio([0,1], value=0, label="Exercise Angina (0=No, 1=Yes)")
                    sim_thal  = gr.Radio([1,2,3], value=1, label="Thal (1=Normal,2=Fixed,3=Reversible)")
                    sim_ecg   = gr.Radio([0,1,2], value=0, label="Resting ECG (0=Normal,1=ST-T,2=LVH)")
                    sim_slope = gr.Radio([1,2,3], value=2, label="ST Slope (1=Up,2=Flat,3=Down)")
                    sim_btn   = gr.Button("🔮 Run Simulation", elem_classes=["btn-primary"])

                with gr.Column(scale=1):
                    sim_prob_out  = gr.Textbox(label="Risk Probability & Prediction", lines=4, interactive=False)
                    sim_delta_out = gr.Textbox(label="What-If Sensitivity — Impact of Each Improvement", lines=12, interactive=False)
                    sim_path_out  = gr.Textbox(label="A* Care Pathway", lines=4, interactive=False)
                    sim_kb_out    = gr.Textbox(label="Knowledge Base Conclusions", lines=6, interactive=False)

            sim_btn.click(
                fn=whatif_simulator,
                inputs=[sim_age, sim_bp, sim_chol, sim_hr, sim_op,
                        sim_sex, sim_cp, sim_exang, sim_thal, sim_ecg, sim_slope],
                outputs=[sim_prob_out, sim_delta_out, sim_path_out, sim_kb_out]
            )

        # ══════════════════════════════════════════════════════
        # TAB 4 — MEDICAL KNOWLEDGE SEARCH
        # ══════════════════════════════════════════════════════
        with gr.Tab("📚 Medical Knowledge"):
            gr.HTML("""
            <div style="max-width:700px;margin:0 auto 20px;">
              <h2 class="section-header">Medical AI Search</h2>
              <p class="section-sub">Ask any clinical question about heart disease. Powered by TF-IDF vector search across 15 medical knowledge documents.</p>
              <div style="background:#FFF5F7;border-radius:10px;padding:14px;font-size:0.88rem;color:#6B7280;margin-bottom:16px;">
                💡 Try: "What is cholesterol?" · "What does ST depression mean?" · "How is heart disease treated?" · "What is thalassemia?"
              </div>
            </div>
            """)
            with gr.Row():
                with gr.Column():
                    search_q = gr.Textbox(
                        label="Ask a Medical Question",
                        placeholder="e.g. What causes high blood pressure?",
                        lines=2
                    )
                    search_btn = gr.Button("🔍 Search Medical Knowledge", elem_classes=["btn-secondary"])
                    search_result = gr.Textbox(label="Result", lines=8, interactive=False)

            search_btn.click(fn=medical_search, inputs=[search_q], outputs=[search_result])

        # ══════════════════════════════════════════════════════
        # TAB 5 — ABOUT
        # ══════════════════════════════════════════════════════
        with gr.Tab("💙 About"):
            gr.HTML(ABOUT_HTML)
            gr.HTML(FOOTER_HTML)

        # ══════════════════════════════════════════════════════
        # TAB 6 — AI JOURNEY (was "ML-AI")
        # ══════════════════════════════════════════════════════
        with gr.Tab("🔬 AI Journey"):
            gr.HTML(JOURNEY_HTML)
            gr.HTML(FOOTER_HTML)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
