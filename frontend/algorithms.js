/* ─────────────────────────────────────────────────────────────────── */
/* CARDIOAI — ALGORITHMS */
/* A* Search, Knowledge Base, TF-IDF Search */
/* ─────────────────────────────────────────────────────────────────── */

// ─────────────────────────────────────────────────────────────────
// HEURISTIC VALUES FOR A* SEARCH
// ─────────────────────────────────────────────────────────────────
const heuristic = {
    'High Risk': 5,
    'Medium Risk': 3,
    'Low Risk': 1,
    'Blood Pressure Management': 4,
    'Cholesterol Management': 4,
    'ECG Evaluation': 3,
    'Stress Evaluation': 3,
    'Cardiology Consultation': 2,
    'Risk Stratification': 1,
    'Treatment Planning': 1,
    'Goal State': 0
};

// ─────────────────────────────────────────────────────────────────
// A* SEARCH ALGORITHM
// ─────────────────────────────────────────────────────────────────
function aStarSearch(graph, start, goal) {
    const queue = [];
    const visited = new Set();

    // Priority queue with [f_score, node, path, g_score]
    const push = (f, node, path, g) => {
        queue.push([f, node, path, g]);
        queue.sort((a, b) => a[0] - b[0]);
    };

    push(0, start, [start], 0);

    while (queue.length > 0) {
        const [f, node, path, g] = queue.shift();

        if (node === goal) {
            return { path, cost: g };
        }

        if (visited.has(node)) {
            continue;
        }

        visited.add(node);

        const neighbors = graph[node] || {};
        for (const [neighbor, cost] of Object.entries(neighbors)) {
            if (neighbor in heuristic) {
                const newG = g + cost;
                const newF = newG + (heuristic[neighbor] || 2);
                push(newF, neighbor, [...path, neighbor], newG);
            }
        }
    }

    return { path: [start, goal], cost: 1 };
}

// ─────────────────────────────────────────────────────────────────
// KNOWLEDGE BASE
// ─────────────────────────────────────────────────────────────────
const knowledgeBase = [
    { id: 'R01', if: ['elderly_patient'], then: 'age_related_cardiac_risk', source: 'AHA' },
    { id: 'R02', if: ['high_bp'], then: 'hypertension', source: 'AHA/ACC' },
    { id: 'R03', if: ['high_cholesterol'], then: 'hyperlipidemia', source: 'NHLBI' },
    { id: 'R04', if: ['low_max_hr'], then: 'reduced_cardiac_reserve', source: 'Mayo Clinic' },
    { id: 'R05', if: ['exercise_angina'], then: 'possible_ischemia', source: 'NHLBI' },
    { id: 'R06', if: ['high_oldpeak'], then: 'abnormal_stress_response', source: 'Cardiology guidelines' },
    { id: 'R07', if: ['reversible_thal'], then: 'reversible_perfusion_defect', source: 'Nuclear Cardiology' },
    { id: 'R08', if: ['ecg_abnormal'], then: 'ecg_detected_abnormality', source: 'AHA' },
    { id: 'R09', if: ['silent_ischemia_risk'], then: 'asymptomatic_cp', source: 'Dataset insight' },
    { id: 'R10', if: ['hypertension', 'hyperlipidemia'], then: 'elevated_cardiovascular_risk', source: 'AHA' },
    { id: 'R11', if: ['possible_ischemia', 'high_risk'], then: 'suspected_coronary_artery_disease', source: 'ACC' },
    { id: 'R12', if: ['abnormal_stress_response', 'possible_ischemia'], then: 'requires_ecg', source: 'Mayo Clinic' },
    { id: 'R13', if: ['elderly_patient', 'silent_ischemia_risk'], then: 'critical_screening_needed', source: 'AHA' },
    { id: 'R14', if: ['reduced_cardiac_reserve', 'reversible_perfusion_defect'], then: 'exercise_cardiac_failure_risk', source: 'Cardiology' },
    { id: 'R15', if: ['requires_ecg'], then: 'diagnostic_testing', source: 'Clinical pathway' },
    { id: 'R16', if: ['diagnostic_testing'], then: 'cardiology_consultation', source: 'Clinical pathway' },
    { id: 'R17', if: ['high_risk'], then: 'close_monitoring', source: 'AHA' },
    { id: 'R18', if: ['close_monitoring', 'diagnostic_testing'], then: 'specialist_followup', source: 'Clinical pathway' },
    { id: 'R19', if: ['specialist_followup'], then: 'treatment_planning', source: 'Clinical pathway' },
    { id: 'R20', if: ['low_risk'], then: 'preventive_education', source: 'WHO' }
];

// ─────────────────────────────────────────────────────────────────
// FORWARD CHAINING INFERENCE
// ─────────────────────────────────────────────────────────────────
function forwardChaining(initialFacts, rules) {
    const inferred = new Set(initialFacts);
    const trace = [];
    let changed = true;

    while (changed) {
        changed = false;

        for (const rule of rules) {
            const allConditionsMet = rule.if.every(condition => inferred.has(condition));
            
            if (allConditionsMet && !inferred.has(rule.then)) {
                inferred.add(rule.then);
                trace.push({
                    conditions: rule.if,
                    conclusion: rule.then,
                    source: rule.source
                });
                changed = true;
            }
        }
    }

    return { inferred: Array.from(inferred), trace };
}

// ─────────────────────────────────────────────────────────────────
// GENERATE INITIAL FACTS
// ─────────────────────────────────────────────────────────────────
function generateInitialFacts(age, trestbps, chol, exang, oldpeak, riskState, thalch = 150, thalChoice = 1, restecgChoice = 0, cpChoice = 4) {
    const facts = [];

    if (riskState === 'High Risk') facts.push('high_risk');
    else if (riskState === 'Medium Risk') facts.push('medium_risk');
    else facts.push('low_risk');

    if (age >= 60) facts.push('elderly_patient');
    if (trestbps >= 140) facts.push('high_bp');
    if (chol >= 240) facts.push('high_cholesterol');
    if (thalch < 120) facts.push('low_max_hr');
    if (exang === 1) facts.push('exercise_angina');
    if (oldpeak > 2) facts.push('high_oldpeak');
    if (thalChoice === 3) facts.push('reversible_thal');
    if ([1, 2].includes(restecgChoice)) facts.push('ecg_abnormal');
    if (cpChoice === 4) facts.push('silent_ischemia_risk');

    return facts;
}

// ─────────────────────────────────────────────────────────────────
// BUILD MEDICAL GRAPH FOR A* SEARCH
// ─────────────────────────────────────────────────────────────────
function buildMedicalGraph(riskState, trestbps, chol, exang, oldpeak, age) {
    const graph = {};
    graph[riskState] = {};

    if (trestbps >= 140) graph[riskState]['Blood Pressure Management'] = 2;
    if (chol >= 240) graph[riskState]['Cholesterol Management'] = 2;
    if (exang === 1) graph[riskState]['ECG Evaluation'] = 1;
    if (oldpeak > 2) graph[riskState]['Stress Evaluation'] = 1;
    if (age >= 60 || riskState === 'High Risk') graph[riskState]['Cardiology Consultation'] = 1;

    if (riskState === 'Low Risk' && Object.keys(graph[riskState]).length === 0) {
        graph[riskState]['Goal State'] = 1;
    }

    graph['Blood Pressure Management'] = { 'Risk Stratification': 2 };
    graph['Cholesterol Management'] = { 'Risk Stratification': 2 };
    graph['ECG Evaluation'] = { 'Cardiology Consultation': 1 };
    graph['Stress Evaluation'] = { 'Cardiology Consultation': 1 };
    graph['Cardiology Consultation'] = { 'Risk Stratification': 1 };
    graph['Risk Stratification'] = { 'Treatment Planning': 1 };
    graph['Treatment Planning'] = { 'Goal State': 1 };
    graph['Goal State'] = {};

    return graph;
}

// ─────────────────────────────────────────────────────────────────
// MEDICAL DOCUMENTS FOR TF-IDF SEARCH
// ─────────────────────────────────────────────────────────────────
const medicalDocs = [
    {
        title: 'Heart Disease Overview',
        text: 'Heart disease encompasses conditions affecting the heart and blood vessels including coronary artery disease, heart failure, and arrhythmias. Main causes include hypertension, high cholesterol, smoking, and diabetes.'
    },
    {
        title: 'Hypertension & Blood Pressure',
        text: 'High blood pressure (systolic BP >= 140 mmHg) is Stage 2 hypertension per AHA/ACC guidelines. It forces the heart to work harder and is a leading cause of heart attack and stroke. Treatment includes ACE inhibitors, beta-blockers, and lifestyle changes.'
    },
    {
        title: 'Cholesterol & Hyperlipidemia',
        text: 'Total cholesterol >= 240 mg/dl is high risk per NHLBI guidelines. LDL cholesterol causes plaque formation in arteries. Statins are the primary treatment. Diet rich in fruits, vegetables and omega-3 fatty acids reduces cholesterol.'
    },
    {
        title: 'ECG & Electrocardiography',
        text: 'Electrocardiography records heart electrical activity and detects ST-T wave changes, left ventricular hypertrophy, and arrhythmias. Resting ECG abnormalities significantly increase cardiac risk and require further evaluation.'
    },
    {
        title: 'Exercise-Induced Angina',
        text: 'Exercise angina is chest pain during physical activity indicating myocardial ischemia from coronary artery disease. Exercise stress testing evaluates severity. It is a key diagnostic marker requiring immediate cardiology attention.'
    },
    {
        title: 'ST Depression & Oldpeak',
        text: 'ST depression induced by exercise (oldpeak > 2mm) indicates significant myocardial ischemia and abnormal cardiac stress response. Values above 2mm warrant immediate stress testing and cardiology consultation.'
    },
    {
        title: 'Thalassemia & Perfusion',
        text: 'A reversible thalassemia defect on nuclear stress testing indicates stress-induced ischemia where myocardium is at risk but not permanently damaged. This distinguishes viable from non-viable heart tissue.'
    },
    {
        title: 'Maximum Heart Rate',
        text: 'Maximum heart rate = 220 minus age. Achieving less than 85% of predicted maximum during stress testing indicates reduced cardiac reserve and elevated cardiovascular risk requiring specialist evaluation.'
    },
    {
        title: 'Chest Pain Classification',
        text: 'Typical angina is predictable exertional chest pain. Atypical angina has unusual characteristics. Asymptomatic chest pain paradoxically shows highest disease prevalence in clinical studies as disease progresses silently.'
    },
    {
        title: 'Cardiology Consultation',
        text: 'Cardiology consultation is required for high-risk patients, abnormal stress tests, multiple cardiovascular risk factors, or patients over 60. Cardiologist performs comprehensive assessment including echocardiogram and coronary angiography.'
    },
    {
        title: 'Risk Stratification',
        text: 'Cardiac risk stratification uses HEART score, TIMI score, and Framingham Risk Score to estimate 10-year cardiovascular event probability. High-risk patients are prioritised for aggressive medical intervention.'
    },
    {
        title: 'Treatment Planning',
        text: 'Heart disease treatment includes medication management, lifestyle interventions, cardiac rehabilitation, and revascularisation procedures. Plans are personalised based on risk stratification, comorbidities, and patient preferences.'
    },
    {
        title: 'Coronary Artery Disease',
        text: 'Coronary artery disease involves narrowing of coronary arteries due to atherosclerosis. Risk factors include hypertension, hyperlipidemia, diabetes, smoking, and family history. Treatment includes statins, antiplatelets, and revascularisation.'
    },
    {
        title: 'Preventive Cardiology',
        text: 'Preventive cardiology focuses on reducing cardiovascular risk through lifestyle modifications including regular exercise, heart-healthy diet, smoking cessation, weight management, and stress reduction techniques.'
    },
    {
        title: 'Metabolic Syndrome',
        text: 'Metabolic syndrome combines hypertension, high cholesterol, insulin resistance, and obesity. It significantly triples the risk of heart attack. Management requires addressing all components simultaneously through medication and lifestyle changes.'
    }
];

// ─────────────────────────────────────────────────────────────────
// SIMPLE TF-IDF IMPLEMENTATION
// ─────────────────────────────────────────────────────────────────
class TFIDFSearch {
    constructor(documents) {
        this.documents = documents;
        this.vocabulary = new Set();
        this.documentVectors = [];

        // Build vocabulary
        documents.forEach(doc => {
            const words = doc.text.toLowerCase().match(/\b\w+\b/g) || [];
            words.forEach(word => this.vocabulary.add(word));
        });

        // Remove common stop words
        const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being']);
        this.vocabulary = new Set([...this.vocabulary].filter(word => !stopWords.has(word)));

        // Build document vectors
        documents.forEach(doc => {
            const vector = this.getVector(doc.text);
            this.documentVectors.push(vector);
        });
    }

    getVector(text) {
        const vector = {};
        const words = text.toLowerCase().match(/\b\w+\b/g) || [];
        const totalWords = words.length;

        for (const word of this.vocabulary) {
            const count = words.filter(w => w === word).length;
            vector[word] = count / Math.max(totalWords, 1);
        }

        return vector;
    }

    cosineSimilarity(vec1, vec2) {
        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;

        for (const word of this.vocabulary) {
            const v1 = vec1[word] || 0;
            const v2 = vec2[word] || 0;
            dotProduct += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }

        const denominator = Math.sqrt(norm1) * Math.sqrt(norm2);
        return denominator === 0 ? 0 : dotProduct / denominator;
    }

    search(query) {
        const queryVector = this.getVector(query);
        let bestScore = -1;
        let bestIdx = 0;

        for (let i = 0; i < this.documentVectors.length; i++) {
            const score = this.cosineSimilarity(queryVector, this.documentVectors[i]);
            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }

        if (bestScore < 0.05) {
            return null;
        }

        return {
            doc: this.documents[bestIdx],
            score: bestScore
        };
    }
}

// Initialize TF-IDF search
const tfidfSearch = new TFIDFSearch(medicalDocs);

// ─────────────────────────────────────────────────────────────────
// MEDICAL SEARCH FUNCTION
// ─────────────────────────────────────────────────────────────────
function medicalSearch(query) {
    if (!query.trim()) {
        return null;
    }

    const result = tfidfSearch.search(query);

    if (!result) {
        return {
            title: 'No results found',
            text: "Try different keywords like 'cholesterol', 'blood pressure', 'chest pain', 'ECG', 'heart disease'."
        };
    }

    return {
        ...result.doc,
        score: result.score
    };
}

// ─────────────────────────────────────────────────────────────────
// GENERATE RECOMMENDATIONS
// ─────────────────────────────────────────────────────────────────
function generateRecommendations(trestbps, chol, exang, oldpeak, age, thalch, riskState) {
    const recs = [];

    if (trestbps >= 140) recs.push('🔴 Blood pressure management — antihypertensive therapy recommended');
    if (chol >= 240) recs.push('🔴 Cholesterol management — statin therapy and dietary intervention');
    if (exang === 1) recs.push('🟠 ECG evaluation — exercise-induced angina warrants 12-lead ECG');
    if (oldpeak > 2.0) recs.push('🟠 Stress test evaluation — ST depression indicates cardiac stress');
    if (age >= 60) recs.push('🟡 Cardiology consultation — age-related risk requires specialist review');
    if (thalch < 120) recs.push('🟠 Cardiac reserve assessment — low max heart rate indicates reduced function');

    if (riskState === 'High Risk') {
        recs.push('🚨 URGENT: Schedule cardiologist appointment within 7 days');
    } else if (riskState === 'Medium Risk') {
        recs.push('📅 Schedule follow-up within 4-6 weeks');
    } else {
        recs.push('✅ Continue preventive care — annual cardiac check-up recommended');
    }

    return recs.length > 0 ? recs.join('\n') : '✅ No immediate interventions required. Maintain healthy lifestyle.';
}
