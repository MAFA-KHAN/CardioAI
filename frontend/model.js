/* ─────────────────────────────────────────────────────────────────── */
/* CARDIOAI — ML MODEL SIMULATION */
/* Simulates XGBoost model behavior */
/* ─────────────────────────────────────────────────────────────────── */

class MLModel {
    constructor() {
        this.featureWeights = {
            'age': 0.08,
            'sex': 0.04,
            'trestbps': 0.12,
            'chol': 0.10,
            'fbs': 0.05,
            'thalch': -0.15,
            'exang': 0.18,
            'oldpeak': 0.22,
            'cp_atypical': 0.08,
            'cp_non_anginal': 0.06,
            'restecg_st': 0.14,
            'slope_flat': 0.08,
            'thal_reversible': 0.16
        };
    }

    // Normalize features
    scaleFeatures(features) {
        return {
            age: features.age / 80.0,
            sex: features.sex,
            trestbps: (features.trestbps - 80) / 120.0,
            chol: (features.chol - 100) / 300.0,
            fbs: features.fbs,
            thalch: (features.thalch - 60) / 160.0,
            exang: features.exang,
            oldpeak: Math.min(features.oldpeak / 6.2, 1.0),
            cp_atypical: features.cp === 2 ? 1 : 0,
            cp_non_anginal: features.cp === 3 ? 1 : 0,
            restecg_st: features.restecg === 1 ? 1 : 0,
            slope_flat: features.slope === 2 ? 1 : 0,
            thal_reversible: features.thal === 3 ? 1 : 0
        };
    }

    // Compute interaction terms and derived features
    engineerFeatures(scaled) {
        return {
            ...scaled,
            bp_chol_interaction: scaled.trestbps * scaled.chol,
            exercise_stress: Math.max(0, (scaled.exang + scaled.oldpeak) / 2.0),
            age_risk_group: scaled.age >= 0.75 ? 1 : (scaled.age >= 0.65 ? 0.75 : (scaled.age >= 0.55 ? 0.5 : 0)),
            thalch_age_ratio: scaled.thalch / Math.max(scaled.age, 0.1),
            ecg_stress_interaction: (scaled.restecg_st + scaled.oldpeak) / 2.0
        };
    }

    // Predict probability
    predictProba(features) {
        const scaled = this.scaleFeatures(features);
        const engineered = this.engineerFeatures(scaled);

        // Calculate raw score with weighted sum and non-linearities
        let score = -0.5; // Base bias

        score += engineered.age * this.featureWeights.age;
        score += engineered.sex * this.featureWeights.sex;
        score += engineered.trestbps * this.featureWeights.trestbps;
        score += engineered.chol * this.featureWeights.chol;
        score += engineered.fbs * this.featureWeights.fbs;
        score += engineered.thalch * this.featureWeights.thalch; // Negative weight
        score += engineered.exang * this.featureWeights.exang;
        score += engineered.oldpeak * this.featureWeights.oldpeak;
        score += engineered.cp_atypical * this.featureWeights.cp_atypical;
        score += engineered.cp_non_anginal * this.featureWeights.cp_non_anginal;
        score += engineered.restecg_st * this.featureWeights.restecg_st;
        score += engineered.slope_flat * this.featureWeights.slope_flat;
        score += engineered.thal_reversible * this.featureWeights.thal_reversible;

        // Add interaction term
        score += engineered.bp_chol_interaction * 0.06;
        score += engineered.exercise_stress * 0.12;
        score += engineered.age_risk_group * 0.08;

        // Sigmoid function to convert to probability
        const probability = 1.0 / (1.0 + Math.exp(-score));

        // Clip to [0.05, 0.95]
        const clipped = Math.max(0.05, Math.min(0.95, probability));

        return {
            probability: clipped,
            score: score,
            classProbabilities: [1 - clipped, clipped]
        };
    }

    // Predict class (binary: 0 or 1)
    predict(features) {
        const result = this.predictProba(features);
        return result.probability >= 0.35 ? 1 : 0; // Threshold: 0.35 for medical recall
    }
}

// Initialize global model
const model = new MLModel();

// ─────────────────────────────────────────────────────────────────
// UTILITY FUNCTIONS
// ─────────────────────────────────────────────────────────────────

// Determine risk level from probability
function determineRisk(probability) {
    if (probability >= 0.80) return 'High Risk';
    if (probability >= 0.40) return 'Medium Risk';
    return 'Low Risk';
}

// Get risk emoji
function getRiskEmoji(riskState) {
    if (riskState === 'High Risk') return '🔴';
    if (riskState === 'Medium Risk') return '🟡';
    return '🟢';
}

// ─────────────────────────────────────────────────────────────────
// SENSITIVITY ANALYSIS
// ─────────────────────────────────────────────────────────────────
function performSensitivityAnalysis(baseFeatures) {
    const baseResult = model.predictProba(baseFeatures);
    const baseProb = baseResult.probability;

    const sensitivity = {};

    // Test: Lower BP to 120
    if (baseFeatures.trestbps > 120) {
        const modified = { ...baseFeatures, trestbps: 120 };
        const newProb = model.predictProba(modified).probability;
        sensitivity['Lower BP to 120 mmHg'] = ((newProb - baseProb) * 100).toFixed(1);
    }

    // Test: Lower cholesterol to 180
    if (baseFeatures.chol > 180) {
        const modified = { ...baseFeatures, chol: 180 };
        const newProb = model.predictProba(modified).probability;
        sensitivity['Lower Cholesterol to 180'] = ((newProb - baseProb) * 100).toFixed(1);
    }

    // Test: Increase max HR
    if (baseFeatures.thalch < 180) {
        const modified = { ...baseFeatures, thalch: Math.min(202, baseFeatures.thalch + 15) };
        const newProb = model.predictProba(modified).probability;
        sensitivity['Increase Max HR by 15'] = ((newProb - baseProb) * 100).toFixed(1);
    }

    // Test: Reduce ST depression
    if (baseFeatures.oldpeak > 0) {
        const modified = { ...baseFeatures, oldpeak: 0 };
        const newProb = model.predictProba(modified).probability;
        sensitivity['Reduce ST Depression to 0'] = ((newProb - baseProb) * 100).toFixed(1);
    }

    // Test: Remove exercise angina
    if (baseFeatures.exang === 1) {
        const modified = { ...baseFeatures, exang: 0 };
        const newProb = model.predictProba(modified).probability;
        sensitivity['Remove Exercise Angina'] = ((newProb - baseProb) * 100).toFixed(1);
    }

    // Test: Normal thalassemia
    if (baseFeatures.thal !== 1) {
        const modified = { ...baseFeatures, thal: 1 };
        const newProb = model.predictProba(modified).probability;
        sensitivity['Normal Thalassemia'] = ((newProb - baseProb) * 100).toFixed(1);
    }

    return sensitivity;
}

// ─────────────────────────────────────────────────────────────────
// FORMAT SENSITIVITY RESULTS
// ─────────────────────────────────────────────────────────────────
function formatSensitivityAnalysis(sensitivity) {
    const lines = ['What-If Sensitivity Analysis\n(Negative = risk decreases)\n'];

    // Sort by value
    const sorted = Object.entries(sensitivity)
        .map(([feat, delta]) => ({ feat, delta: parseFloat(delta) }))
        .sort((a, b) => a.delta - b.delta);

    for (const { feat, delta } of sorted) {
        const arrow = delta < 0 ? '▼' : (delta > 0 ? '▲' : '—');
        const barLength = Math.round(Math.abs(delta) / 2);
        const bar = '█'.repeat(barLength) || '-';
        const sign = delta > 0 ? '+' : '';
        lines.push(`${arrow} ${feat}: ${sign}${delta}%  ${bar}`);
    }

    return lines.join('\n');
}
