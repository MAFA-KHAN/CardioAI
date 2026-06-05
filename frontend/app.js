/* ─────────────────────────────────────────────────────────────────── */
/* CARDIOAI — MAIN APPLICATION */
/* UI Logic, Event Handlers, Analysis Functions */
/* ─────────────────────────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

// ─────────────────────────────────────────────────────────────────
// INITIALIZE APP
// ─────────────────────────────────────────────────────────────────
function initializeApp() {
    setupTabNavigation();
    prefillSimulatorWithAssessment();
}

// ─────────────────────────────────────────────────────────────────
// TAB NAVIGATION
// ─────────────────────────────────────────────────────────────────
window.switchTab = function(tabId) {
    const panes = document.querySelectorAll('.tab-pane');
    const navItems = document.querySelectorAll('.nav-item');
    
    // Hide all panes and remove active states
    panes.forEach(p => p.classList.remove('active'));
    navItems.forEach(b => b.classList.remove('active'));

    // Show target pane
    const targetPane = document.getElementById(tabId);
    if (targetPane) {
        targetPane.classList.add('active');
    }

    // Highlight corresponding nav item if it exists
    const targetNav = document.querySelector(`.nav-item[data-tab="${tabId}"]`);
    if (targetNav) {
        targetNav.classList.add('active');
    }
};

window.scrollToTop = function() {
    window.switchTab('home');
    setTimeout(() => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }, 50);
};

window.scrollToSection = function(sectionId) {
    // Ensure we are on the Home tab first
    window.switchTab('home');
    
    // Small delay to allow DOM to render if home tab was hidden
    setTimeout(() => {
        const section = document.getElementById(sectionId);
        if (section) {
            section.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }, 50);
};

function setupTabNavigation() {
    // Only attach tab logic to buttons that actually have a data-tab attribute
    const buttons = document.querySelectorAll('[data-tab]');

    buttons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            if (tabId) {
                window.switchTab(tabId);
            }
        });
    });
}

// ─────────────────────────────────────────────────────────────────
// HELPER: GET FORM VALUES
// ─────────────────────────────────────────────────────────────────
function getAssessmentValues() {
    return {
        age: parseInt(document.getElementById('age').value),
        sex: parseInt(document.getElementById('sex').value),
        cp: parseInt(document.getElementById('cp').value),
        trestbps: parseInt(document.getElementById('trestbps').value),
        chol: parseInt(document.getElementById('chol').value),
        fbs: 0, // Not present in UI, defaulting to 0
        restecg: parseInt(document.getElementById('restecg').value),
        thalch: parseInt(document.getElementById('thalch').value),
        exang: parseInt(document.getElementById('exang').value),
        oldpeak: parseFloat(document.getElementById('oldpeak').value),
        slope: parseInt(document.getElementById('slope').value),
        thal: parseInt(document.getElementById('thal').value)
    };
}

function getSimulatorValues() {
    return {
        age: parseInt(document.getElementById('sim-age').value),
        sex: parseInt(document.getElementById('sim-sex').value),
        cp: parseInt(document.getElementById('sim-cp').value),
        trestbps: parseInt(document.getElementById('sim-bp').value),
        chol: parseInt(document.getElementById('sim-chol').value),
        fbs: 0,
        restecg: parseInt(document.getElementById('sim-ecg').value),
        thalch: parseInt(document.getElementById('sim-hr').value),
        exang: parseInt(document.getElementById('sim-exang').value),
        oldpeak: parseFloat(document.getElementById('sim-op').value),
        slope: parseInt(document.getElementById('sim-slope').value),
        thal: parseInt(document.getElementById('sim-thal').value)
    };
}

// ─────────────────────────────────────────────────────────────────
// ANALYZE PATIENT — Full AI Pipeline (ML + A* + KB)
// ─────────────────────────────────────────────────────────────────
async function analyzePatient() {
    const values = getAssessmentValues();

    // Hide previous results, show loader
    document.getElementById('ml-status').style.display = 'block';
    ['ml-output','astar-output','recs-output','kb-output'].forEach(id => {
        document.getElementById(id).style.display = 'none';
    });

    let data;
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(values)
        });
        data = await response.json();
        if (!data.success) {
            console.error('Backend error:', data.error);
            alert('Error from ML backend. See console.');
            document.getElementById('ml-status').style.display = 'none';
            return;
        }
    } catch (err) {
        console.error('Failed to connect to ML backend:', err);
        alert('Could not connect to ML backend. Is server.py running?');
        document.getElementById('ml-status').style.display = 'none';
        return;
    }

    document.getElementById('ml-status').style.display = 'none';

    // ── Panel 1: ML Risk Prediction ──────────────────────────────
    const prob = data.probability;
    const riskState = data.risk_state;
    const riskColor = riskState === 'High Risk' ? '#DC2626' : riskState === 'Medium Risk' ? '#D97706' : '#16A34A';
    const diagnosisLabel = data.prediction === 1 ? 'Consultation Recommended' : 'Routine Health Outlook';

    document.getElementById('ml-result').innerHTML = `
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:20px; padding:16px; background:var(--accent-green-light); border-radius:16px;">
            <div style="font-size:40px; font-weight:800; color:${riskColor};">${(prob*100).toFixed(1)}%</div>
            <div>
                <div style="font-weight:700; font-size:18px; color:var(--text-primary);">${diagnosisLabel}</div>
                <div style="font-size:14px; color:${riskColor}; font-weight:600;">${riskState}</div>
                <div style="font-size:12px; color:var(--text-secondary); margin-top:4px;">XGBoost | AUC: 0.898 | Recall: 0.931 | Threshold: 0.35</div>
            </div>
        </div>
    `;
    document.getElementById('ml-output').style.display = 'block';

    // ── Panel 2: A* Optimal Care Pathway ─────────────────────────
    const pathway = data.pathway || [];
    const cost = data.pathway_cost || 0;
    const pathSteps = pathway.map((step, i) => {
        const isFirst = i === 0;
        const isLast = i === pathway.length - 1;
        const color = isFirst ? riskColor : isLast ? '#16A34A' : 'var(--accent-green)';
        return `
            <div style="display:flex; align-items:center; gap:8px; margin:8px 0;">
                ${i > 0 ? '<div style="color:var(--text-secondary); font-size:18px; margin-left:8px;">↓</div>' : ''}
            </div>
            <div style="display:flex; align-items:center; gap:12px; padding:10px 16px; border:1px solid ${color}30; border-left:3px solid ${color}; border-radius:10px; background:${color}08;">
                <div style="font-weight:600; color:var(--text-primary);">${step}</div>
            </div>
        `;
    }).join('');

    document.getElementById('astar-result').innerHTML = `
        <div style="font-size:13px; color:var(--text-secondary); margin-bottom:16px;">Total Cost: ${cost} steps | Algorithm: A* Search with admissible heuristic f(n) = g(n) + h(n)</div>
        ${pathSteps}
    `;
    document.getElementById('astar-output').style.display = 'block';

    // ── Panel 3: Clinical Recommendations ────────────────────────
    const recs = data.recommendations || [];
    const recIcons = { 'URGENT': '🚨', 'Blood pressure': '🔴', 'Cholesterol': '🔴', 'ECG': '🟠', 'Stress': '🟠', 'Cardiology': '🟡', 'Cardiac': '🟠', 'Schedule': '📅', 'Continue': '✅' };
    const recsHtml = recs.map(rec => {
        const icon = Object.entries(recIcons).find(([k]) => rec.includes(k))?.[1] || '📋';
        return `<div style="display:flex; gap:12px; align-items:flex-start; padding:10px 0; border-bottom:1px solid var(--border-color);">
            <span style="font-size:18px;">${icon}</span>
            <span style="color:var(--text-secondary); font-size:14px; line-height:1.5;">${rec}</span>
        </div>`;
    }).join('');
    document.getElementById('recs-result').innerHTML = recsHtml || '<div style="color:var(--text-secondary);">No immediate interventions required.</div>';
    document.getElementById('recs-output').style.display = 'block';

    // ── Panel 4: Knowledge Base Inference ────────────────────────
    const initFacts = data.initial_facts || [];
    const derivedFacts = data.derived_facts || [];
    const kbTrace = data.kb_trace || [];

    const initHtml = initFacts.map(f => `<span style="display:inline-block; background:var(--accent-green-light); color:var(--accent-green); padding:3px 10px; border-radius:20px; font-size:12px; margin:3px;">${f}</span>`).join('');
    const derivedHtml = derivedFacts.map(f => `<span style="display:inline-block; background:#FEF3C7; color:#92400E; padding:3px 10px; border-radius:20px; font-size:12px; margin:3px;">${f}</span>`).join('');
    const traceHtml = kbTrace.map(t => `
        <div style="padding:8px 12px; margin:6px 0; background:var(--bg-main); border-radius:10px; border-left:3px solid var(--accent-green);">
            <div style="font-size:13px; color:var(--text-primary); font-weight:600;">[${t.conditions.join(' + ')}] &rarr; <span style="color:var(--accent-green);">${t.conclusion}</span></div>
            <div style="font-size:11px; color:var(--text-secondary); margin-top:3px;">Source: ${t.source}</div>
        </div>
    `).join('');

    document.getElementById('kb-result').innerHTML = `
        <div style="margin-bottom:20px;">
            <div style="font-weight:600; margin-bottom:8px; color:var(--text-primary);">Initial Facts (${initFacts.length})</div>
            <div>${initHtml || '<span style="color:var(--text-secondary);">None</span>'}</div>
        </div>
        <div style="margin-bottom:20px;">
            <div style="font-weight:600; margin-bottom:8px; color:var(--text-primary);">Rules Fired (${kbTrace.length})</div>
            <div>${traceHtml || '<div style="color:var(--text-secondary);">No rules triggered.</div>'}</div>
        </div>
        <div>
            <div style="font-weight:600; margin-bottom:8px; color:var(--text-primary);">Derived Conclusions (${derivedFacts.length})</div>
            <div>${derivedHtml || '<span style="color:var(--text-secondary);">No additional facts derived.</span>'}</div>
        </div>
    `;
    document.getElementById('kb-output').style.display = 'block';

    // Reinitialize icons for dynamically added content
    if (window.lucide) lucide.createIcons();

    // Prefill simulator
    prefillSimulatorWithAssessment();
}

// ─────────────────────────────────────────────────────────────────
// PREFILL SIMULATOR
// ─────────────────────────────────────────────────────────────────
function prefillSimulatorWithAssessment() {
    const values = getAssessmentValues();
    document.getElementById('sim-age').value = values.age;
    document.getElementById('sim-sex').value = values.sex;
    document.getElementById('sim-cp').value = values.cp;
    document.getElementById('sim-bp').value = values.trestbps;
    document.getElementById('sim-chol').value = values.chol;
    document.getElementById('sim-hr').value = values.thalch;
    document.getElementById('sim-op').value = values.oldpeak;
    document.getElementById('sim-exang').value = values.exang;
    document.getElementById('sim-thal').value = values.thal;
    document.getElementById('sim-ecg').value = values.restecg;
    document.getElementById('sim-slope').value = values.slope;
}

// ─────────────────────────────────────────────────────────────────
// HUMAN RISK ICON + RING GAUGE
// ─────────────────────────────────────────────────────────────────
function setHumanRisk(prob) {
    const el    = document.getElementById('human-risk-container');
    const label = document.getElementById('human-risk-label');
    const arc   = document.getElementById('risk-ring-arc');
    const pctTxt= document.getElementById('risk-ring-pct');
    if (!el) return;

    // Animate ring (circumference = 2π×50 ≈ 314)
    const circ = 314;
    const filled = circ * prob;
    if (arc) {
        arc.setAttribute('stroke-dasharray', `${filled} ${circ - filled}`);
    }
    if (pctTxt) pctTxt.textContent = (prob * 100).toFixed(0) + '%';

    // Color ring + human icon
    el.className = 'human-risk-wrapper';
    let riskLabel = '', riskColor = '';
    if (prob < 0.25) {
        el.classList.add('human-low');
        riskLabel = '🟢 LOW RISK — Healthy Profile';
        riskColor = '#16A34A';
        if (arc) arc.setAttribute('stroke', '#16A34A');
        if (pctTxt) pctTxt.setAttribute('fill', '#16A34A');
    } else if (prob < 0.55) {
        el.classList.add('human-medium');
        riskLabel = '🟡 MODERATE RISK — Monitor Closely';
        riskColor = '#D97706';
        if (arc) arc.setAttribute('stroke', '#D97706');
        if (pctTxt) pctTxt.setAttribute('fill', '#D97706');
    } else if (prob < 0.80) {
        el.classList.add('human-high');
        riskLabel = '🔴 HIGH RISK — Urgent Evaluation';
        riskColor = '#DC2626';
        if (arc) arc.setAttribute('stroke', '#DC2626');
        if (pctTxt) pctTxt.setAttribute('fill', '#DC2626');
    } else {
        el.classList.add('human-critical');
        riskLabel = '☠ CRITICAL — Immediate Intervention';
        riskColor = '#ff4444';
        if (arc) arc.setAttribute('stroke', '#ff4444');
        if (pctTxt) pctTxt.setAttribute('fill', '#ff4444');
    }
    if (label) { label.textContent = riskLabel; label.style.color = riskColor; }
}

// ─────────────────────────────────────────────────────────────────
// DEMO PROFILE LOADER
// ─────────────────────────────────────────────────────────────────
const DEMO_PROFILES = {
    low: {
        age: 42, sex: '0', cp: '2', trestbps: 118, chol: 180,
        thalch: 172, oldpeak: 0.0, exang: '0', slope: '1', thal: '1', restecg: '0'
    },
    medium: {
        age: 55, sex: '1', cp: '3', trestbps: 138, chol: 245,
        thalch: 145, oldpeak: 1.2, exang: '0', slope: '2', thal: '2', restecg: '1'
    },
    high: {
        age: 63, sex: '1', cp: '4', trestbps: 165, chol: 320,
        thalch: 118, oldpeak: 3.5, exang: '1', slope: '3', thal: '3', restecg: '2'
    },
    critical: {
        age: 72, sex: '1', cp: '4', trestbps: 185, chol: 380,
        thalch: 95, oldpeak: 5.0, exang: '1', slope: '3', thal: '3', restecg: '2'
    }
};

window.loadProfile = function(type) {
    const p = DEMO_PROFILES[type];
    if (!p) return;

    document.getElementById('sim-age').value      = p.age;
    document.getElementById('sim-sex').value      = p.sex;
    document.getElementById('sim-cp').value       = p.cp;
    document.getElementById('sim-bp').value       = p.trestbps;
    document.getElementById('sim-chol').value     = p.chol;
    document.getElementById('sim-hr').value       = p.thalch;
    document.getElementById('sim-op').value       = p.oldpeak;
    document.getElementById('sim-exang').value    = p.exang;
    document.getElementById('sim-slope').value    = p.slope;
    document.getElementById('sim-thal').value     = p.thal;
    document.getElementById('sim-ecg').value      = p.restecg;

    // Highlight active profile tab
    ['low','medium','high','critical'].forEach(t => {
        const btn = document.getElementById(`ptab-${t}`);
        if (!btn) return;
        btn.classList.remove(`ptab-active-${t}`);
    });
    const activeBtn = document.getElementById(`ptab-${type}`);
    if (activeBtn) activeBtn.classList.add(`ptab-active-${type}`);

    // Reset icon to idle
    const el = document.getElementById('human-risk-container');
    if (el) el.className = 'human-risk-wrapper human-idle';
    const label = document.getElementById('human-risk-label');
    if (label) { label.textContent = 'CALCULATING...'; label.style.color = ''; }
    const arc = document.getElementById('risk-ring-arc');
    if (arc) { arc.setAttribute('stroke-dasharray','0 314'); arc.setAttribute('stroke','#CBD5E1'); }
    const pctTxt = document.getElementById('risk-ring-pct');
    if (pctTxt) { pctTxt.textContent = '...'; }

    runSimulator();
};

// ─────────────────────────────────────────────────────────────────
// RUN SIMULATOR
// ─────────────────────────────────────────────────────────────────
async function runSimulator() {
    const values = getSimulatorValues();

    // Show Loader
    document.getElementById('sim-status').style.display = 'block';
    document.getElementById('sim-prob-output').style.display = 'none';
    document.getElementById('sim-delta-output').style.display = 'none';

    try {
        const response = await fetch('http://localhost:5000/simulate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(values)
        });
        
        const data = await response.json();
        document.getElementById('sim-status').style.display = 'none';

        if (data.success) {
            const prob = data.base_probability;
            const pct = (prob * 100).toFixed(1);
            const riskState = determineRisk(prob);
            const riskEmoji = getRiskEmoji(riskState);
            const pred = prob >= 0.35 ? 'Elevated Risk' : 'Routine Status';

            // Animate human icon + ring
            setHumanRisk(prob);

            // Populate structured outlook rows
            document.getElementById('sim-prob-output').style.display = 'block';
            const statusEl = document.getElementById('sim-outlook-status');
            const pctEl    = document.getElementById('sim-outlook-pct');
            const catEl    = document.getElementById('sim-outlook-cat');
            if (statusEl) statusEl.textContent = pred;
            if (pctEl)    { pctEl.textContent = pct + '%'; pctEl.style.color = prob >= 0.55 ? '#DC2626' : prob >= 0.35 ? '#D97706' : '#16A34A'; }
            if (catEl)    catEl.textContent = `${riskEmoji} ${riskState}`;

            // Also keep sim-prob-result if it exists (fallback)
            const probResultEl = document.getElementById('sim-prob-result');
            if (probResultEl) probResultEl.textContent = '';

            // Hide empty state
            const emptyEl = document.getElementById('sim-empty-state');
            if (emptyEl) emptyEl.style.display = 'none';

            // Build visual driver bars
            const sensitivity = data.sensitivity;
            const sorted = Object.entries(sensitivity)
                .map(([feat, delta]) => ({ feat, delta: parseFloat(delta) }))
                .sort((a, b) => a.delta - b.delta);

            const maxAbs = Math.max(...sorted.map(s => Math.abs(s.delta)), 1);
            let barsHTML = '';
            for (const { feat, delta } of sorted) {
                const isGood = delta <= 0;
                const pctWidth = Math.min(100, (Math.abs(delta) / maxAbs) * 100).toFixed(1);
                const sign = delta > 0 ? '+' : '';
                const badgeClass = isGood ? 'driver-delta-good' : 'driver-delta-bad';
                const barClass   = isGood ? 'driver-bar-good'   : 'driver-bar-bad';
                barsHTML += `
                <div class="driver-row">
                    <div class="driver-label-row">
                        <span>${feat}</span>
                        <span class="driver-delta-badge ${badgeClass}">${sign}${delta}%</span>
                    </div>
                    <div class="driver-bar-track">
                        <div class="driver-bar-fill ${barClass}" style="width:0%" data-width="${pctWidth}"></div>
                    </div>
                </div>`;
            }
            const driversContainer = document.getElementById('sim-delta-result');
            if (driversContainer) driversContainer.innerHTML = barsHTML;
            document.getElementById('sim-delta-output').style.display = 'block';

            // Animate bars after paint
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    document.querySelectorAll('.driver-bar-fill[data-width]').forEach(bar => {
                        bar.style.width = bar.getAttribute('data-width') + '%';
                    });
                });
            });

            if (window.lucide) lucide.createIcons();

        } else {
            console.error("Backend error:", data.error);
            alert("Error from ML backend. See console.");
        }
    } catch (err) {
        console.error("Failed to connect to ML backend:", err);
        alert("Could not connect to ML backend. Is server.py running?");
        document.getElementById('sim-status').style.display = 'none';
    }
}

// ─────────────────────────────────────────────────────────────────
// CLINICAL KNOWLEDGE BASE — TF-IDF Search (from cardioai_gradio.py)
// ─────────────────────────────────────────────────────────────────
window.searchMedicalKB = function() {
    const query = document.getElementById('knowledge-query').value;
    if (!query.trim()) {
        alert('Please enter a search term');
        return;
    }
    renderKBResult(query);
};

window.quickSearch = function(term) {
    document.getElementById('knowledge-query').value = term;
    renderKBResult(term);
};

function renderKBResult(query) {
    const result = medicalSearch(query);
    const container = document.getElementById('knowledge-results');

    if (!result || !result.title) {
        container.innerHTML = `
            <div class="glass-card" style="text-align:center; padding:40px;">
                <i data-lucide="search-x" style="color:var(--text-secondary); width:40px; height:40px; margin-bottom:16px;"></i>
                <p style="color:var(--text-secondary);">No results found. Try: <em>cholesterol</em>, <em>blood pressure</em>, <em>chest pain</em>, <em>ECG</em>.</p>
            </div>`;
        if (window.lucide) lucide.createIcons();
        return;
    }

    const relevanceScore = result.score ? (result.score * 100).toFixed(1) : 'N/A';
    const relevanceColor = parseFloat(relevanceScore) > 50 ? 'var(--accent-green)' : '#D97706';

    container.innerHTML = `
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:16px;">
                <h2 class="card-title" style="margin:0;"><i data-lucide="book-open" class="text-green"></i> ${result.title}</h2>
                <span style="background:var(--accent-green-light); color:${relevanceColor}; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:700; white-space:nowrap;">Relevance: ${relevanceScore}%</span>
            </div>
            <p style="color:var(--text-secondary); font-size:15px; line-height:1.8;">${result.text}</p>
            <div style="margin-top:16px; padding:12px; background:var(--bg-main); border-radius:12px; font-size:12px; color:var(--text-secondary);">
                <i data-lucide="info" style="width:12px; height:12px; vertical-align:middle;"></i>
                Search powered by TF-IDF cosine similarity across 15 AHA/ACC/NHLBI/Mayo Clinic clinical documents.
            </div>
        </div>
    `;
    if (window.lucide) lucide.createIcons();
}

// ─────────────────────────────────────────────────────────────────
// KEYBOARD SHORTCUTS
// ─────────────────────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to run analysis
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (document.activeElement.closest('#assess')) {
            analyzePatient();
        }
    }
});
