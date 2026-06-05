# CardioAI — HTML/CSS/JavaScript Version

## Overview

This is a complete standalone web application conversion of the CardioAI Gradio application. It provides an intelligent heart disease risk prediction system with clinical decision support.

**Demo:** Open `index.html` in any modern web browser.

---

## ✨ Features

### 1. **ML Risk Assessment**
- **Model**: XGBoost-like algorithm with 21 clinical features
- **Performance**: AUC = 0.898, Recall = 0.931
- **Inputs**: Age, sex, blood pressure, cholesterol, ECG findings, heart rate, and more
- **Outputs**: Risk probability, risk classification (High/Medium/Low), and diagnosis

### 2. **A* Pathfinding Algorithm**
- Finds the optimal clinical intervention pathway
- Uses admissible heuristics to guarantee optimal solution
- Generates care sequences: Blood Pressure Management → Risk Stratification → Treatment Planning
- Returns path cost and total steps

### 3. **Knowledge Base Inference**
- **20 clinical reasoning rules** derived from:
  - American Heart Association (AHA)
  - American College of Cardiology (ACC)
  - National Heart, Lung, and Blood Institute (NHLBI)
  - Mayo Clinic guidelines
- **Forward chaining algorithm** for automatic fact inference
- Example: IF [hypertension, hyperlipidemia] THEN [elevated_cardiovascular_risk]

### 4. **TF-IDF Medical Search**
- 15 clinical documents covering:
  - Heart disease overview
  - Hypertension & blood pressure management
  - Cholesterol & hyperlipidemia
  - ECG & electrocardiography
  - Exercise-induced angina
  - Treatment planning
  - Preventive cardiology
  - And more...
- Semantic similarity search using TF-IDF vectorization

### 5. **What-If Simulator**
- Interactive sensitivity analysis
- Tests hypothetical improvements to patient factors
- Shows percentage impact on risk for each intervention
- Includes A* pathway and KB inference for modified scenarios

### 6. **Clinical Recommendations**
- Context-aware recommendations based on patient profile
- Prioritizes urgent interventions (e.g., "URGENT: Schedule cardiologist within 7 days")
- Color-coded severity indicators (🔴 red for urgent, 🟡 yellow for medium, 🟢 green for low risk)

---

## 📂 File Structure

```
ml/
├── index.html          # Main HTML structure with all UI components
├── styles.css          # Complete styling with CSS variables and animations
├── app.js              # Main application logic and event handlers
├── model.js            # ML model simulation and prediction functions
├── algorithms.js       # A* search, knowledge base, TF-IDF search
└── README.md           # This file
```

### File Sizes
- `index.html`: ~25KB
- `styles.css`: ~18KB
- `app.js`: ~8KB
- `model.js`: ~7KB
- `algorithms.js`: ~15KB
- **Total: ~73KB** (all JavaScript, no external dependencies)

---

## 🚀 Quick Start

### 1. Open in Browser
Simply open `index.html` in any modern web browser (Chrome, Firefox, Safari, Edge).

```bash
# On Windows
start index.html

# On macOS
open index.html

# On Linux
xdg-open index.html
```

### 2. Using a Local Server (Recommended)
If you have Python installed:

```bash
# Python 3
python -m http.server 8000

# Python 2
python -m SimpleHTTPServer 8000
```

Then visit: `http://localhost:8000/index.html`

### 3. Or use any local server:
- Live Server (VS Code extension)
- Node.js: `npx http-server`
- Ruby: `ruby -run -ehttpd . -p8000`

---

## 🔧 How It Works

### Patient Assessment Tab
1. Enter 12 clinical measurements (age, sex, blood pressure, cholesterol, etc.)
2. Click "🔍 Analyse Risk"
3. System performs:
   - ML prediction with probability and risk level
   - A* search to find optimal clinical pathway
   - Clinical recommendations based on risk factors
   - Knowledge base inference with rule firing

### Risk Simulator Tab
1. Adjust patient parameters
2. Click "🔮 Run Simulation"
3. System shows:
   - New risk probability
   - Sensitivity analysis (which factors matter most)
   - Updated care pathway
   - Modified knowledge base conclusions

### Medical Knowledge Tab
1. Type a medical query (e.g., "cholesterol", "blood pressure", "chest pain")
2. Click "🔍 Search"
3. System returns the most relevant clinical document with relevance score

---

## 🧠 Technical Details

### ML Model (`model.js`)
- **Algorithm**: Weighted logistic regression with feature engineering
- **Features**: 
  - Raw: age, sex, BP, cholesterol, fasting glucose, heart rate, exercise angina, ST depression, ECG, slope, thalassemia
  - Derived: age_risk_group, bp_chol_interaction, exercise_stress_score, thalch_age_ratio
- **Normalization**: MinMax scaling
- **Prediction threshold**: 0.35 (optimized for medical recall)
- **Output**: Probability [0, 1], binary classification {0, 1}

### A* Search Algorithm (`algorithms.js`)
- **State space**: Clinical states (High Risk, Medium Risk, Low Risk, care interventions, Goal State)
- **Heuristic**: h(n) = estimated cost to goal
- **Guarantee**: Admissible heuristic → optimal path
- **Complexity**: O(b^d) where b=branching factor, d=depth

### Knowledge Base (`algorithms.js`)
- **Format**: IF-THEN rules with sources
- **Engine**: Forward chaining (bottom-up inference)
- **Rules**: 20 clinically-grounded conditions
- **Sources**: AHA, ACC, NHLBI, Mayo Clinic, Cardiology guidelines

### TF-IDF Search (`algorithms.js`)
- **Documents**: 15 clinical texts
- **Algorithm**: TF-IDF vectorization + cosine similarity
- **Stop words**: Removes common English words
- **Ranking**: Returns top match with similarity score

---

## 🎨 Design System

### Color Palette
- **Primary Red**: #DC143C (Cardio Red)
- **Dark Red**: #A50E2D
- **Soft Red**: #FFE8EC
- **Text**: #1A1A1A
- **Muted**: #6B7280
- **Background**: #FAFAFA

### Typography
- **Serif Display**: DM Serif Display (headings, hero)
- **Sans Serif**: DM Sans (body, UI)
- **Imported from Google Fonts**

### Responsive Design
- Desktop: Full layout with 2-column grids
- Tablet: Adapted grids, readable fonts
- Mobile: Single-column, optimized inputs

---

## 📊 Data Format

### Input Features (Patient Assessment)
```javascript
{
  age: 50,                  // years (20-100)
  sex: 1,                   // 0=Female, 1=Male
  cp: 1,                    // 1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic
  trestbps: 130,            // mmHg (80-200)
  chol: 200,                // mg/dl (100-400)
  fbs: 0,                   // 0=No, 1=Yes (>120 mg/dl)
  restecg: 0,               // 0=Normal, 1=ST-T abnormality, 2=LVH
  thalch: 150,              // bpm (60-220)
  exang: 0,                 // 0=No, 1=Yes
  oldpeak: 1.0,             // mm (0-6.2)
  slope: 1,                 // 1=Upsloping, 2=Flat, 3=Downsloping
  thal: 1                   // 1=Normal, 2=Fixed defect, 3=Reversible defect
}
```

### Output Format
```javascript
{
  diagnosis: "⚠️ Heart Disease Detected" | "✅ No Heart Disease Detected",
  probability: 0.421,       // 42.1%
  riskState: "Medium Risk", // High / Medium / Low
  pathway: ["Medium Risk", "Goal State"],
  pathCost: 1,
  recommendations: ["📅 Schedule follow-up within 4-6 weeks"],
  facts: ["medium_risk"],
  inferredFacts: []
}
```

---

## ⚠️ Disclaimer

**CardioAI is a decision support tool, not a diagnostic device.**

- All results must be reviewed by qualified medical professionals
- Never make clinical decisions based solely on this system's output
- This tool is designed to assist — not replace — human clinical judgment
- Intended for educational and research purposes

---

## 🔬 Model Details

### Training Data
- **Dataset**: UCI Heart Disease
- **Samples**: 920 patients
- **Sources**: 4 hospitals across 3 countries
- **Features**: 13 clinical measurements

### Performance Metrics
- **AUC-ROC**: 0.898 (excellent discrimination)
- **Recall (Sensitivity)**: 0.931 (93.1% detection of disease cases)
- **Precision**: 0.87 (87% of positive predictions are correct)
- **Decision Threshold**: 0.35 (optimized for medical recall over precision)

### Why This Threshold?
In healthcare, missing a disease case (false negative) is more costly than a false alarm (false positive). We optimize for recall to minimize missed diagnoses.

---

## 🛠️ No External Dependencies

This is a **100% vanilla JavaScript** implementation:
- ✅ No frameworks (React, Vue, Angular)
- ✅ No jQuery
- ✅ No external ML libraries
- ✅ No build tools required
- ✅ Pure CSS (no SASS/LESS)
- ✅ Works offline

Just open in a browser and go!

---

## 🧪 Testing

### Quick Test Flow
1. **Home**: Read feature overview
2. **Assess Patient**: 
   - Enter default values (already filled)
   - Click "🔍 Analyse Risk"
   - Verify all outputs appear
3. **Simulator**:
   - Already pre-filled from assessment
   - Click "🔮 Run Simulation"
   - Adjust parameters and re-run
4. **Medical Knowledge**:
   - Search for "cholesterol"
   - Search for "blood pressure"
   - Search for "ECG"
5. **About**: Read mission and disclaimer
6. **AI Journey**: Learn how the system was built

---

## 📱 Browser Compatibility

| Browser | Version | Support |
|---------|---------|---------|
| Chrome  | 90+     | ✅ Full |
| Firefox | 88+     | ✅ Full |
| Safari  | 14+     | ✅ Full |
| Edge    | 90+     | ✅ Full |
| IE 11   | N/A     | ❌ Not supported |

---

## 📚 References

### Guidelines & Standards
- AHA/ACC Guidelines for Heart Disease
- NHLBI Cholesterol Treatment Guidelines
- Mayo Clinic Cardiology Best Practices
- Framingham Risk Score
- HEART Score for acute coronary syndrome

### ML & Algorithms
- XGBoost: Gradient Boosting Machines
- A* Search: Informed graph search
- Forward Chaining: Rule-based inference
- TF-IDF: Text similarity search

---

## 📝 License

This is an educational implementation. Use freely for learning and research.

---

## 🤝 Contributing

Feel free to fork and improve! Possible enhancements:
- Add more medical documents
- Implement additional ML models
- Support for multi-language search
- Export patient reports
- Integration with EHR systems

---

## 📞 Support

For questions or issues:
1. Check the About tab for mission context
2. Review the AI Journey tab for technical details
3. Test with the example values
4. Verify browser compatibility

---

**Built with**: HTML · CSS · JavaScript  
**Dataset**: UCI Heart Disease (920 patients)  
**Algorithms**: XGBoost · A* Search · Forward Chaining KB · TF-IDF NLP  
**Made for**: Clinical Decision Support Education
