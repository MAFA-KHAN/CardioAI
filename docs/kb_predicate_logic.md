# CardioAI — Knowledge Base in First-Order Predicate Logic (FOPL)

This document presents the formal specification of the CardioAI Knowledge Base (24 rules) using **First-Order Predicate Logic (FOPL)**. 

---

## 1. Formal Quantifiers and Predicates

To represent clinical reasoning formally, we define the domain of discourse as the set of all patients $x$.

### A. Primitive Predicates (Measurements)
* $\text{Patient}(x)$: $x$ is a patient.
* $\text{Age}(x)$: Age of patient $x$ in years.
* $\text{TrestBPS}(x)$: Resting blood pressure of patient $x$ in mmHg.
* $\text{Chol}(x)$: Serum cholesterol of patient $x$ in mg/dL.
* $\text{Thalach}(x)$: Maximum heart rate achieved by patient $x$ in bpm.
* $\text{Exang}(x)$: $x$ experiences exercise-induced angina ($1 = \text{Yes}$, $0 = \text{No}$).
* $\text{Oldpeak}(x)$: ST depression induced by exercise in mm.
* $\text{Thal}(x)$: Thalassemia type of patient $x$ ($\text{Normal}$, $\text{FixedDefect}$, $\text{ReversibleDefect}$).
* $\text{RestECG}(x)$: Resting electrocardiographic results ($\text{Normal}$, $\text{ST}$, $\text{LVH}$).
* $\text{CP}(x)$: Chest pain type ($\text{TypicalAngina}$, $\text{AtypicalAngina}$, $\text{NonAnginal}$, $\text{Asymptomatic}$).
* $\text{ML\_Probability}(x)$: Risk probability predicted by XGBoost for $x$.

### B. Derived Logical Predicates (Inferred Clinical Concepts)
* $\text{ElderlyPatient}(x)$: $x$ is categorized as an elderly patient.
* $\text{HighBP}(x)$: $x$ has high blood pressure.
* $\text{HighCholesterol}(x)$: $x$ has high cholesterol.
* $\text{LowMaxHR}(x)$: $x$ has an abnormally low maximum heart rate.
* $\text{ExerciseAngina}(x)$: $x$ has exercise-induced chest pain.
* $\text{HighOldpeak}(x)$: $x$ has significant ST depression.
* $\text{ReversibleThal}(x)$: $x$ has a reversible perfusion defect.
* $\text{ECGAbnormal}(x)$: $x$ has an abnormal resting ECG.
* $\text{SilentIschemiaRisk}(x)$: $x$ exhibits silent ischemia risk.
* $\text{AgeRelatedCardiacRisk}(x)$: $x$ has age-related cardiac risk factors.
* $\text{Hypertension}(x)$: $x$ has clinical Stage 2 hypertension.
* $\text{Hyperlipidemia}(x)$: $x$ has clinical hyperlipidemia.
* $\text{ReducedCardiacReserve}(x)$: $x$ has reduced cardiovascular reserve.
* $\text{PossibleIschemia}(x)$: $x$ has suspected myocardial ischemia.
* $\text{AbnormalStressResponse}(x)$: $x$ has an abnormal stress test response.
* $\text{ReversiblePerfusionDefect}(x)$: $x$ has a reversible perfusion defect.
* $\text{ECGDetectedAbnormality}(x)$: $x$ has a verified ECG abnormality.
* $\text{AsymptomaticCP}(x)$: $x$ has asymptomatic chest pain.
* $\text{ElevatedCardiovascularRisk}(x)$: $x$ has elevated cardiovascular risk.
* $\text{SuspectedCAD}(x)$: $x$ is suspected of Coronary Artery Disease.
* $\text{RequiresECG}(x)$: $x$ requires an electrocardiogram.
* $\text{CriticalScreeningNeeded}(x)$: $x$ requires critical cardiac screening.
* $\text{ExerciseCardiacFailureRisk}(x)$: $x$ is at risk of cardiovascular failure during exercise.
* $\text{DiagnosticTesting}(x)$: $x$ is referred for further diagnostic testing.
* $\text{CardiologyConsultation}(x)$: $x$ is referred for a cardiology consultation.
* $\text{CloseMonitoring}(x)$: $x$ is put under a close monitoring protocol.
* $\text{SpecialistFollowup}(x)$: $x$ requires specialist follow-up.
* $\text{TreatmentPlanning}(x)$: $x$ requires a formalized treatment plan.
* $\text{PreventiveEducation}(x)$: $x$ is referred for lifestyle education.
* $\text{HealthyLifestyle}(x)$: $x$ is guided on healthy lifestyle changes.
* $\text{RoutineMonitoring}(x)$: $x$ is placed on routine annual monitoring.
* $\text{FollowupAssessment}(x)$: $x$ requires follow-up re-evaluation.
* $\text{LifestyleCounseling}(x)$: $x$ receives custom lifestyle advice.
* $\text{HighRisk}(x)$: $x$ is classified as High Risk by the ML model.
* $\text{MediumRisk}(x)$: $x$ is classified as Medium Risk by the ML model.
* $\text{LowRisk}(x)$: $x$ is classified as Low Risk by the ML model.

---

## 2. FOPL Rules Specification

### Tier 1: Single-Feature Risk Flags (R01 – R09)

* **R01 (Age Risk)**:
  $$\forall x \, (\text{Patient}(x) \land \text{Age}(x) \ge 60 \rightarrow \text{AgeRelatedCardiacRisk}(x))$$
* **R02 (High Blood Pressure)**:
  $$\forall x \, (\text{Patient}(x) \land \text{TrestBPS}(x) \ge 140 \rightarrow \text{Hypertension}(x))$$
* **R03 (High Cholesterol)**:
  $$\forall x \, (\text{Patient}(x) \land \text{Chol}(x) \ge 240 \rightarrow \text{Hyperlipidemia}(x))$$
* **R04 (Low Maximum Heart Rate)**:
  $$\forall x \, (\text{Patient}(x) \land \text{Thalach}(x) < 120 \rightarrow \text{ReducedCardiacReserve}(x))$$
* **R05 (Exercise Angina)**:
  $$\forall x \, (\text{Patient}(x) \land \text{Exang}(x) == 1 \rightarrow \text{PossibleIschemia}(x))$$
* **R06 (High Oldpeak)**:
  $$\forall x \, (\text{Patient}(x) \land \text{Oldpeak}(x) > 2.0 \rightarrow \text{AbnormalStressResponse}(x))$$
* **R07 (Reversible Thalassemia)**:
  $$\forall x \, (\text{Patient}(x) \land \text{Thal}(x) == \text{ReversibleDefect} \rightarrow \text{ReversiblePerfusionDefect}(x))$$
* **R08 (Rest ECG Abnormality)**:
  $$\forall x \, (\text{Patient}(x) \land (\text{RestECG}(x) == \text{ST} \lor \text{RestECG}(x) == \text{LVH}) \rightarrow \text{ECGDetectedAbnormality}(x))$$
* **R09 (Asymptomatic Chest Pain)**:
  $$\forall x \, (\text{Patient}(x) \land \text{CP}(x) == \text{Asymptomatic} \rightarrow \text{AsymptomaticCP}(x))$$

---

### Tier 2: Combined Risk Rules (R10 – R14)

* **R10 (Metabolic Risk Syndrome)**:
  $$\forall x \, (\text{Patient}(x) \land \text{Hypertension}(x) \land \text{Hyperlipidemia}(x) \rightarrow \text{ElevatedCardiovascularRisk}(x))$$
* **R11 (Suspected Coronary Artery Disease)**:
  $$\forall x \, (\text{Patient}(x) \land \text{PossibleIschemia}(x) \land \text{HighRisk}(x) \rightarrow \text{SuspectedCAD}(x))$$
* **R12 (Mandatory ECG Referral)**:
  $$\forall x \, (\text{Patient}(x) \land \text{AbnormalStressResponse}(x) \land \text{PossibleIschemia}(x) \rightarrow \text{RequiresECG}(x))$$
* **R13 (Critical Missed Screening Risk)**:
  $$\forall x \, (\text{Patient}(x) \land \text{AgeRelatedCardiacRisk}(x) \land \text{AsymptomaticCP}(x) \rightarrow \text{CriticalScreeningNeeded}(x))$$
* **R14 (Exercise Cardiac Failure Risk)**:
  $$\forall x \, (\text{Patient}(x) \land \text{ReducedCardiacReserve}(x) \land \text{ReversiblePerfusionDefect}(x) \rightarrow \text{ExerciseCardiacFailureRisk}(x))$$

---

### Tier 3: Clinical Pathway Rules (R15 – R24)

* **R15 (ECG Requirement Referral)**:
  $$\forall x \, (\text{Patient}(x) \land \text{RequiresECG}(x) \rightarrow \text{DiagnosticTesting}(x))$$
* **R16 (Diagnostic Escalation)**:
  $$\forall x \, (\text{Patient}(x) \land \text{DiagnosticTesting}(x) \rightarrow \text{CardiologyConsultation}(x))$$
* **R17 (High Risk Patient Protocol)**:
  $$\forall x \, (\text{Patient}(x) \land \text{HighRisk}(x) \rightarrow \text{CloseMonitoring}(x))$$
* **R18 (Specialist Follow-up Trigger)**:
  $$\forall x \, (\text{Patient}(x) \land \text{CloseMonitoring}(x) \land \text{DiagnosticTesting}(x) \rightarrow \text{SpecialistFollowup}(x))$$
* **R19 (Treatment Formulation)**:
  $$\forall x \, (\text{Patient}(x) \land \text{SpecialistFollowup}(x) \rightarrow \text{TreatmentPlanning}(x))$$
* **R20 (Low Risk Education)**:
  $$\forall x \, (\text{Patient}(x) \land \text{LowRisk}(x) \rightarrow \text{PreventiveEducation}(x))$$
* **R21 (Prevention Engagement)**:
  $$\forall x \, (\text{Patient}(x) \land \text{PreventiveEducation}(x) \rightarrow \text{HealthyLifestyle}(x))$$
* **R22 (Stable Routine Monitoring)**:
  $$\forall x \, (\text{Patient}(x) \land \text{HealthyLifestyle}(x) \rightarrow \text{RoutineMonitoring}(x))$$
* **R23 (Medium Risk Re-evaluation)**:
  $$\forall x \, (\text{Patient}(x) \land \text{MediumRisk}(x) \rightarrow \text{FollowupAssessment}(x))$$
* **R24 (Tailored Lifestyle Intervention)**:
  $$\forall x \, (\text{Patient}(x) \land \text{FollowupAssessment}(x) \rightarrow \text{LifestyleCounseling}(x))$$
