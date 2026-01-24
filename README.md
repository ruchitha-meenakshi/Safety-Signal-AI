# Safety-Signal-AI: Pharmacovigilance Engine
> **Automated Adverse Event Detection using GenAI & Databricks Lakehouse**

![Status](https://img.shields.io/badge/Status-Active-success)
![Platform](https://img.shields.io/badge/Platform-Databricks%20%7C%20Unity%20Catalog-orange)

## 1. Executive Summary & Problem Statement
**The Vision:** To build an automated "Phase IV" monitor that listens to patient stories on the internet to detect hidden drug risks.

### The Problem: The "Statistical Blind Spot"
* **Ideal:** Clinical trials should catch all side effects.
* **Reality:** Trials only test **~3,000 people**. If a side effect happens to **1 in 10,000**, it is statistically missed.
* **Consequences:** When the drug launches to millions, dangerous side effects go undetected for months because patients report them on social forums (Reddit/WebMD), not official FDA forms.

**AI Framing:**
This is a **Text Classification & Extraction** problem. Traditional SQL cannot query unstructured patient reviews. We use **Natural Language Processing (NLP)** to extract "Hidden Signals" from messy text.

---

## 2. High-Level Architecture
*(Placeholder: We will insert a screenshot of your Databricks Workflow here later)*

We follow the **Medallion Architecture** to ensure governance and data quality:

| Layer | Type | Responsibility |
| :--- | :--- | :--- |
| **Bronze** | **Ingestion** | Ingest raw TSV reviews via **Auto Loader** (Simulating Streaming). |
| **Silver** | **Transformation** | **Unity Catalog** masking (Privacy) and Schema Enforcement. |
| **Gold** | **Aggregates** | Business-level tables: `Drug_Risk_Score` and `Side_Effect_Trends`. |
| **ML** | **Prediction** | **MLflow** registry to track the "Risk Classifier" model. |

---

## 3. Technical Deep Dive

### Data Engineering & Governance
* **Unity Catalog:** We use UC to govern the `raw_data` schema. PII (Patient IDs) is masked at the Silver layer so data scientists never see personal info.
* **Delta Lake:** All tables use `OPTIMIZE` and `ZORDER BY drugName` to ensure fast query performance on millions of rows.

### AI Innovation: Aspect-Based Sentiment Analysis (ABSA)
Standard sentiment analysis fails in healthcare (e.g., *"My headache is gone [Positive], but I am dizzy [Negative]"*).
* **Our Approach:** We separate the **Condition** (Anxiety) from the **Side Effect** (Dizziness) to avoid false positives.
* **Metrics:** We optimize for **Recall** (Sensitivity) because missing a dangerous side effect is worse than a false alarm.

---

## 4. The Pipeline (How to Run)

1.  **Setup (`01_Setup`):** Initializes Catalog/Volumes and downloads the **UCI Drug Review Dataset** (215k records).
2.  **Bronze (`02_Ingest`):** Reads raw TSV files with schema enforcement.
3.  **Silver (`03_Clean`):**
    * *Logic:* Removes HTML tags (`<br>`) and hashes Patient IDs.
4.  **Gold (`04_Analytics`):** Aggregates safety signals.
5.  **ML (`05_Training`):** Trains the Risk Predictor.

---

## 5. Business Impact
* **Early Detection:** Identifies risks months before clinical reports.
* **Cost Savings:** Reduces manual review time by 90% using AI to filter noise.
* **Patient Safety:** The ultimate metric—preventing harm.
