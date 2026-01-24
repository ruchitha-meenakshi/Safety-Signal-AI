# Data Dictionary: UCI Drug Review Dataset

## Overview
This dataset contains patient reviews on specific drugs, along with related conditions and a 10-point satisfaction rating. It is used to train a Natural Language Processing (NLP) model to detect adverse drug events (safety signals) from unstructured text.

## Data Schema & Roles

| Column Name | Data Type | Description | Role in AI Pipeline |
| :--- | :--- | :--- | :--- |
| **uniqueID** | Integer | Unique identifier for each review. | **Ignored** (Hashed in Silver Layer for HIPAA Privacy compliance). |
| **drugName** | String | Name of the medication. | **Feature** (Used for grouping and finding drug-specific side effects). |
| **condition** | String | The patient's condition (e.g., "Depression"). | **Context** (Helps validate if a symptom is a side effect or part of the disease). |
| **review** | String | The raw text narrative. | **Input Feature** (The primary source for NLP tokenization and TF-IDF vectorization). |
| **rating** | Integer | 1-10 satisfaction score. | **Target Variable** (Converted to binary `is_adverse_event` label). |
| **date** | String | Date of the review entry. | **Time Series** (Parsed to `event_date` for trend analysis). |
| **usefulCount** | Integer | Number of users who found the review helpful. | **Weight** (Potential signal for review credibility). |

## Target Definition (Problem Framing)
We have framed this as a **Binary Classification** problem to predict safety signals.

* **Input:** `review` (Text)
* **Output:** `is_adverse_event` (0 or 1)
* **Business Logic:**
    * **Class 1 (Adverse Event / Negative):** `Rating 1-4`. These reviews likely contain descriptions of side effects or ineffectiveness.
    * **Class 0 (Safe / Positive):** `Rating 7-10`. These reviews indicate the drug worked as intended.
    * **Excluded (Noise Reduction):** `Rating 5-6`. These are neutral/ambiguous reviews that confuse the model, so they are filtered out during training.

---

## Source & Acknowledgements

**Primary Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com)  
**Mirror:** [Kaggle - UCI Drug Review Dataset](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018)

**Citation:**
> Gräßer, F., Kallumadi, S., Malberg, H., & Zaunseder, S. (2018). *Aspect-Based Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning*. In Proceedings of the 2018 International Conference on Digital Health (DH '18). ACM, New York, NY, USA, 121-125.

**Original Authors:**
Surya Kallumadi & Felix Gräßer

**Terms of Use:**
* Used for research/educational purposes only.
* Non-commercial use.
* No redistribution of raw data.
