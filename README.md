# AI Customer Support Ticket Triage

An NLP-based customer support ticket triage system that automatically predicts the **ticket type** and **priority**, while identifying low-confidence predictions that should be reviewed by a human.

The system classifies tickets into four types:

* `Incident`
* `Problem`
* `Request`
* `Change`

It also predicts ticket priority as:

* `High`
* `Medium`
* `Low`

---

## 👩‍💻 About the Project

**Developed by:** Drishti Goel
**College:** Akhilesh Das Gupta College (ADGIPS)
**University:** GGSIPU
**Batch:** ADGIPS'28
**Program:** BTech – Machine Learning & Deep Learning

This project was developed as an NLP/ML project focused on automating the initial classification and routing of customer support tickets.

---

## 📌 Project Overview

Customer support teams often receive a large number of tickets that need to be categorized and prioritized before they can be handled.

This project uses text embeddings and machine learning models to automate that process.

For every incoming ticket, the system:

1. Takes the ticket subject and description.
2. Converts the text into an embedding.
3. Predicts the ticket type.
4. Predicts its priority.
5. Calculates prediction confidence.
6. Automatically handles sufficiently confident Type predictions.
7. Sends uncertain Type predictions for human review.

The review mechanism is based on the confidence of the **Type prediction**, rather than Priority.

---

## 📊 Model Performance

The final evaluation uses a group-aware split designed to prevent similar or near-duplicate tickets from appearing in both training and testing data.

### Type Classification

The system achieved approximately **0.76–0.77 macro-F1** for the four Type classes, compared with a majority-class baseline of approximately **0.14**.

At a Type confidence threshold of `0.7`:

* Around **47%** of tickets can be automatically handled.
* These automatically handled tickets achieve around **93% accuracy**.
* Lower-confidence tickets are sent for human review.

### Priority Classification

Priority is considerably harder to predict from text alone.

The final Priority macro-F1 is approximately **0.41**, compared with a baseline of about **0.20**.

Because Priority predictions are less reliable, Priority confidence is reported separately and does not decide whether a ticket requires review.

---

## ⚙️ How the System Works

### 1. Data Preparation

The project uses English records from multiple customer-support-ticket CSV files.

The preprocessing pipeline:

* Removes missing ticket text.
* Removes repeated ticket bodies.
* Combines subject and body into a single text field.
* Detects near-duplicate tickets.
* Keeps similar tickets within the same split during evaluation.

After cleaning:

* **25,137 tickets** remained.
* **4,596 duplicate bodies** were removed.
* Training set: **20,110 tickets**
* Test set: **5,027 tickets**

The Type distribution is:

| Type     | Tickets |
| -------- | ------: |
| Incident |  10,023 |
| Request  |   7,253 |
| Problem  |   5,199 |
| Change   |   2,662 |

---

### 2. Text Embeddings

Each support ticket is converted into a numerical vector using:

`BAAI/bge-small-en-v1.5`

These embeddings are then used as input for the machine learning classifiers.

---

### 3. Models Tested

Several models were compared during experimentation:

* Logistic Regression
* LinearSVC
* Random Forest
* Majority-class baseline

Class balancing was used where applicable to reduce the effect of class imbalance.

---

## 🔬 Evaluation Strategy

A major part of this project was making sure that the evaluation was not artificially inflated by similar tickets appearing in both training and testing sets.

Near-duplicate tickets were grouped using cosine similarity greater than `0.95`.

These groups were then kept together using a **StratifiedGroupKFold** based split.

This prevents highly similar tickets from leaking information between the training and test sets.

---

## 🚨 Data Leakage Investigation

During the initial random split, Random Forest produced unusually high Priority performance.

Further similarity analysis showed that many test tickets were extremely similar to training tickets.

For example:

* Median nearest-training similarity was approximately **0.966**.
* Around **62.5%** of test tickets had a training ticket with cosine similarity above `0.95`.
* A simple 1-NN label-copying approach achieved approximately **0.894 Type accuracy** and **0.81 Priority accuracy** without actually learning a classification model.

To address this, near-duplicate tickets were grouped before splitting the data.

After applying the fix:

* **13,564 similarity groups** were created.
* The largest group contained **499 tickets**.
* Maximum test-to-train similarity was reduced to approximately `0.95`.
* 1-NN Priority accuracy dropped to approximately **0.461**.

This group-aware evaluation is used for the final reported results.

---

## 📈 Final Evaluation

### Group-Aware Split

| Model               | Type Accuracy | Type Macro-F1 | Priority Accuracy | Priority Macro-F1 |
| ------------------- | ------------: | ------------: | ----------------: | ----------------: |
| Baseline            |         0.399 |         0.143 |             0.424 |             0.198 |
| Logistic Regression |         0.745 |         0.764 |             0.416 |             0.408 |
| LinearSVC           |         0.774 |         0.773 |             0.434 |             0.408 |
| Random Forest       |         0.765 |         0.739 |             0.469 |             0.393 |

Logistic Regression was selected for deployment because it provides probability estimates through `predict_proba` while producing performance close to the other strong models.

---

## 🎯 Confidence-Based Human Review

The system uses the confidence of the Type prediction to decide whether automatic handling is appropriate.

| Confidence Threshold | Automatically Handled | Accuracy |
| -------------------: | --------------------: | -------: |
|                  0.0 |                  100% |    0.745 |
|                  0.5 |                   95% |    0.763 |
|                  0.6 |                   67% |    0.841 |
|                  0.7 |                   47% |    0.930 |
|                  0.8 |                   35% |    0.987 |

### Current Threshold

The deployed Type confidence threshold is:

`0.7`

If Type confidence is below `0.7`, the ticket is marked:

```text
needs_review = true
```

Priority confidence is also returned, with `priority_reliable` becoming true at confidence `>= 0.6`.

However, Priority confidence does **not** trigger human review.

---

## 🧹 Preprocessing Experiment

Different text preprocessing strategies were also compared using Logistic Regression.

| Preprocessing                   | Type Macro-F1 | Priority Macro-F1 |
| ------------------------------- | ------------: | ----------------: |
| Raw text                        |         0.764 |             0.408 |
| Lowercase + punctuation removal |         0.765 |             0.404 |
| Stop-word removal               |         0.746 |             0.411 |

The experiments did not show a meaningful improvement from aggressive preprocessing.

Therefore, the final pipeline uses relatively raw ticket text because the embedding model can already handle casing and punctuation effectively.

---

## 🖥️ Application

The project provides both:

### Streamlit Interface

A simple interface where users can enter or submit a customer support ticket and receive:

* Predicted Type
* Type confidence
* Predicted Priority
* Priority confidence
* Priority reliability
* Human-review requirement

### FastAPI Backend

The project also exposes a `/triage` API endpoint that can be used by other applications.

---

## 📂 Project Structure

```text
AI-Customer-Support-Ticket-Triage/
│
├── app.py
├── clf_priority.joblib
├── clf_type.joblib
├── data_processing.ipynb
├── LinearSVC_RandomForest.ipynb
├── requirements.txt
├── streamlit_app.py
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/drishti-g/AI-Customer-Support-Ticket-Triage.git
cd AI-Customer-Support-Ticket-Triage
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

#### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

```bash
python -m streamlit run streamlit_app.py
```

The Streamlit interface will then be available in your browser.

---

## 🔌 Run the FastAPI Server

```bash
python -m uvicorn app:app --reload
```

API documentation can be accessed at:

```text
http://127.0.0.1:8000/docs
```

---

## 📡 API Usage

The API accepts a support ticket through the `/triage` endpoint.

Example request:

```json
{
  "text": "Subject: Feature change. Body: Could you please add an export to CSV option in the reports dashboard?"
}
```

Example response:

```json
{
  "type": "Change",
  "type_confidence": 0.922,
  "priority": "low",
  "priority_confidence": 0.593,
  "priority_reliable": false,
  "needs_review": false
}
```

---

## 📚 Dataset

The project uses the **Multilingual Customer Support Tickets** dataset available through Kaggle.

The original dataset contains multiple languages, but this project uses the **English records** for the experiments.

The raw CSV files are not included in the repository.

Required dataset files:

```text
dataset-tickets-multi-lang-4-20k.csv
dataset-tickets-multi-lang3-4k.csv
aa_dataset-tickets-multi-lang-5-2-50-version.csv
```

Place the files inside:

```text
datasets/
```

---

## 📓 Reproducing the Experiments

Run:

```text
data_processing.ipynb
```

first.

This notebook prepares the dataset and generates the embedding files.

After that, run:

```text
LinearSVC_RandomForest.ipynb
```

This notebook contains the model comparison, similarity analysis, grouped splitting, evaluation and final classifier generation.

The project uses `scikit-learn==1.9.1`, so keeping the same version is recommended when loading the saved `.joblib` models.

---

## ⚠️ Limitations

A few limitations of the current system are:

* `Problem` and `Incident` are relatively difficult to distinguish.
* Priority cannot be predicted very reliably from text alone.
* The evaluation is based on one train/test split.
* The dataset contains a significant number of near-duplicate tickets.
* The current pipeline supports English text only.
* Ticket Type labels can sometimes be subjective.
* Small differences in evaluation scores may not represent meaningful performance improvements.

---

## 🧠 Key Learnings

This project helped explore several practical aspects of machine learning and NLP:

* Text embedding generation
* Multi-class classification
* Handling imbalanced classes
* Model comparison
* Confidence-based prediction
* Human-in-the-loop ML systems
* Data leakage detection
* Near-duplicate detection
* Group-aware evaluation
* FastAPI model serving
* Streamlit application development

One important lesson from the project was that **a high evaluation score does not always mean a model is genuinely performing well**. Checking for duplicate or near-duplicate records is important when working with real-world text datasets.

---

## 👩‍💻 Author

**Drishti Goel**

BTech – Machine Learning & Deep Learning
GGSIPU | ADGIPS'28

Interested in **AI, GENAI , Agentic AI, Machine Learning, Data Science, UI/UX and Graphics Designing**.

---

## ⭐ Acknowledgement

Dataset used in this project:

**Multilingual Customer Support Tickets Dataset — Kaggle**

The project was developed for learning and experimentation with NLP-based customer support automation.
