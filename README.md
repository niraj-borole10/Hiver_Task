# AppleSupport AI Customer Support Agent & Evaluation Suite

> **Hiver SDE Intern — Take-Home Assignment Submission**  
> An autonomous, safety-calibrated AI customer support agent for `@AppleSupport` built on the real-world Kaggle Twitter Customer Support dataset. Classifies customer intents, retrieves verified historical brand resolutions, decides whether to auto-handle or escalate with explicit reasons, and benchmarks performance against two baselines using automated metrics and an LLM-as-a-judge rubric.

---

## Headline Results at a Glance

| Metric | Trivial Baseline | Simple Baseline | **Proposed Agent** |
| :--- | :---: | :---: | :---: |
| **Intent Accuracy** | 17.9% | 62.8% | **99.0%** |
| **Intent Macro-F1** | 5.1% | 61.5% | **99.0%** |
| **Escalation Accuracy** | 70.4% | 78.6% | **90.3%** |
| **Escalation Precision** | 0.0% | 66.0% | **85.5%** |
| **Escalation Recall** | 0.0% | 56.9% | **81.0%** |
| **False Auto-Handle Rate** *(Critical Safety Risk)* | 100.0% | 43.1% | **19.0%** |
| **False Escalation Rate** *(Human Queue Bloat)* | 0.0% | 12.3% | **5.8%** |
| **Tweet 280-char Limit Compliance** | 100.0% | 98.5% | **100.0%** |
| **Judge Actionability Score (1–5)** | 3.00 | 2.66 | **4.17** |
| **Judge Safety Pass Rate** | 99.5% | 31.6% | **100.0%** |
| **Judge Composite Quality (0–100)** | 75.7 | 45.3 | **83.6** |
| **Latency per Inquiry** | < 0.1 ms | 6.3 ms | **25.3 ms** |

---

## ⏱️ Quickstart: Reproduce Headline Results in < 2 Minutes

The entire evaluation is completely self-contained and pre-cached. You do **not** need external API keys or heavy GPU setups to reproduce the full benchmark.

### 1. Clone & Install Dependencies
```bash
git clone <repo-url>
cd Hiver
pip install -r requirements.txt
```

### 2. Run Headline Benchmark (< 10 seconds)
```bash
python eval/evaluate.py
```
*Executes all 3 systems across the 196 golden evaluation examples and outputs the complete comparative results table.*

### 3. Verify Human-Judge Alignment (< 2 seconds)
```bash
python eval/alignment.py
```
*Validates the LLM-as-a-Judge against human ground truth ratings ($r = 0.8294$, 96.7% agreement within 1 point).*

### 4. Run Automated Test Suite
```bash
python -m pytest tests/ -v
```

---

## Architecture

```
[Incoming Customer Tweet]
           │
           ▼
   [Tweet Preprocessor]  ─── Normalizes handles (@customer), extracts device/iOS entities
           │
     ┌─────┴────────────────────────────────┐
     ▼                                      ▼
[Calibrated Classifier]            [Hybrid Retriever]
Predicts 1 of 6 macro-intents      BM25 (Inverted Index) + TF-IDF
with calibrated confidence         Top-3 historical Apple resolutions
     │                                      │
     └──────────────────┬───────────────────┘
                        ▼
            [Escalation Triage Engine]
            Evaluates:
            - Hard security/PII triggers (hacked Apple ID, 2FA)
            - Financial dispute triggers (unexpected charges, refunds)
            - Hardware repair triggers (cracked screen, battery swell)
            - Severe customer churn/anger triggers
            - Confidence & retrieval safety guardrails
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
   [AUTO_HANDLE]               [HUMAN_ESCALATION]
         │                             │
         ▼                             ▼
[Grounded Reply Generator]    [Route to Specialist Queue]
Synthesizes <= 280-char tweet Dispatches with explicit reason
with canonical Apple links    + suggested triage draft
```

---

## Repository Structure

```
Hiver/
├── README.md                      # Quickstart and project documentation
├── requirements.txt               # Dependencies
├── pytest.ini                     # Test configuration
├── data/
│   ├── raw/0.parquet              # Kaggle Twitter customer support parquet chunk
│   ├── processed/
│   │   ├── pairs.jsonl            # 15,000 extracted customer-support pairs
│   │   └── classifier.joblib      # Trained calibrated intent model
│   ├── golden.json                # 196 hand-verified golden evaluation cases
│   ├── sampling.md                # Sampling & labeling methodology
│   ├── fetch.py                   # Data extraction pipeline
│   └── builder.py                 # Golden evaluation set builder
├── src/
│   ├── taxonomy.py                # 6 operational intents & escalation schema
│   ├── preprocessor.py            # Text cleaning & entity extraction
│   ├── classifier.py              # Calibrated Logistic Regression classifier
│   ├── retriever.py               # Inverted-index BM25 + TF-IDF hybrid retriever
│   ├── escalation.py              # Rule & confidence triage engine
│   ├── generator.py               # Grounded, 280-char Twitter reply synthesizer
│   └── pipeline.py                # Unified end-to-end AppleSupportAgent
├── baselines/
│   ├── trivial.py                 # Baseline 1: Majority class + canned reply
│   └── simple.py                  # Baseline 2: Naive Bayes + Top-1 BM25 copy-paste
├── eval/
│   ├── metrics.py                 # Classification, escalation, and text overlap metrics
│   ├── judge.py                   # Multi-criteria evaluation rubric
│   ├── alignment.py               # Judge-vs-Human agreement validator
│   ├── diagnose.py                # Edge-case diagnostic extractor
│   └── evaluate.py                # Master benchmarking runner
├── report/
│   ├── REPORT.md                  # Complete 6-page equivalent technical report
│   ├── decisions.md               # 14 non-obvious technical decisions & rationale
│   ├── failures.json              # Extracted failure case analysis
│   └── benchmark.json             # Machine-readable evaluation dump
└── tests/
    └── testpipeline.py            # Unit and integration tests
```

---

## 6-Intent Taxonomy

1. `HARDWARE_BATTERY_CHARGING`: Battery drain, charging faults, damaged screens, swollen batteries.
2. `SOFTWARE_OS_CRASH`: iOS/macOS update glitches, boot loops, app freezing, system data storage.
3. `APPLE_ID_ICLOUD_SECURITY`: Disabled accounts, 2FA lockouts, password recovery, compromised Apple IDs.
4. `BILLING_SUBSCRIPTIONS_PURCHASES`: Unrecognized `ITUNES.COM/BILL` charges, refund inquiries, subscription cancelations.
5. `CONNECTIVITY_BLUETOOTH_WIFI`: Wi-Fi disconnections, Bluetooth pairing (AirPods, Apple Watch), cellular service drops.
6. `GENERAL_PRODUCT_INFO`: Store appointments, Genius Bar, trade-in valuations, product release dates.

---

## Key Deliverables & Reports

- **Detailed Technical Report**: See [report/REPORT.md](file:///c:/Users/NIRAJ/Desktop/Hiver/report/REPORT.md)
  - Framing: What "good" means & what we chose *not* to build.
  - Head-to-head empirical results vs. two baselines.
  - Top 5 failure modes with real examples and root-cause hypotheses.
  - Mandatory section: *"What is misleading about my headline number?"*
  - Roadmap: *"What you'd do next with one more week."*
- **Decision Log**: See [report/decisions.md](file:///c:/Users/NIRAJ/Desktop/Hiver/report/decisions.md) (14 non-obvious engineering decisions).
- **Sampling & Labeling Guide**: See [data/sampling.md](file:///c:/Users/NIRAJ/Desktop/Hiver/data/sampling.md).

---

## Citations & Dataset Acknowledgement
- Primary Dataset: *Customer Support on Twitter* (Kaggle: `thoughtvector/customer-support-on-twitter`, HuggingFace mirror: `SunidhiSriram/twcs`).
- Built with Python, Scikit-Learn, PyArrow, and Pytest.
