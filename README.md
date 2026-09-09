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

## Quickstart: Reproduce Headline Results in < 2 Minutes

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

## 6-Intent Operational Taxonomy

Each incoming inquiry is mapped to one of six production-defined operational intents designed around distinct resolution workflows:

| Intent Category | Scope & Diagnostic Triggers | Example Customer Query | Resolution Pathway |
| :--- | :--- | :--- | :--- |
| **`HARDWARE_BATTERY_CHARGING`**<br>`Hardware` | Battery health decay, rapid drain, charging port faults, swollen battery, physical screen crack. | *"My iPhone 7 battery is draining from 100% to 20% in 2 hours since updating."* | **Self-Serve**: Battery Health settings guide (`apple.co/BatteryInfo`).<br>**Escalate**: Physical damage or replacement intake. |
| **`SOFTWARE_OS_CRASH`**<br>`Software & OS` | iOS/macOS update glitches, continuous boot loops, app freezes, "System Data" storage bugs. | *"Phone is stuck on the Apple logo after updating to iOS 11 and won't turn on."* | **Self-Serve**: Force restart & DFU mode steps (`apple.co/ForceRestart`).<br>**Escalate**: Persistent bricking/hardware restore errors. |
| **`APPLE_ID_ICLOUD_SECURITY`**<br>`Security & Auth` | Disabled Apple IDs, 2FA code delivery failures, password recovery, unauthorized logins. | *"Someone hacked into my Apple ID and changed my recovery email! Help!"* | **Self-Serve**: Official portal (`iforgot.apple.com`).<br>**Escalate**: Account takeover, stolen devices, 2FA lockout. |
| **`BILLING_SUBSCRIPTIONS_PURCHASES`**<br>`Billing & Subscriptions` | Unrecognized `ITUNES.COM/BILL` charges, refund requests, unwanted subscription renewals. | *"Why was I charged $9.99 from ITUNES.COM/BILL? I bought nothing, refund me now."* | **Self-Serve**: Subscriptions management (`reportaproblem.apple.com`).<br>**Escalate**: Fraud reports, double charges, payment disputes. |
| **`CONNECTIVITY_BLUETOOTH_WIFI`**<br>`Connectivity` | Wi-Fi disconnects, AirPods/Apple Watch pairing failure, cellular "No Service" errors. | *"My AirPods keep disconnecting every 5 minutes from my MacBook Pro during calls."* | **Self-Serve**: Network settings reset & re-pairing guide.<br>**Escalate**: Carrier lock, SIM card hardware failure. |
| **`GENERAL_PRODUCT_INFO`**<br>`Store & Product` | Store hours, Genius Bar appointments, trade-in estimates, official warranty coverage. | *"Can I trade in my iPhone 6s for the new iPhone 8 at the Genius Bar?"* | **Self-Serve**: Store locator & AppleCare coverage check.<br>**Escalate**: Lost/stolen pre-order shipments requiring order lookup. |

---

## Key Deliverables & Documentation

> [!NOTE]
> All deliverables required by the Hiver specification are fully documented, linked below, and reproducible.

<table>
  <tr>
    <td width="50%">
      <h3><a href="report/REPORT.md">Technical Report</a></h3>
      <p><b>Comprehensive 6-page equivalent engineering deep-dive:</b></p>
      <ul>
        <li><b>Problem Framing</b>: What "good" means for @AppleSupport & intentional non-goals.</li>
        <li><b>Head-to-Head Benchmarks</b>: Trivial Baseline vs. Simple Baseline vs. Proposed Agent.</li>
        <li><b>Top 5 Failure Modes</b>: Real verbatim examples, root-cause hypotheses, and mitigations.</li>
        <li><b>"What is Misleading About My Headline Number?"</b>: Critique of ROUGE copy-paste bias & accuracy paradox.</li>
        <li><b>Roadmap</b>: Concrete engineering goals for "one more week".</li>
      </ul>
    </td>
    <td width="50%">
      <h3><a href="report/decisions.md">Decision Log</a></h3>
      <p><b>14 non-obvious engineering & product decisions:</b></p>
      <ul>
        <li>Why @AppleSupport was chosen over @AmazonHelp.</li>
        <li>Why a 6-intent taxonomy outperforms a 77-class fine-grained taxonomy.</li>
        <li>Why the golden set was enriched to 30% escalations.</li>
        <li>Inverted-index BM25 design for sub-millisecond retrieval.</li>
        <li>Asymmetric cost matrix: False Auto-Handles vs. False Escalations.</li>
        <li>Whitelisting official <code>apple.co/*</code> domains to prevent hallucination.</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3><a href="data/golden.json">Golden Benchmark Set</a></h3>
      <p><b>196 hand-verified, stratified test inquiries:</b></p>
      <ul>
        <li>Balanced across all 6 operational intents.</li>
        <li>Calibrated <b>70.4% Auto-Handle / 29.6% Human Escalation</b> split.</li>
        <li>Includes real edge cases: severe customer anger, compound inquiries, financial disputes.</li>
        <li>Full annotations: <code>ground_truth_intent</code>, <code>ground_truth_escalation</code>, <code>escalation_reason</code>, <code>reference_reply</code>.</li>
      </ul>
    </td>
    <td width="50%">
      <h3><a href="data/sampling.md">Sampling & Labeling Guide</a></h3>
      <p><b>Protocol & taxonomy documentation:</b></p>
      <ul>
        <li>Stratified keyword cluster sampling methodology.</li>
        <li>Decision boundary definitions between self-serve and human escalation.</li>
        <li>Quality assurance and anonymization protocols.</li>
        <li>Inter-annotator edge-case guidelines.</li>
      </ul>
    </td>
  </tr>
</table>

---

## Tech Stack & Dataset Citations

<div align="center">

| Component | Technology / Source | Description |
| :--- | :--- | :--- |
| **Primary Dataset** | [Kaggle Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) | ~3M real-world multi-turn support tweets (`@AppleSupport` subset: ~102k tweets). |
| **ML Engine** | `Scikit-Learn` + `NumPy` | TF-IDF n-gram vectorization with calibrated multi-class Logistic Regression. |
| **Retrieval** | Custom Inverted-Index BM25 + Vector Cosine | High-speed hybrid lexical & semantic retrieval over 5,000 historical resolutions. |
| **Safety Guardrails** | Deterministic Regex + Confidence Calibration | Multi-tier triage engine protecting security, payment, and hardware policies. |
| **Evaluation** | `Pytest` + Automated Multi-Axis Rubric | Pearson-aligned LLM-as-a-judge ($r = 0.8294$) and automated statistical metrics. |

</div>

