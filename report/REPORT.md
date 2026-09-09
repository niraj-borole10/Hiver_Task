# Technical Report: AI Customer Support Agent for @AppleSupport

**Author**: SDE Intern Candidate  
**Role**: AI Engineering & Systems Evaluation Take-Home  
**Dataset**: Kaggle Customer Support on Twitter (`@AppleSupport` subset: ~102,000 interactions)  
**Golden Evaluation Set**: 196 hand-verified, stratified real-world customer inquiries  

---

## 1. Problem Framing

### 1.1 What "Good" Means for @AppleSupport
Customer support for `@AppleSupport` on Twitter operates under strict operational and brand constraints distinct from general enterprise support:

1. **Safety & Zero Security Compromise**:
   Apple never solicits credentials, passwords, two-factor authentication codes, or IMEI numbers in a public forum. A "good" support agent must instantly recognize when an inquiry crosses into private account management (e.g. Apple ID lockouts, activation lock, device theft, billing disputes) and enforce immediate, secure handoff to private direct channels or authenticated portals (`iforgot.apple.com`, `reportaproblem.apple.com`).
2. **Strict Grounding & Zero Hallucination**:
   Inventing fictitious iOS settings paths (e.g., *"Go to Settings > Battery > Super Clean"*) or dead URLs erodes brand trust and creates user frustration. Replies must cite legitimate, canonical Apple support resources (`apple.co/*`).
3. **Concise, Empathetic, and Action-Oriented**:
   Customer tweets are character-constrained and often written in moments of acute device frustration. Good responses maintain Apple's calm, helpful, non-robotic tone, diagnose the symptom rapidly, and provide a single actionable troubleshooting step or verified documentation link within Twitter's 280-character limit.
4. **Calibrated Escalation Triage**:
   Auto-handling an inquiry that requires human authentication is a critical safety failure. Conversely, escalating trivial self-serve issues (e.g. how to force restart an iPhone) floods human support queues and explodes labor costs. "Good" triage optimizes for high escalation recall on sensitive issues while minimizing false escalations.

### 1.2 What We Chose NOT to Build
To ensure production viability and reliability, we made intentional negative engineering choices:

* **No Automated In-Tweet Transaction Execution**: We explicitly chose *not* to build autonomous refund issuance or Apple ID password resets directly within the bot. Executing financial or security transactions in a public/semi-public social media layer violates security isolation.
* **No Unconstrained Open-Ended LLM Free-Form Chat**: We rejected allowing the LLM to generate replies without structured grounding or link whitelisting. Unconstrained generation inevitably hallucinates URLs and diagnostic advice.
* **No 77-Class Fine-Grained Intent Bloat**: We intentionally avoided granular micro-intents (e.g., separate classes for "iPhone 7 battery drain" vs. "iPhone 8 battery drain"). Micro-intents degrade classification robustness without providing actionable routing differences. We designed a compact 6-intent taxonomy mapped directly to distinct resolution workflows.

---

## 2. Empirical Benchmark Results vs. Baselines

We evaluated three complete end-to-end systems against the **196 hand-verified golden evaluation set**:

1. **Trivial Baseline**: Majority-class intent classifier (`SOFTWARE_OS_CRASH`) + static generic canned reply (*"Thanks for reaching out! Please send us a DM..."*) + zero-escalation policy (always auto-handles).
2. **Simple Baseline**: Uncalibrated TF-IDF + Multinomial Naive Bayes classifier + Top-1 BM25 verbatim historical tweet copy-paste + naive keyword-lookup escalation.
3. **Proposed Agent**: Calibrated Intent Classifier with n-grams & keyword guardrails + Hybrid BM25/TF-IDF Resolution Retriever + Multi-tier Escalation Triage Engine + Grounded Response Synthesizer.

### Head-to-Head Performance Comparison Table

| Metric Category | Metric | Trivial Baseline | Simple Baseline | Proposed Agent | Impact / Delta |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Intent Classification** | Accuracy | 17.9% | 62.8% | **99.0%** | **+36.2%** over Simple |
| | Macro-F1 | 5.1% | 61.5% | **99.0%** | **+37.5%** over Simple |
| **Escalation Triage** | Overall Accuracy | 70.4% | 78.6% | **90.3%** | **+11.7%** over Simple |
| | Escalation Precision | 0.0% | 66.0% | **85.5%** | **+19.5%** over Simple |
| | Escalation Recall | 0.0% | 56.9% | **81.0%** | **+24.1%** over Simple |
| | **False Auto-Handle Rate** *(Critical Safety Risk)* | 100.0% | 43.1% | **19.0%** | **-24.1%** reduction in safety leaks |
| | **False Escalation Rate** *(Labor Cost)* | 0.0% | 12.3% | **5.8%** | **-6.5%** reduction in queue bloat |
| **Generation & Compliance** | 280-Char Tweet Compliance | 100.0% | 98.5% | **100.0%** | Zero truncation failures |
| | ROUGE-1 Overlap (vs Reference) | 0.201 | 0.996* | 0.150 | *See Section 4 on ROUGE distortion |
| **LLM-as-a-Judge Rubric** | Groundedness (1–5) | 3.56 | 4.15 | **4.23** | Highest factual grounding |
| | Empathy & Brand Tone (1–5) | 5.00 | 3.67 | **4.13** | Balanced, professional tone |
| | Actionability (1–5) | 3.00 | 2.66 | **4.17** | **+1.51 points** over Simple |
| | Safety Pass Rate (%) | 99.5% | 31.6% | **100.0%** | Zero safety/policy violations |
| | **Composite Quality Score (0–100)** | 75.7 | 45.3 | **83.6** | **+38.3 points** over Simple |
| **System Latency** | Runtime on 196 queries | < 0.1s | 1.23s | **4.96s** (25.3 ms/req) | Real-time ready |

---

## 3. Failure Analysis: Top 5 Failure Modes

Analysis of the 20 failure cases on the golden benchmark reveals five primary edge cases:

### Mode 1: Compound / Multi-Issue Entanglement
* **Real Example (ID 15)**:  
  *Customer*: `"Hey @AppleSupport why after updating my iPhone to this new update that my phone restarts, has no service, and is slower than ever 🙄"`
* **Observed Outcome**: Model predicted `SOFTWARE_OS_CRASH` and triggered human escalation for general low confidence, but ground truth was `CONNECTIVITY_BLUETOOTH_WIFI` with carrier auth escalation.
* **Root Cause Hypothesis**: The customer conflated three distinct operational domains in one sentence (software update bug, random reboots, and cellular carrier loss). Single-label classification architectures inevitably collapse complex multi-symptom inquiries into a single dominant label.
* **Mitigation**: Implement multi-label classification or hierarchical decomposition where compound tweets are split into sub-clauses before triage.

### Mode 2: Inverted Transaction State (Waiting on Prior Resolution)
* **Real Example (ID 33)**:  
  *Customer*: `"@AppleSupport I got refunded for an in app purchase on Monday & got a conformation email, but there’s no sign of it in my account yet..."`
* **Observed Outcome**: Model categorized as `BILLING_SUBSCRIPTIONS_PURCHASES` but flagged as `AUTO_HANDLE` because standard refund keywords triggered the self-serve `reportaproblem.apple.com` link.
* **Root Cause Hypothesis**: The user was not requesting an initial refund (which is self-serve), but reporting an uncredited bank transfer following an approved refund. The model failed to detect that the temporal state of the transaction had transitioned from "refund request" to "inter-bank settlement delay."
* **Mitigation**: Add state-tracking intent predicates (`POST_REFUND_DELAY` vs. `INITIAL_REFUND_REQUEST`) and check for words indicating prior confirmation ("already refunded", "confirmation email").

### Mode 3: Rhetorical / Post-Resolution Queries
* **Real Example (ID 14)**:  
  *Customer*: `"@AppleSupport Where I live I barely have service, so it says “no service” but I usually send texts through WiFi & it still doesn’t work. But now it’s working. Why is that?"`
* **Observed Outcome**: System generated standard troubleshooting steps to reset network settings, ignoring that the user noted the issue had already resolved.
* **Root Cause Hypothesis**: The classifier detected symptom keywords ("no service", "WiFi doesn't work") and fired the standard diagnostic pathway without recognizing the past-tense resolution clause ("But now it's working").
* **Mitigation**: Sentiment/discourse parser to distinguish active outages from informational inquiries.

### Mode 4: False-Positive Escalation from Colloquial Frustration
* **Real Example (ID 4)**:  
  *Customer*: `"how come when I am listening to ... my music keeps pausing by itself????? ... I'm sick of your shit Apple"`
* **Observed Outcome**: Triggered `HUMAN_ESCALATION` under `LOW_CONFIDENCE` guardrail.
* **Root Cause Hypothesis**: The customer used colloquial profanity while describing a basic Spotify/Apple background audio playback glitch. Aggressive wording suppressed confidence scores below the automation threshold.
* **Trade-off**: While this represents an operational inefficiency (wasting human agent time on a minor audio bug), it is safer to over-escalate frustrated customers than to dismiss them with a robotic canned message.

### Mode 5: Physical Injury / Defective Hardware Safety Warnings
* **Real Example (ID 78)**:  
  *Customer*: `"Got burned and scarred with my magsafe charger and Apple didn't give a shit, so I'm switching to Microsoft. @AppleSupport"`
* **Observed Outcome**: Escalated correctly under `HARDWARE_REPAIR_BOOKING`, but predicted intent shifted toward billing/accessories.
* **Root Cause Hypothesis**: Extreme safety events (burns, smoke, melting chargers) cross traditional hardware repair categories and should trigger an urgent, specialized `PRODUCT_SAFETY_HAZARD` routing rather than standard Genius Bar booking.

---

## 4. "What is Misleading About My Headline Number?"

A core tenet of senior ML engineering is acknowledging the deceptive nature of aggregate metrics:

1. **The ROUGE-1 "Copy-Paste" Illusion**:
   * On paper, the Simple Baseline achieved a near-perfect ROUGE-1 score of **0.996**, while the Proposed Agent scored **0.150**.
   * **Why this is misleading**: The Simple Baseline simply copy-pasted verbatim historical tweets from the training set. Because Twitter customer support conversations frequently contain customer handles (`@115712`), greeting boilerplate, and sign-offs (`^CB`), verbatim copying duplicates superficial n-grams while completely failing to provide clean, customized guidance. In fact, the Simple Baseline had an abysmal **Actionability score of 2.66/5.0** and failed **68.4% of safety evaluations**. High ROUGE rewarded memorization, not helpfulness.

2. **The Escalation Accuracy Paradox**:
   * The Trivial Baseline achieved a deceptively respectable **70.4% Escalation Accuracy** simply by predicting `AUTO_HANDLE` for every single tweet.
   * **Why this is misleading**: In support triage, accuracy is the wrong metric. Under the Trivial Baseline, the **False Auto-Handle Rate was 100%**, meaning every compromised Apple ID, credit card dispute, and shattered screen was ignored. In production, a 70% accurate model that misses 100% of security threats is catastrophic. Our Proposed Agent achieves an **81.0% recall on escalations** and reduces the critical false-auto-handle rate to **19.0%**.

3. **Golden Set Distribution Bias**:
   * Our golden set purposefully sampled an enriched **~30% escalation rate** to make the evaluation rigorous and sensitive.
   * **Why this is misleading**: In the wild, inbound Twitter volume often has an escalation rate closer to 10–15%. If evaluated on raw unstratified Twitter volume, the absolute accuracy numbers would artificially rise, but the precision-recall trade-offs would shift.

---

## 5. What We'd Do Next with One More Week

1. **Multi-Turn Contextual Thread Modeling**:
   * Real Twitter support is multi-turn. Current evaluation treats each inbound tweet independently. With more time, we would pass the prior 2–3 turns of conversation history into the retriever and classifier to resolve pronoun references ("it still doesn't work") and track previously attempted troubleshooting steps.
2. **Dense Semantic Embeddings with Fine-Tuning**:
   * Replace the TF-IDF dense layer with domain-fine-tuned `bge-small` or `text-embedding-3-small` vectors conditioned on Apple-specific error codes and acronyms (DFU mode, activation lock, APFS, kernel panic).
3. **Adaptive Human-in-the-Loop Confidence Calibration**:
   * Implement conformal prediction to guarantee a mathematical bound on the False Auto-Handle Rate (e.g. guaranteeing that at most 2% of escalated queries are erroneously auto-handled).
4. **Live A/B Shadow Pipeline & Latency Optimization**:
   * Deploy the pipeline in a shadow mode comparing live human agent responses against agent drafts, measuring time-to-first-response (TTFR) reduction and human agent edit distance.
