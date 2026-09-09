# Decision Log: 14 Non-Obvious Decisions & Rationale

1. **Selected `@AppleSupport` Over `@AmazonHelp` for Grounding Rigor**
   * *Decision*: Focused on `@AppleSupport` despite `@AmazonHelp` having higher total volume in the dataset.
   * *Why*: Amazon inquiries are heavily order/courier-specific (carrier tracking, lost package IDs) which require private backend API lookups that cannot be realistically resolved on Twitter. Apple support has a rich repository of reproducible technical troubleshooting procedures, public self-serve documentation (`apple.co/*`), and crisp policy boundaries between self-serve and technician escalation.

2. **6-Intent Operational Taxonomy Over 77 Fine-Grained Classes (e.g. Banking77)**
   * *Decision*: Consolidated customer issues into 6 macro operational intents rather than adopting a 50+ fine-grained taxonomy.
   * *Why*: In customer support, an intent is only valuable if it triggers a distinct resolution workflow or routing destination. Distinguishing "iPhone 7 battery drain" from "iPhone 8 battery drain" as separate intents wastes model capacity with zero operational difference. Every intent in our taxonomy maps 1-to-1 to a distinct action.

3. **Enriched the Golden Set with a 30% Escalation Rate (Over-Indexing on Edge Cases)**
   * *Decision*: Enriched the golden evaluation set to contain ~30% escalations (58/196), even though real Twitter streams have lower natural escalation rates (~10-12%).
   * *Why*: If escalations are only 8-10%, a naive system that blindly predicts `AUTO_HANDLE` for every message achieves ~90% accuracy while being dangerously dysfunctional. Over-indexing on high-risk edge cases forces the evaluation harness to measure safety recall and penalize reckless automation.

4. **Inverted Index Implementation for BM25 Retrieval**
   * *Decision*: Built a custom inverted index (`token -> [(doc_id, tf)]`) inside `BM25Retriever` instead of using naive linear scans or heavyweight external Java Lucene servers.
   * *Why*: Naive linear scanning across 5,000 historical documents took over 12 seconds for 196 queries. The inverted index reduced retrieval latency to under 1.2 milliseconds per query, allowing full offline reproducibility with zero external server dependencies.

5. **Asymmetric Cost Penalization for Escalation Triage**
   * *Decision*: Evaluated escalation decisions using asymmetric metrics (`False Auto-Handle Rate` vs. `False Escalation Rate`) rather than aggregate accuracy.
   * *Why*: In customer support, failing to escalate a compromised Apple ID or billing dispute is a catastrophic brand and security failure. In contrast, mistakenly escalating a minor troubleshooting question to a human agent is merely a small labor cost. The model was tuned to minimize false auto-handles at all costs.

6. **Normalization of Anonymous Customer Handles (`@115712` -> `@customer`)**
   * *Decision*: Normalized all Twitter-anonymized numerical handles to `@customer` during preprocessing.
   * *Why*: Leaving raw numerical IDs in the text causes TF-IDF and embeddings to overfit to pseudo-random numbers, treating specific customer IDs as predictive tokens. Normalization preserved conversational structure while eliminating numerical leakages.

7. **Strict Whitelisting of Official Apple Support Domains (`apple.co`, `apple.com`)**
   * *Decision*: Enforced an explicit domain whitelist in both the reply generator and the LLM judge.
   * *Why*: Generative models frequently hallucinate realistic-sounding third-party or expired domains (e.g., `apple-support-fix.com`). Enforcing canonical URLs protects users from phishing and preserves brand compliance.

8. **Hard Rule Guardrails Layered on Top of Probabilistic Classifier**
   * *Decision*: Placed deterministic regex guardrails on high-risk keywords (e.g., "hacked", "stolen", "lawsuit", "refund", "burned") that take precedence over the ML model's prediction.
   * *Why*: Pure statistical classifiers suffer from long-tail blind spots. A single high-risk legal or security word must guarantee an immediate human handoff regardless of the classifier's prior probability.

9. **Deterministic Fallback Generator for 100% Offline Benchmark Reproducibility**
   * *Decision*: Designed the response generator to support both live LLM APIs and a deterministic grounded template synthesizer.
   * *Why*: The assignment requires headline results to be fully reproducible in under 15 minutes by external evaluators who may not have API keys configured. The deterministic engine delivers repeatable benchmark scores with zero external API rate limits or costs.

10. **Dual Baseline Architecture (Trivial Majority vs. Simple Naive Bayes + BM25)**
    * *Decision*: Implemented both a trivial majority-class canned responder and an uncalibrated ML/retrieval baseline.
    * *Why*: A trivial baseline exposes whether an evaluation dataset suffers from the accuracy paradox. The simple baseline tests whether naive retrieval (copy-pasting the closest past tweet) is sufficient. Comparing against both highlights the exact incremental value of structured triage and grounded generation.

11. **Separation of Groundedness, Tone, and Actionability in the LLM Judge Rubric**
    * *Decision*: Decomposed the evaluation rubric into four orthogonal axes rather than asking for a single holistic 1-5 score.
    * *Why*: Single holistic scores conflate polite writing with factual accuracy. A response can be extraordinarily empathetic yet completely useless or dangerous. Multi-axis scoring ensures that a polite hallucination fails the safety gate.

12. **Empirical Human-Judge Alignment Verification on 30 Samples**
    * *Decision*: Computed Pearson correlation ($r = 0.8294$) and MAE between human ground truth and the automated judge.
    * *Why*: An LLM judge cannot be trusted blindly without empirical proof of calibration against human experts. Validating alignment guarantees that judge scores reflect genuine human standards of support quality.

13. **Strict 280-Character Boundary Enforcement with Smart Truncation**
    * *Decision*: Implemented a smart sentence-boundary truncation algorithm ensuring all draft replies strictly adhere to Twitter's 280-character limit.
    * *Why*: Truncating mid-word or allowing replies to exceed 280 characters causes tweets to fail at the Twitter API dispatch layer. The agent achieves a 100% character compliance rate.

14. **Pre-computation and Serialization of Calibrated Classifier (`intent_classifier.joblib`)**
    * *Decision*: Pre-trained and cached the calibrated model during first run.
    * *Why*: Eliminates cold-start training overhead during continuous testing, allowing the entire 196-query benchmark suite to run in just 4.96 seconds.
