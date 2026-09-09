"""
Master Evaluation Runner for Hiver Take-Home Assignment.
Executes head-to-head benchmarking across:
1. Trivial Baseline (Majority Class + Canned Reply)
2. Simple Baseline (TF-IDF + Naive Bayes + Top-1 BM25 Verbatim Reply)
3. Proposed System (AppleSupport AI Agent with Hybrid RAG + Calibrated Escalation)
Evaluates against data/golden_eval_set.json (196 items) and outputs comprehensive metrics.
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import json
import time
from typing import Dict, Any, List
import pandas as pd

from baselines.trivial import TrivialBaseline
from baselines.simple import SimpleBaseline
from src.pipeline import AppleSupportAgent
from eval.metrics import (
    compute_classification_metrics,
    compute_escalation_metrics,
    compute_text_overlap,
)
from eval.judge import ReplyQualityJudge

GOLDEN_SET_PATH = "data/golden.json"
OUTPUT_REPORT_PATH = "report/benchmark.json"


def evaluate_system(name: str, system_instance: Any, golden_set: List[Dict[str, Any]], judge: ReplyQualityJudge) -> Dict[str, Any]:
    print(f"\n--- Running Evaluation for: {name} ---")
    start_time = time.time()

    y_true_intent = [item["ground_truth_intent"] for item in golden_set]
    y_true_esc = [item["ground_truth_escalation"] for item in golden_set]
    references = [item["reference_reply"] for item in golden_set]

    y_pred_intent = []
    y_pred_esc = []
    hypotheses = []
    judge_scores = []

    for i, item in enumerate(golden_set):
        cust_query = item["customer_text"]
        out = system_instance.handle_tweet(cust_query)

        pred_intent = out["predicted_intent"]
        pred_esc = out["escalation_decision"]
        draft_reply = out["draft_reply"]

        y_pred_intent.append(pred_intent)
        y_pred_esc.append(pred_esc)
        hypotheses.append(draft_reply)

        # Evaluate via LLM-as-a-Judge
        j_eval = judge.evaluate_reply(
            customer_query=cust_query,
            predicted_intent=pred_intent,
            escalation_decision=pred_esc,
            draft_reply=draft_reply,
            reference_reply=item["reference_reply"]
        )
        judge_scores.append(j_eval)

    elapsed = round(time.time() - start_time, 2)
    print(f"Completed {len(golden_set)} evaluations in {elapsed} seconds ({elapsed/len(golden_set)*1000:.1f} ms/query).")

    # Metrics computation
    cls_metrics = compute_classification_metrics(y_true_intent, y_pred_intent)
    esc_metrics = compute_escalation_metrics(y_true_esc, y_pred_esc)
    text_metrics = compute_text_overlap(references, hypotheses)

    # Average Judge metrics
    avg_groundedness = round(sum(s["groundedness_score"] for s in judge_scores) / len(judge_scores), 2)
    avg_tone = round(sum(s["tone_empathy_score"] for s in judge_scores) / len(judge_scores), 2)
    avg_actionability = round(sum(s["actionability_score"] for s in judge_scores) / len(judge_scores), 2)
    safety_pass_rate = round(sum(1 for s in judge_scores if s["safety_passed"]) / len(judge_scores) * 100, 1)
    avg_composite_quality = round(sum(s["composite_quality_score"] for s in judge_scores) / len(judge_scores), 1)

    return {
        "system_name": name,
        "runtime_seconds": elapsed,
        "classification": cls_metrics,
        "escalation": esc_metrics,
        "text_quality": text_metrics,
        "llm_judge": {
            "avg_groundedness_5": avg_groundedness,
            "avg_tone_empathy_5": avg_tone,
            "avg_actionability_5": avg_actionability,
            "safety_pass_rate_pct": safety_pass_rate,
            "composite_quality_100": avg_composite_quality,
        }
    }


def main():
    if not os.path.exists(GOLDEN_SET_PATH):
        raise FileNotFoundError(f"Golden dataset not found at {GOLDEN_SET_PATH}. Run build_golden_set.py first.")

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    print(f"Loaded {len(golden_set)} golden evaluation samples.")

    judge = ReplyQualityJudge()

    # 1. Trivial Baseline
    trivial_sys = TrivialBaseline()
    res_trivial = evaluate_system("Trivial Baseline", trivial_sys, golden_set, judge)

    # 2. Simple Baseline
    simple_sys = SimpleBaseline()
    simple_sys.fit()
    res_simple = evaluate_system("Simple Baseline (TF-IDF + BM25)", simple_sys, golden_set, judge)

    # 3. Proposed System
    proposed_sys = AppleSupportAgent()
    res_proposed = evaluate_system("Proposed System (AppleSupport AI Agent)", proposed_sys, golden_set, judge)

    all_results = {
        "trivial_baseline": res_trivial,
        "simple_baseline": res_simple,
        "proposed_agent": res_proposed,
    }

    # Print summary comparative table
    print("\n" + "=" * 90)
    print(f"{'METRIC':<35} | {'TRIVIAL BASELINE':<16} | {'SIMPLE BASELINE':<16} | {'PROPOSED AGENT':<16}")
    print("=" * 90)

    rows = [
        ("Intent Accuracy", f"{res_trivial['classification']['accuracy']*100:.1f}%", f"{res_simple['classification']['accuracy']*100:.1f}%", f"{res_proposed['classification']['accuracy']*100:.1f}%"),
        ("Intent Macro-F1", f"{res_trivial['classification']['macro_f1']*100:.1f}%", f"{res_simple['classification']['macro_f1']*100:.1f}%", f"{res_proposed['classification']['macro_f1']*100:.1f}%"),
        ("Escalation Accuracy", f"{res_trivial['escalation']['escalation_accuracy']*100:.1f}%", f"{res_simple['escalation']['escalation_accuracy']*100:.1f}%", f"{res_proposed['escalation']['escalation_accuracy']*100:.1f}%"),
        ("Escalation Precision", f"{res_trivial['escalation']['escalation_precision']*100:.1f}%", f"{res_simple['escalation']['escalation_precision']*100:.1f}%", f"{res_proposed['escalation']['escalation_precision']*100:.1f}%"),
        ("Escalation Recall", f"{res_trivial['escalation']['escalation_recall']*100:.1f}%", f"{res_simple['escalation']['escalation_recall']*100:.1f}%", f"{res_proposed['escalation']['escalation_recall']*100:.1f}%"),
        ("False Auto-Handle Rate (Safety Risk!)", f"{res_trivial['escalation']['false_auto_handle_rate']*100:.1f}%", f"{res_simple['escalation']['false_auto_handle_rate']*100:.1f}%", f"{res_proposed['escalation']['false_auto_handle_rate']*100:.1f}%"),
        ("False Escalation Rate (Labor Cost)", f"{res_trivial['escalation']['false_escalation_rate']*100:.1f}%", f"{res_simple['escalation']['false_escalation_rate']*100:.1f}%", f"{res_proposed['escalation']['false_escalation_rate']*100:.1f}%"),
        ("Tweet 280-char Compliance Rate", f"{res_trivial['text_quality']['compliance_rate_280_chars']*100:.1f}%", f"{res_simple['text_quality']['compliance_rate_280_chars']*100:.1f}%", f"{res_proposed['text_quality']['compliance_rate_280_chars']*100:.1f}%"),
        ("ROUGE-1 Overlap (vs Reference)", f"{res_trivial['text_quality']['rouge_1_approx']:.3f}", f"{res_simple['text_quality']['rouge_1_approx']:.3f}", f"{res_proposed['text_quality']['rouge_1_approx']:.3f}"),
        ("Judge Groundedness (1-5)", f"{res_trivial['llm_judge']['avg_groundedness_5']}", f"{res_simple['llm_judge']['avg_groundedness_5']}", f"{res_proposed['llm_judge']['avg_groundedness_5']}"),
        ("Judge Empathy & Brand Tone (1-5)", f"{res_trivial['llm_judge']['avg_tone_empathy_5']}", f"{res_simple['llm_judge']['avg_tone_empathy_5']}", f"{res_proposed['llm_judge']['avg_tone_empathy_5']}"),
        ("Judge Actionability (1-5)", f"{res_trivial['llm_judge']['avg_actionability_5']}", f"{res_simple['llm_judge']['avg_actionability_5']}", f"{res_proposed['llm_judge']['avg_actionability_5']}"),
        ("Judge Safety Pass Rate", f"{res_trivial['llm_judge']['safety_pass_rate_pct']:.1f}%", f"{res_simple['llm_judge']['safety_pass_rate_pct']:.1f}%", f"{res_proposed['llm_judge']['safety_pass_rate_pct']:.1f}%"),
        ("Judge Composite Quality (0-100)", f"{res_trivial['llm_judge']['composite_quality_100']}", f"{res_simple['llm_judge']['composite_quality_100']}", f"{res_proposed['llm_judge']['composite_quality_100']}"),
    ]

    for label, v1, v2, v3 in rows:
        print(f"{label:<35} | {v1:<16} | {v2:<16} | {v3:<16}")
    print("=" * 90 + "\n")

    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Detailed benchmark metrics saved to {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
