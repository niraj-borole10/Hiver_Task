"""
Automated evaluation metrics for Intent Classification, Escalation Triage, and Reply Generation.
"""
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def compute_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    p_class, r_class, f_class, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=unique_labels, zero_division=0
    )

    per_class = {}
    for lbl, p, r, f, s in zip(unique_labels, p_class, r_class, f_class, supp):
        per_class[lbl] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }

    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec), 4),
        "macro_recall": round(float(rec), 4),
        "macro_f1": round(float(f1), 4),
        "per_class": per_class,
    }


def compute_escalation_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Computes critical business metrics for customer support triage:
    - Overall Accuracy
    - False Auto-Handle Rate (Dangerous errors: query needed human escalation but model auto-handled)
    - False Escalation Rate (Operational inefficiency: query was safe but model escalated)
    """
    acc = accuracy_score(y_true, y_pred)
    # Binary metrics focusing on 'HUMAN_ESCALATION' as the positive class
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label="HUMAN_ESCALATION", average="binary", zero_division=0
    )

    total = len(y_true)
    true_esc = sum(1 for y in y_true if y == "HUMAN_ESCALATION")
    true_auto = sum(1 for y in y_true if y == "AUTO_HANDLE")

    false_auto_handles = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "HUMAN_ESCALATION" and yp == "AUTO_HANDLE")
    false_escalations = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "AUTO_HANDLE" and yp == "HUMAN_ESCALATION")

    false_auto_handle_rate = false_auto_handles / max(1, true_esc)
    false_escalation_rate = false_escalations / max(1, true_auto)

    return {
        "escalation_accuracy": round(float(acc), 4),
        "escalation_precision": round(float(p), 4),
        "escalation_recall": round(float(r), 4),
        "escalation_f1": round(float(f), 4),
        "false_auto_handles": false_auto_handles,
        "false_auto_handle_rate": round(float(false_auto_handle_rate), 4),
        "false_escalations": false_escalations,
        "false_escalation_rate": round(float(false_escalation_rate), 4),
    }


def compute_text_overlap(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Computes standard word-level n-gram overlap and length compliance."""
    r1_scores = []
    r2_scores = []
    length_compliant = 0

    for ref, hyp in zip(references, hypotheses):
        if len(hyp) <= 280:
            length_compliant += 1

        ref_words = ref.lower().split()
        hyp_words = hyp.lower().split()

        # ROUGE-1 approx (unigram recall)
        common_1 = set(ref_words) & set(hyp_words)
        r1 = len(common_1) / max(1, len(set(ref_words)))
        r1_scores.append(r1)

        # ROUGE-2 approx (bigram recall)
        ref_bi = set(zip(ref_words[:-1], ref_words[1:]))
        hyp_bi = set(zip(hyp_words[:-1], hyp_words[1:]))
        common_2 = ref_bi & hyp_bi
        r2 = len(common_2) / max(1, len(ref_bi)) if len(ref_bi) > 0 else 0.0
        r2_scores.append(r2)

    return {
        "rouge_1_approx": round(float(np.mean(r1_scores)), 4),
        "rouge_2_approx": round(float(np.mean(r2_scores)), 4),
        "compliance_rate_280_chars": round(length_compliant / max(1, len(hypotheses)), 4),
    }
