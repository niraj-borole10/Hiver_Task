"""
Diagnose failure cases of Proposed System on the Golden Dataset.
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import json
from src.pipeline import AppleSupportAgent

with open("data/golden.json", "r", encoding="utf-8") as f:
    golden = json.load(f)

agent = AppleSupportAgent()
failures = []

for item in golden:
    out = agent.handle_tweet(item["customer_text"])
    esc_match = (out["escalation_decision"] == item["ground_truth_escalation"])
    intent_match = (out["predicted_intent"] == item["ground_truth_intent"])

    if not esc_match or not intent_match:
        failures.append({
            "id": item["id"],
            "text": item["customer_text"],
            "gt_intent": item["ground_truth_intent"],
            "pred_intent": out["predicted_intent"],
            "gt_esc": item["ground_truth_escalation"],
            "pred_esc": out["escalation_decision"],
            "gt_reason": item["escalation_reason"],
            "pred_reason": out["escalation_reason_code"],
            "draft_reply": out["draft_reply"]
        })

print(f"Total failure cases: {len(failures)} out of {len(golden)}")
with open("report/failures.json", "w", encoding="utf-8") as f_out:
    json.dump(failures, f_out, indent=2)

for f in failures[:10]:
    print("---")
    print(f"ID: {f['id']}")
    safe_text = f['text'].encode('ascii', 'backslashreplace').decode('ascii')
    print(f"Text: {safe_text}")
    print(f"Intent: GT={f['gt_intent']} | Pred={f['pred_intent']}")
    print(f"Escalation: GT={f['gt_esc']} | Pred={f['pred_esc']} (GT_Reason: {f['gt_reason']} vs Pred_Reason: {f['pred_reason']})")
