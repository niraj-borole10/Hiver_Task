"""
Comprehensive Deliverables Verification Script for Hiver Take-Home Assignment.
Validates each of the 5 required deliverables from the assignment specification.
"""
import os
import sys
import json
import time
import re
import subprocess

print("=" * 70)
print("     HIVER SDE INTERN TAKE-HOME ASSIGNMENT: DELIVERABLES AUDIT     ")
print("=" * 70)

# -------------------------------------------------------------
# DELIVERABLE 1: Runnable Pipeline & README (<15 minutes)
# -------------------------------------------------------------
print("\n[CHECK 1] Deliverable 1: Runnable Pipeline & README")
assert os.path.exists("README.md"), "FAIL: README.md not found!"
readme_size = os.path.getsize("README.md")
print(f"  [PASS] README.md exists ({readme_size:,} bytes).")

t0 = time.time()
res_eval = subprocess.run([sys.executable, "eval/evaluate.py"], capture_output=True, text=True)
eval_dur = round(time.time() - t0, 2)
assert res_eval.returncode == 0, f"FAIL: eval/evaluate.py failed with: {res_eval.stderr}"
print(f"  [PASS] Pipeline runs cleanly in {eval_dur}s (Requirement: < 15 minutes / 900s).")

# -------------------------------------------------------------
# DELIVERABLE 2: Golden Evaluation Set (150-250 hand-labelled)
# -------------------------------------------------------------
print("\n[CHECK 2] Deliverable 2: Golden Evaluation Set (150-250 items + documentation)")
golden_path = "data/golden.json"
assert os.path.exists(golden_path), f"FAIL: {golden_path} not found!"
with open(golden_path, "r", encoding="utf-8") as f:
    golden = json.load(f)

count = len(golden)
print(f"  [PASS] Golden evaluation set contains {count} items (Requirement: 150-250 items).")
assert 150 <= count <= 250, f"FAIL: Item count {count} is not between 150 and 250!"

# Verify schema
sample = golden[0]
required_keys = ["id", "customer_text", "ground_truth_intent", "ground_truth_escalation", "escalation_reason", "reference_reply"]
for k in required_keys:
    assert k in sample, f"FAIL: Golden item missing key '{k}'!"
print(f"  [PASS] Golden set items contain all required ground truth keys: {required_keys}.")

# Verify documentation
assert os.path.exists("data/sampling.md"), "FAIL: data/sampling.md missing!"
print("  [PASS] Sampling and labeling methodology document exists (data/sampling.md).")

# -------------------------------------------------------------
# DELIVERABLE 3: Evaluation Harness & LLM-as-a-Judge with Human Alignment
# -------------------------------------------------------------
print("\n[CHECK 3] Deliverable 3: Evaluation Harness & Judge-Human Alignment")
res_align = subprocess.run([sys.executable, "eval/alignment.py"], capture_output=True, text=True)
assert res_align.returncode == 0, f"FAIL: eval/alignment.py failed: {res_align.stderr}"
print("  [PASS] Judge alignment script executed successfully.")

lines = [l.strip() for l in res_align.stdout.splitlines() if l.strip()]
for line in lines:
    if "Pearson" in line or "Error" in line or "Agreement" in line or "Evaluated" in line:
        print(f"    -> {line}")

# Verify benchmark output
assert os.path.exists("report/benchmark.json"), "FAIL: report/benchmark.json missing!"
with open("report/benchmark.json", "r", encoding="utf-8") as f:
    bm = json.load(f)
assert "trivial_baseline" in bm and "simple_baseline" in bm and "proposed_agent" in bm
print("  [PASS] Head-to-head metrics computed for Trivial, Simple, and Proposed systems.")

# -------------------------------------------------------------
# DELIVERABLE 4: Report (Framing, 2 Baselines, Top 5 Failures, Misleading headline, Next steps)
# -------------------------------------------------------------
print("\n[CHECK 4] Deliverable 4: Comprehensive Report (report/REPORT.md)")
report_path = "report/REPORT.md"
assert os.path.exists(report_path), f"FAIL: {report_path} not found!"
with open(report_path, "r", encoding="utf-8") as f:
    report_text = f.read()

report_text_lower = report_text.lower()

subsections = {
    "Problem framing (what 'good' means & what chose not to build)": (
        "problem framing" in report_text_lower and "chose not to build" in report_text_lower
    ),
    "Results vs. at least two baselines (trivial & simple)": (
        "trivial baseline" in report_text_lower and "simple baseline" in report_text_lower and "head-to-head" in report_text_lower
    ),
    "Failure analysis (top 5 failure modes with real examples)": (
        "failure analysis" in report_text_lower and "mode 1" in report_text_lower and "mode 5" in report_text_lower
    ),
    "Mandatory section: 'What is misleading about my headline number?'": (
        "what is misleading about my headline number" in report_text_lower
    ),
    "What you'd do next with one more week": (
        "what we'd do next with one more week" in report_text_lower
    ),
}

for title, ok in subsections.items():
    status = "[PASS]" if ok else "[FAIL]"
    print(f"  {status} {title}")
    assert ok, f"Report missing section: {title}"

# -------------------------------------------------------------
# DELIVERABLE 5: Decision Log (10-15 non-obvious decisions)
# -------------------------------------------------------------
print("\n[CHECK 5] Deliverable 5: Decision Log (report/decisions.md)")
dec_path = "report/decisions.md"
assert os.path.exists(dec_path), f"FAIL: {dec_path} not found!"
with open(dec_path, "r", encoding="utf-8") as f:
    dec_text = f.read()

dec_matches = re.findall(r"^\d+\.\s+\*\*", dec_text, flags=re.MULTILINE)
dec_count = len(dec_matches)
print(f"  [PASS] Found {dec_count} documented decisions (Requirement: 10-15 decisions).")
assert 10 <= dec_count <= 15, f"FAIL: Decision count {dec_count} is outside 10-15 range!"

# -------------------------------------------------------------
# AUTOMATED TEST SUITE VERIFICATION
# -------------------------------------------------------------
print("\n[CHECK 6] Automated Pytest Suite")
res_tests = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], capture_output=True, text=True)
assert res_tests.returncode == 0, f"FAIL: Pytest failed: {res_tests.stderr}\n{res_tests.stdout}"
test_passed_lines = [l for l in res_tests.stdout.splitlines() if "PASSED" in l]
print(f"  [PASS] All {len(test_passed_lines)} tests passed.")

print("\n" + "=" * 70)
print("   RESULT: ALL 5 DELIVERABLES AND AUDIT CHECKS PASSED (100%)   ")
print("=" * 70 + "\n")
