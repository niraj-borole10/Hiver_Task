# Golden Evaluation Set: Sampling & Labeling Methodology

## 1. Overview
The Golden Evaluation Set consists of **196 high-fidelity, hand-verified customer-support interactions** between real Twitter users and `@AppleSupport`. It is designed to rigorously benchmark:
1. **Intent Classification Accuracy & Macro-F1** across 6 distinct operational intents.
2. **Escalation Decision Triage** (`AUTO_HANDLE` vs. `HUMAN_ESCALATION`) and specific policy reasons.
3. **Draft Reply Quality** against historical brand resolutions.

---

## 2. Sampling Strategy
To avoid sampling bias (e.g. over-indexing on simple OS restart questions), we adopted a **stratified sampling with risk-boundary oversampling** protocol:

1. **Stratification Across Core Intents**:
   - `HARDWARE_BATTERY_CHARGING`: 35 examples
   - `SOFTWARE_OS_CRASH`: 35 examples
   - `APPLE_ID_ICLOUD_SECURITY`: 35 examples
   - `BILLING_SUBSCRIPTIONS_PURCHASES`: 35 examples
   - `CONNECTIVITY_BLUETOOTH_WIFI`: 30 examples
   - `GENERAL_PRODUCT_INFO`: 26 examples

2. **Escalation Distribution Calibration**:
   - In production, a naive model that predicts `AUTO_HANDLE` 100% of the time could achieve deceptively high accuracy if escalations are rare.
   - We intentionally sampled a challenging **~30% Human Escalation rate** (58 escalations, 138 auto-handles).
   - This ensures both `Precision` and `Recall` on escalations are statistically meaningful and penalizes reckless auto-handlers.

3. **Inclusion of High-Risk Edge Cases**:
   - **Financial disputes**: Double charges, unauthorized App Store transactions, subscription cancelation friction.
   - **Security crises**: Apple ID lockouts, 2FA recovery failures, device activation locks.
   - **Severe hardware faults**: Swollen batteries, cracked screens, liquid damage.
   - **High emotional distress**: Explicit threats to switch platforms, legal mentions, intense frustration.

---

## 3. Labeling Taxonomy & Decision Schema

| Intent Category | Typical Auto-Handle Scenario | Escalation Trigger Scenario |
| :--- | :--- | :--- |
| `HARDWARE_BATTERY_CHARGING` | Battery drain troubleshooting, settings check, diagnostic tips | Physical screen shatter, water damage, swollen battery |
| `SOFTWARE_OS_CRASH` | Minor app crashes, cache clearing, standard update restart | Boot loop, bricked device, fatal restore error |
| `APPLE_ID_ICLOUD_SECURITY` | General iCloud backup tips, self-service iforgot link | Account compromised, 2FA lockout, account takeover |
| `BILLING_SUBSCRIPTIONS_PURCHASES`| Receipt check instructions, reportaproblem portal link | Fraudulent transactions, unauthorized charges, refund disputes |
| `CONNECTIVITY_BLUETOOTH_WIFI` | Wi-Fi toggle, network settings reset, AirPods re-pairing | Carrier locked, persistent SIM failure, dropped 911/emergency calls |
| `GENERAL_PRODUCT_INFO` | Store hours, Genius Bar booking link, trade-in estimator | Stolen or lost delivery, missing pre-orders requiring order lookup |

---

## 4. Human Verification & Quality Assurance
Each example was verified for:
- Text cleanliness (handles normalized to `@customer`, links preserved, no truncated fragments).
- Exact alignment with Apple's public support policies (e.g., Apple agents never ask for passwords over Twitter; sensitive transactions always redirect to secure DM or official auth portals).
- Non-ambiguity of ground truth intent.
