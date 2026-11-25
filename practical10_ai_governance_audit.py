
from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 23
OUT = Path("./practical10_output")

CONTROL_RULES = [
    ("F1", "Disparate impact ratio is at least 0.80", "HIGH", "Fairness"),
    ("F2", "Demographic parity difference is at most 0.10", "MEDIUM", "Fairness"),
    ("F3", "Equalized odds difference is at most 0.10", "MEDIUM", "Fairness"),
    ("A1", "Classification accuracy is at least 0.85", "MEDIUM", "Performance"),
    ("P1", "Differential privacy epsilon is at most 10, or an alternative is documented", "HIGH", "Privacy"),
    ("P2", "Privacy delta is below the reciprocal training-set size", "MEDIUM", "Privacy"),
    ("K1", "Risk management process documented", "HIGH", "Governance"),
    ("K2", "Training data provenance and representation reviewed", "HIGH", "Governance"),
    ("K3", "Technical documentation available", "HIGH", "Governance"),
    ("K4", "Inference events logged", "MEDIUM", "Governance"),
    ("K5", "User-facing transparency information available", "MEDIUM", "Governance"),
    ("K6", "Human oversight and override available", "HIGH", "Governance"),
    ("K7", "Conformity review completed where applicable", "HIGH", "Governance"),
    ("D1", "Consent withdrawal reflected in training snapshots", "HIGH", "Data rights"),
    ("D2", "Personal data retention and deletion process documented", "MEDIUM", "Data rights"),
    ("I1", "Incident handling process documented", "HIGH", "Operations"),
    ("I2", "Impact assessment recorded", "HIGH", "Operations"),
]
ALL_CONTROLS = {row[0] for row in CONTROL_RULES}


def profile(system_id: str, name: str, domain: str, purpose: str, training_rows: int,
            accuracy: float, source: str, *, fairness=None, privacy=None,
            alternative_privacy=False, prohibited=False, flags=None) -> dict:
    return {
        "system_id": system_id, "name": name, "domain": domain, "purpose": purpose,
        "training_rows": training_rows, "accuracy": accuracy, "source": source,
        "fairness": fairness, "privacy": privacy,
        "alternative_privacy_controls": alternative_privacy, "prohibited_use": prohibited,
        "processes_personal_data": training_rows > 0,
        "flags": {"risk_management": True, "data_governance": True,
                  "technical_documentation": True, "logging": True,
                  "transparency": True, "human_oversight": True,
                  "conformity_assessment": True, "incident_process": True,
                  "impact_assessment": True, "consent_mechanism": True, **(flags or {})},
    }


def make_registry() -> list[dict]:
    return [
        profile("REC-BASE", "Recruitment baseline classifier", "employment", "rank job applicants", 17094, .87,
                "Practical 8 synthetic metrics", fairness={"di": .87, "dp": .17, "eo": .06},
                alternative_privacy=True, flags={"logging": False, "conformity_assessment": False}),
        profile("REC-DP", "Recruitment demographic parity mitigation", "employment", "rank job applicants", 17094, .85,
                "Practical 8 synthetic metrics", fairness={"di": 1.00, "dp": .004, "eo": .30}, alternative_privacy=True),
        profile("REC-EO", "Recruitment equalized odds mitigation", "employment", "rank job applicants", 17094, .86,
                "Practical 8 synthetic metrics", fairness={"di": .86, "dp": .10, "eo": .008}, alternative_privacy=True),
        profile("MED-BASE", "Medical image classifier", "healthcare", "support image review", 60000, .983,
                "Practical 9 synthetic metrics", flags={"transparency": False, "conformity_assessment": False,
                                                       "incident_process": False}),
        profile("MED-DP-1", "Medical image classifier DP-SGD epsilon 1", "healthcare", "support image review", 60000, .943,
                "Practical 9 synthetic metrics", privacy={"epsilon": 1.0, "delta": 1e-5, "noise_multiplier": 1.06}),
        profile("MED-DP-10", "Medical image classifier DP-SGD epsilon 10", "healthcare", "support image review", 60000, .944,
                "Practical 9 synthetic metrics", privacy={"epsilon": 10.0, "delta": 1e-5, "noise_multiplier": .49}),
        profile("ASSIST-GPT2", "Writing assistant", "general assistant", "generate user-requested text", 0, .90,
                "Simulated", alternative_privacy=True),
        profile("SOCIAL-DEMO", "Synthetic citizen trust scoring", "social scoring", "score citizens for service access", 5890, .80,
                "Synthetic demonstration", prohibited=True),
    ]


def validate_profiles(records: list[dict]) -> list[dict]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise SystemExit("jsonschema is required: pip install jsonschema") from exc
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
        "required": ["system_id", "name", "domain", "purpose", "training_rows", "accuracy", "flags"],
        "properties": {
            "system_id": {"type": "string", "pattern": "^[A-Z][A-Z0-9-]{2,19}$"},
            "name": {"type": "string", "minLength": 1}, "domain": {"type": "string", "minLength": 1},
            "purpose": {"type": "string", "minLength": 1},
            "training_rows": {"type": "integer", "minimum": 0},
            "accuracy": {"type": "number", "minimum": 0, "maximum": 1},
            "flags": {"type": "object", "required": ["risk_management", "data_governance", "technical_documentation",
                "logging", "transparency", "human_oversight", "conformity_assessment", "incident_process",
                "impact_assessment", "consent_mechanism"], "additionalProperties": {"type": "boolean"}},
            "prohibited_use": {"type": "boolean"},
        },
    }
    validator = Draft202012Validator(schema)
    results = []
    for item in records:
        errors = sorted(validator.iter_errors(item), key=lambda e: list(e.path))
        results.append({"system_id": item.get("system_id", "<missing>"), "status": "VALID" if not errors else "INVALID",
                        "errors": [error.message for error in errors]})
    return results


def tier_for(item: dict) -> str:
    if item.get("prohibited_use") or item["domain"].lower() == "social scoring":
        return "UNACCEPTABLE"
    if item["domain"].lower() in {"healthcare", "employment", "education", "critical infrastructure"}:
        return "HIGH"
    if item["domain"].lower() == "general assistant":
        return "LIMITED"
    return "MINIMAL"


def simulate_consent(item: dict, count: int = 500) -> dict:
    rng = np.random.default_rng(SEED + sum(map(ord, item["system_id"])))
    revoked = rng.random(count) < .08
    days = np.where(revoked, rng.integers(1, 366, count), 0)
    leak_rate = .08 if item["system_id"] == "MED-BASE" else .10 if item["system_id"] == "REC-BASE" else .01
    in_snapshot = np.where(~revoked, True, np.where(days <= 30, rng.random(count) < .5, rng.random(count) < leak_rate))
    violations = revoked & in_snapshot & (days > 30)
    return {"records": count, "revoked_records": int(revoked.sum()), "violations": int(violations.sum()),
            "grace_period_days": 30}


def evaluate(item: dict, consent: dict) -> dict[str, str]:
    tier = tier_for(item)
    flags = item["flags"]
    fairness = item.get("fairness") or {}
    results: dict[str, str] = {}
    for code, _description, _severity, category in CONTROL_RULES:
        # Fairness controls apply to decisions about people; privacy controls apply when personal data is used.
        if tier == "UNACCEPTABLE":
            results[code] = "N/A"
            continue
        if category == "Fairness" and item["domain"] not in {"employment", "healthcare", "education"}:
            results[code] = "N/A"; continue
        if category == "Data rights" and not item["processes_personal_data"]:
            results[code] = "N/A"; continue
        if code == "F1":
            value = fairness.get("di")
            results[code] = "NOT_ASSESSED" if value is None else "PASS" if value >= .80 else "FAIL"
        elif code == "F2":
            value = fairness.get("dp")
            results[code] = "NOT_ASSESSED" if value is None else "PASS" if value <= .10 else "FAIL"
        elif code == "F3":
            value = fairness.get("eo")
            results[code] = "NOT_ASSESSED" if value is None else "PASS" if value <= .10 else "FAIL"
        elif code == "A1":
            results[code] = "PASS" if item["accuracy"] >= .85 else "FAIL"
        elif code == "P1":
            privacy = item.get("privacy")
            passes = (privacy and privacy["epsilon"] <= 10) or item.get("alternative_privacy_controls")
            results[code] = "PASS" if passes else "FAIL"
        elif code == "P2":
            privacy = item.get("privacy")
            results[code] = "N/A" if not privacy else "PASS" if privacy["delta"] < 1 / max(item["training_rows"], 1) else "FAIL"
        elif code == "D1":
            results[code] = "FAIL" if consent["violations"] else "PASS"
        elif code == "D2":
            results[code] = "PASS" if flags["consent_mechanism"] else "FAIL"
        else:
            flag_key = {"K1": "risk_management", "K2": "data_governance", "K3": "technical_documentation",
                        "K4": "logging", "K5": "transparency", "K6": "human_oversight",
                        "K7": "conformity_assessment", "I1": "incident_process", "I2": "impact_assessment"}[code]
            results[code] = "PASS" if flags[flag_key] else "FAIL"
    return results


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = make_registry()
    validation = validate_profiles(registry)
    bad = copy.deepcopy(registry[0]); bad["system_id"] = "bad id"; bad["accuracy"] = 1.4; del bad["flags"]["logging"]
    negative_validation = validate_profiles([bad])[0]
    if negative_validation["status"] != "INVALID" or len(negative_validation["errors"]) < 3:
        raise RuntimeError("Negative schema check failed to detect the malformed profile")
    if any(result["status"] != "VALID" for result in validation):
        raise RuntimeError("At least one model registry profile failed schema validation")
    audits = []
    matrix = {}
    for item in registry:
        consent = simulate_consent(item)
        control_results = evaluate(item, consent)
        applicable = [value for value in control_results.values() if value != "N/A"]
        passed = sum(value == "PASS" for value in applicable)
        failed = sum(value == "FAIL" for value in applicable)
        not_assessed = sum(value == "NOT_ASSESSED" for value in applicable)
        score = passed / len(applicable) if applicable else None
        audits.append({"system_id": item["system_id"], "name": item["name"], "domain": item["domain"],
                       "tier": tier_for(item), "applicable_controls": len(applicable), "passed": passed,
                       "failed": failed, "not_assessed": not_assessed, "compliance_score": score,
                       "readiness": "PROHIBITED" if tier_for(item) == "UNACCEPTABLE" else
                           "READY" if failed == 0 and not_assessed == 0 else "GAPS IDENTIFIED",
                       "consent_audit": consent})
        matrix[item["system_id"]] = control_results
    audit_df = pd.DataFrame(audits)
    audit_df.to_json(OUT / "governance_report.json", orient="records", indent=2)
    audit_df.drop(columns="consent_audit").to_csv(OUT / "governance_summary.csv", index=False)
    with (OUT / "model_registry.json").open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2)
    pd.DataFrame(matrix).to_csv(OUT / "control_matrix.csv")
    with (OUT / "schema_validation.json").open("w", encoding="utf-8") as handle:
        json.dump({"registry_profiles": validation, "negative_test": negative_validation}, handle, indent=2)
    print("SYNTHETIC AI GOVERNANCE REVIEW | Student ID: 23AIML014")
    print(audit_df[["system_id", "tier", "applicable_controls", "passed", "failed", "not_assessed", "readiness"]].to_string(index=False))
    print(f"\nSchema validation passed for {len(validation)} registry profiles; malformed-record check caught {len(negative_validation['errors'])} errors.")
    # Executive overview figure.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_df = audit_df[audit_df.compliance_score.notna()]
    colors = ["#b23a48" if tier == "UNACCEPTABLE" else "#367da3" if tier == "HIGH" else "#55a868" for tier in plot_df.tier]
    axes[0].bar(plot_df.system_id, plot_df.compliance_score, color=colors)
    axes[0].set(ylim=(0, 1.05), ylabel="Passes / applicable controls", title="Control completion by system")
    axes[0].tick_params(axis="x", rotation=35); axes[0].grid(axis="y", alpha=.2)
    axes[1].bar(audit_df.system_id, [a["consent_audit"]["violations"] for a in audits], color="#d17a32")
    axes[1].set(ylabel="Revoked records retained past 30 days", title="Synthetic consent withdrawal check")
    axes[1].tick_params(axis="x", rotation=35); axes[1].grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / "governance_dashboard.png", dpi=180); plt.close(fig)
    print(f"Saved reports, registry, matrix, and dashboard under {OUT.resolve()}")


if __name__ == "__main__":
    main()
