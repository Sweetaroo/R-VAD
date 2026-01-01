import argparse
import json
from pathlib import Path
from typing import Dict, List


LABELS = [
    "ALERT_CRITICAL_OMISSION",
    "ALERT_CONTRADICTION",
    "UNVERIFIED",
    "CONSISTENT",
]

PRED_TO_LABEL = {
    ("ALERT", "CRITICAL_OMISSION"): "ALERT_CRITICAL_OMISSION",
    ("ALERT", "CONTRADICTION"): "ALERT_CONTRADICTION",
    ("UNVERIFIED", "NONVISUAL_UNVERIFIED"): "UNVERIFIED",
    ("CONSISTENT", "NONE"): "CONSISTENT",
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_pred_map(path: Path) -> Dict[str, Dict]:
    pred_map = {}
    for row in read_jsonl(path):
        pred_map[row.get("id")] = row
    return pred_map


def pred_label(row: Dict) -> str:
    verdict = row.get("verdict")
    reason = row.get("reason")
    label = PRED_TO_LABEL.get((verdict, reason))
    if label is None:
        return "UNVERIFIED"
    return label


def compute_metrics(gt_rows: List[Dict], pred_map: Dict[str, Dict]):
    confusion = {gt: {pr: 0 for pr in LABELS} for gt in LABELS}
    counts = {label: 0 for label in LABELS}

    uar_total = 0
    uar_consistent = 0
    co_total = 0
    co_correct = 0
    cr_total = 0
    cr_correct = 0

    for row in gt_rows:
        gt_label = row.get("gt_label")
        if gt_label not in LABELS:
            continue
        pred_row = pred_map.get(row.get("id"))
        if not pred_row:
            continue
        pred_lab = pred_label(pred_row)

        confusion[gt_label][pred_lab] += 1
        counts[gt_label] += 1

        if gt_label in {
            "ALERT_CRITICAL_OMISSION",
            "ALERT_CONTRADICTION",
            "UNVERIFIED",
        }:
            uar_total += 1
            if pred_lab == "CONSISTENT":
                uar_consistent += 1

        if gt_label == "ALERT_CRITICAL_OMISSION":
            co_total += 1
            if pred_lab == "ALERT_CRITICAL_OMISSION":
                co_correct += 1

        if gt_label == "ALERT_CONTRADICTION":
            cr_total += 1
            if pred_lab == "ALERT_CONTRADICTION":
                cr_correct += 1

    uar = uar_consistent / uar_total if uar_total else 0.0
    co_recall = co_correct / co_total if co_total else 0.0
    cr = cr_correct / cr_total if cr_total else 0.0

    return confusion, counts, uar, co_recall, cr


def extract_claim_items(gt_explicit_claims):
    items = []
    for entry in gt_explicit_claims or []:
        if not isinstance(entry, str):
            continue
        parts = entry.split(":")
        if len(parts) != 2:
            continue
        risk_item, polarity = parts
        if polarity == "ASSERT_ABSENT":
            items.append(risk_item)
    return items


def compute_per_risk_breakdown(gt_rows: List[Dict], pred_map: Dict[str, Dict]):
    breakdown = {}
    for row in gt_rows:
        gt_label = row.get("gt_label")
        pred_row = pred_map.get(row.get("id"))
        if not pred_row:
            continue
        pred_lab = pred_label(pred_row)

        if gt_label == "ALERT_CONTRADICTION":
            items = extract_claim_items(row.get("gt_explicit_claims", []))
            for item in items:
                stats = breakdown.setdefault(
                    item, {"contradiction_total": 0, "contradiction_hit": 0, "omission_total": 0, "omission_hit": 0}
                )
                stats["contradiction_total"] += 1
                if pred_lab == "ALERT_CONTRADICTION":
                    stats["contradiction_hit"] += 1
        elif gt_label == "ALERT_CRITICAL_OMISSION":
            items = row.get("gt_critical_present", []) or []
            for item in items:
                stats = breakdown.setdefault(
                    item, {"contradiction_total": 0, "contradiction_hit": 0, "omission_total": 0, "omission_hit": 0}
                )
                stats["omission_total"] += 1
                if pred_lab == "ALERT_CRITICAL_OMISSION":
                    stats["omission_hit"] += 1

    return breakdown


def compute_contradiction_diagnostics(gt_rows: List[Dict], pred_map: Dict[str, Dict]):
    total = 0
    with_claims = 0
    with_sentinel_present = 0
    with_sentinel_uncertain = 0

    for row in gt_rows:
        if row.get("gt_label") != "ALERT_CONTRADICTION":
            continue
        pred_row = pred_map.get(row.get("id"))
        if not pred_row:
            continue
        total += 1
        claims = pred_row.get("claims_extracted", [])
        if claims:
            with_claims += 1

        gt_items = extract_claim_items(row.get("gt_explicit_claims", []))
        sentinel_detected = set(pred_row.get("sentinel_detected", []))
        if any(item in sentinel_detected for item in gt_items):
            with_sentinel_present += 1

        evidence = pred_row.get("evidence", [])
        for ev in evidence:
            if ev.get("present") != "uncertain":
                continue
            if ev.get("risk_item") in gt_items:
                with_sentinel_uncertain += 1
                break

    return total, with_claims, with_sentinel_present, with_sentinel_uncertain


def print_confusion(confusion):
    header = ["GT\\PRED"] + LABELS
    print("\t".join(header))
    for gt in LABELS:
        row = [gt] + [str(confusion[gt][pr]) for pr in LABELS]
        print("\t".join(row))


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions against gt.")
    parser.add_argument("--gt", default="data/gt.jsonl", help="Path to gt jsonl")
    parser.add_argument(
        "--pred", default="outputs/pred_rvad.jsonl", help="Path to pred jsonl"
    )
    args = parser.parse_args()

    gt_path = Path(args.gt)
    pred_path = Path(args.pred)

    gt_rows = list(read_jsonl(gt_path))
    pred_map = load_pred_map(pred_path)

    confusion, counts, uar, co_recall, cr = compute_metrics(gt_rows, pred_map)
    breakdown = compute_per_risk_breakdown(gt_rows, pred_map)
    diag_total, diag_claims, diag_present, diag_uncertain = compute_contradiction_diagnostics(
        gt_rows, pred_map
    )

    print("Metrics")
    print(f"UAR: {uar:.4f}")
    print(f"CO-Recall: {co_recall:.4f}")
    print(f"CR: {cr:.4f}")
    print("\nCounts")
    for label in LABELS:
        print(f"{label}: {counts[label]}")
    print("\nConfusion Matrix")
    print_confusion(confusion)

    cr_to_co = confusion["ALERT_CONTRADICTION"]["ALERT_CRITICAL_OMISSION"]
    cr_total = counts["ALERT_CONTRADICTION"]
    print(f"\nCR->CO misclassifications: {cr_to_co}/{cr_total}")

    if diag_total:
        print(
            "\nContradiction diagnostics "
            f"(n={diag_total}): claims={diag_claims/diag_total:.4f}, "
            f"sentinel_present={diag_present/diag_total:.4f}, "
            f"sentinel_uncertain={diag_uncertain/diag_total:.4f}"
        )

    if breakdown:
        print("\nPer-Risk Breakdown")
        for item in sorted(breakdown.keys()):
            stats = breakdown[item]
            c_total = stats["contradiction_total"]
            c_hit = stats["contradiction_hit"]
            o_total = stats["omission_total"]
            o_hit = stats["omission_hit"]
            c_rate = c_hit / c_total if c_total else 0.0
            o_rate = o_hit / o_total if o_total else 0.0
            print(
                f"{item}: contradiction {c_hit}/{c_total} ({c_rate:.4f}), "
                f"omission {o_hit}/{o_total} ({o_rate:.4f})"
            )


if __name__ == "__main__":
    main()
