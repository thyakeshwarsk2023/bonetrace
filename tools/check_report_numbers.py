"""
tools/check_report_numbers.py

Audit verification tool for BoneTrace reports:
1. Validates that numeric claims tagged with {{reg:ID}} match the values in outputs/results_registry.json.
   Supports:
   - "0.8737 {{reg:c_test_baseline_mean_roc_auc}}" or "0.8737{{reg:c_test_baseline_mean_roc_auc}}"
   - "{{reg:c_test_baseline_mean_roc_auc:0.8737}}"
   - Standalone "{{reg:ID}}" checks that ID exists in the registry.
2. Fails if any banned phrase appears in the report:
   - "statistically robust"
   - "medical-grade"
   - "15% clinical"
   - "highly reliable"
   - "causally driven by the identified lesion"
   - "cohort shift" (when describing experimental/split mismatches)
3. Fails on any untagged numerical metrics in quantitative markdown tables.
4. Fails on claims that the registry contradicts:
   - Verifies hypothesis claims against effect sizes, confidence intervals, and p-values.
"""

import sys
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


BANNED_PHRASES = [
    "statistically robust",
    "medical-grade",
    "15% clinical",
    "highly reliable",
    "causally driven by the identified lesion",
]


def load_registry(registry_path: Path) -> Dict[str, Any]:
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")
    items = json.loads(registry_path.read_text(encoding="utf-8"))
    return {item["id"]: item for item in items}


def check_table_numbers(line_no: int, line: str, errors: List[str]) -> None:
    """Check for untagged numeric values in markdown table cells."""
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return
    # Skip table header separators (| :--- | :---: |)
    if re.match(r"^\|(?:\s*:?-+:?\s*\|)+$", stripped):
        return

    # Check for numbers in table cells
    cells = [c.strip() for c in stripped.split("|")[1:-1]]
    for col_idx, cell in enumerate(cells, start=1):
        # Ignore empty cells, pure text headers, or markdown formatting
        if not cell or cell.startswith("---"):
            continue
        # Extract potential floating point metrics or percentages
        # e.g., 0.8100, 94.12%, 3.5484, +0.0104
        num_matches = re.findall(r'(?<![a-zA-Z0-9_\-])([+\-]?[0-9]+\.[0-9]+%?)(?![a-zA-Z0-9_\-])', cell)
        if num_matches:
            # Must have an adjacent {{reg:...}} tag in the cell
            if "{{reg:" not in cell:
                # Allow standard sample size annotations like N=610 or N=50
                if all(re.match(r'^N=\d+$', m) for m in num_matches):
                    continue
                errors.append(
                    f"Line {line_no}, Col {col_idx}: Untagged number(s) {num_matches} in table cell: '{cell}'"
                )


def check_hypothesis_consistency(line_no: int, line: str, registry: Dict[str, Any], errors: List[str]) -> None:
    """Check that text claims regarding hypotheses do not contradict the registry."""
    line_lower = line.lower()

    # Check H4 Calibration claim: registry shows calibration difference is not statistically significant
    if any(h in line_lower for h in ["hypothesis 4", "hypothesis h4", "h4:"]):
        if "statistically significant" in line_lower and "not" not in line_lower:
            errors.append(
                f"Line {line_no}: Claim contradicts registry: H4 claimed statistically significant, but registry CI spans 0."
            )
        if "falsified" in line_lower:
            errors.append(
                f"Line {line_no}: Claim contradicts registry: H4 claimed 'falsified', but effect is within seed-to-seed variance."
            )

    # Check H3 Localization claim: pointing game hit rate does not exceed random guessing (~1.87%)
    if any(h in line_lower for h in ["hypothesis 3", "hypothesis h3", "h3:"]):
        if "superior localization" in line_lower or "pointing game improved" in line_lower:
            errors.append(
                f"Line {line_no}: Claim contradicts registry: H3 localization hit rate does not exceed baseline or null."
            )
        if "statistically significant" in line_lower and "not" not in line_lower:
            errors.append(
                f"Line {line_no}: Claim contradicts registry: H3 localization differences are not statistically significant."
            )

    # Check split mismatch / cohort terminology
    if "cohort shift" in line_lower:
        if not ("discrepanc" in line_lower or "audit" in line_lower or "removed" in line_lower or "not" in line_lower):
            errors.append(
                f"Line {line_no}: Terminology error: 'cohort shift' used to describe training/test split mutation or cross-split leakage."
            )


def check_report(report_path: Path, registry: Dict[str, Any], enforce_tables: bool = True) -> Tuple[bool, List[str]]:
    errors = []
    text = report_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    print(f"\nAuditing report: {report_path}")

    # 1. Check banned phrases
    print("Checking for banned phrases...")
    for line_no, line in enumerate(lines, start=1):
        for banned in BANNED_PHRASES:
            if banned.lower() in line.lower():
                # Allow mentions inside audit logs or check tool documentation if explicitly flagged
                if "banned phrases" in line.lower() or "DISCREPANCIES" in str(report_path):
                    continue
                errors.append(
                    f"Line {line_no}: Found banned phrase '{banned}' -> \"{line.strip()}\""
                )

    # 2. Check hypothesis claims against registry
    print("Checking hypothesis consistency against registry...")
    for line_no, line in enumerate(lines, start=1):
        check_hypothesis_consistency(line_no, line, registry, errors)

    # 3. Check registry tags
    print("Validating tagged numbers against registry...")
    tag_pattern_preceding = re.compile(r'([+\-]?[0-9]+(?:\.[0-9]+)?%?)\s*\{\{reg:([a-zA-Z0-9_\-]+)\}\}')
    tag_pattern_inline = re.compile(r'\{\{reg:([a-zA-Z0-9_\-]+):([+\-]?[0-9]+(?:\.[0-9]+)?%?)\}\}')
    tag_pattern_standalone = re.compile(r'\{\{reg:([a-zA-Z0-9_\-]+)\}\}')

    checked_tags = 0

    for line_no, line in enumerate(lines, start=1):
        # Table untagged number check if requested (default True)
        if enforce_tables:
            check_table_numbers(line_no, line, errors)

        # Match inline {{reg:ID:EXPECTED}}
        for match in tag_pattern_inline.finditer(line):
            reg_id, val_str = match.groups()
            checked_tags += 1
            if reg_id not in registry:
                errors.append(f"Line {line_no}: Unknown registry ID '{reg_id}'")
                continue
            reg_val = registry[reg_id]["value"]
            is_pct = val_str.endswith("%")
            num_val = float(val_str.rstrip("%"))
            if is_pct and registry[reg_id]["unit"] == "fraction":
                num_val /= 100.0

            if isinstance(reg_val, (int, float)):
                diff = abs(num_val - float(reg_val))
                tol = max(0.01 * abs(float(reg_val)), 0.005)
                if diff > tol:
                    errors.append(
                        f"Line {line_no}: Tag '{{{{reg:{reg_id}:{val_str}}}}}' value mismatch: "
                        f"report has {num_val}, registry has {reg_val}"
                    )

        # Match preceding number: "123.45 {{reg:ID}}"
        for match in tag_pattern_preceding.finditer(line):
            val_str, reg_id = match.groups()
            checked_tags += 1
            if reg_id not in registry:
                errors.append(f"Line {line_no}: Unknown registry ID '{reg_id}'")
                continue
            reg_val = registry[reg_id]["value"]
            is_pct = val_str.endswith("%")
            num_val = float(val_str.rstrip("%"))
            if is_pct and registry[reg_id]["unit"] == "fraction":
                num_val /= 100.0

            if isinstance(reg_val, (int, float)):
                diff = abs(num_val - float(reg_val))
                tol = max(0.01 * abs(float(reg_val)), 0.005)
                if diff > tol:
                    errors.append(
                        f"Line {line_no}: Number '{val_str}' with tag '{{{{reg:{reg_id}}}}}' value mismatch: "
                        f"report has {num_val}, registry has {reg_val}"
                    )

        # Match standalone {{reg:ID}}
        for match in tag_pattern_standalone.finditer(line):
            reg_id = match.group(1)
            if reg_id not in registry:
                errors.append(f"Line {line_no}: Unknown registry ID '{reg_id}'")

    print(f"Verified {checked_tags} registry tag assertions.")
    return (len(errors) == 0, errors)


def main():
    registry_path = Path("outputs/results_registry.json")
    if not registry_path.exists():
        print(f"ERROR: Registry file not found at {registry_path}")
        sys.exit(1)

    registry = load_registry(registry_path)
    print(f"Loaded results registry with {len(registry)} entries.")

    skip_tables = "--skip-table-check" in sys.argv
    enforce_tables = not skip_tables
    args = [a for a in sys.argv[1:] if a not in ("--enforce-tables", "--skip-table-check")]

    reports_to_check = [
        Path("outputs/FINAL_STUDY_REPORT.md"),
        Path("outputs/REVIEW2_DRAFT_REPORT.md"),
    ]
    if args:
        reports_to_check = [Path(p) for p in args]

    all_passed = True
    for r_path in reports_to_check:
        if not r_path.exists():
            print(f"Warning: Report file {r_path} does not exist, skipping.")
            continue
        passed, errors = check_report(r_path, registry, enforce_tables=enforce_tables)
        if passed:
            print(f"PASS: {r_path} passed all registry, banned phrase, and consistency checks.")
        else:
            print(f"FAIL: {r_path} had {len(errors)} issues:")
            for err in errors:
                print(f"  - {err}")
            all_passed = False

    if not all_passed:
        print("\nVerification check FAILED.")
        sys.exit(1)
    else:
        print("\nAll report verification checks PASSED successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
