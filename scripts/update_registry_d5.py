"""
scripts/update_registry_d5.py

Updates outputs/results_registry.json with completed Task D5 lesion-dependence results.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

def update_registry():
    results_file = Path("outputs/evaluation/lesion_dependence_d5_results.json")
    if not results_file.exists():
        print(f"File not found: {results_file}")
        return

    with open(results_file, "r", encoding="utf-8") as f:
        d5_data = json.load(f)

    reg_file = Path("outputs/results_registry.json")
    with open(reg_file, "r", encoding="utf-8") as f:
        registry = json.load(f)

    # Convert registry list to dict by id for easy update/lookup
    reg_map = {item["id"]: item for item in registry}
    now_iso = datetime.now(timezone.utc).isoformat()
    commit_sha = "2cc966c508924901d8202fecedcbafae6cd70242"

    for mtype in ["baseline", "region_supervised"]:
        for s in [42, 123, 2026]:
            s_str = str(s)
            if mtype not in d5_data or s_str not in d5_data[mtype]:
                continue
            entry = d5_data[mtype][s_str]

            # 1. Mask Box
            id_box = f"e_lesion_gt_box_drop_{mtype}_seed_{s}"
            box_item = reg_map.get(id_box, {
                "id": id_box,
                "unit": "logit",
                "seed": s,
                "model": mtype,
                "cohort": "fracture_positive_107",
                "script": "scripts/run_task_d5_lesion_dependence.py",
                "commit": commit_sha,
                "description": f"Logit drop masking Ground-Truth Box ({mtype}, seed {s})"
            })
            box_item["value"] = float(entry["metric_a_drop_gt_box"]["mean"])
            box_item["ci_low"] = float(entry["metric_a_drop_gt_box"]["ci_95"][0])
            box_item["ci_high"] = float(entry["metric_a_drop_gt_box"]["ci_95"][1])
            box_item["timestamp"] = now_iso
            box_item["script"] = "scripts/run_task_d5_lesion_dependence.py"
            reg_map[id_box] = box_item

            # 2. Mask Random FOV
            id_rand = f"e_lesion_matched_random_drop_{mtype}_seed_{s}"
            rand_item = reg_map.get(id_rand, {
                "id": id_rand,
                "unit": "logit",
                "seed": s,
                "model": mtype,
                "cohort": "fracture_positive_107",
                "script": "scripts/run_task_d5_lesion_dependence.py",
                "commit": commit_sha,
                "description": f"Logit drop masking Area-Matched Random FOV Region (100 draws) ({mtype}, seed {s})"
            })
            rand_item["value"] = float(entry["metric_b_drop_rand_fov"]["mean"])
            rand_item["ci_low"] = float(entry["metric_b_drop_rand_fov"]["ci_95"][0])
            rand_item["ci_high"] = float(entry["metric_b_drop_rand_fov"]["ci_95"][1])
            rand_item["timestamp"] = now_iso
            rand_item["script"] = "scripts/run_task_d5_lesion_dependence.py"
            reg_map[id_rand] = rand_item

            # 3. Top 10% Minus Box
            id_minus = f"e_lesion_top10_minus_box_drop_{mtype}_seed_{s}"
            minus_item = reg_map.get(id_minus, {
                "id": id_minus,
                "unit": "logit",
                "seed": s,
                "model": mtype,
                "cohort": "fracture_positive_107",
                "script": "scripts/run_task_d5_lesion_dependence.py",
                "commit": commit_sha,
                "description": f"Logit drop masking Top-10% Attributed Region excluding GT Box ({mtype}, seed {s})"
            })
            minus_item["value"] = float(entry["metric_c_drop_attr_minus_box"]["mean"])
            minus_item["ci_low"] = float(entry["metric_c_drop_attr_minus_box"]["ci_95"][0])
            minus_item["ci_high"] = float(entry["metric_c_drop_attr_minus_box"]["ci_95"][1])
            minus_item["timestamp"] = now_iso
            minus_item["script"] = "scripts/run_task_d5_lesion_dependence.py"
            reg_map[id_minus] = minus_item

            # 4. Ratio a/b Cohort
            id_ratio_cohort = f"e_lesion_ratio_ab_cohort_{mtype}_seed_{s}"
            ratio_cohort_item = reg_map.get(id_ratio_cohort, {
                "id": id_ratio_cohort,
                "unit": "ratio",
                "ci_low": None,
                "ci_high": None,
                "seed": s,
                "model": mtype,
                "cohort": "fracture_positive_107",
                "commit": commit_sha,
                "description": f"Ratio (a)/(b) Mean Drop GT Box / Mean Drop Random FOV ({mtype}, seed {s})"
            })
            ratio_cohort_item["value"] = float(entry["ratio_a_over_b_cohort"])
            ratio_cohort_item["timestamp"] = now_iso
            ratio_cohort_item["script"] = "scripts/run_task_d5_lesion_dependence.py"
            reg_map[id_ratio_cohort] = ratio_cohort_item

            # 5. Ratio a/b Individual (with bootstrap CI)
            id_ratio_ind = f"e_lesion_ratio_ab_individual_{mtype}_seed_{s}"
            ratio_ind_item = reg_map.get(id_ratio_ind, {
                "id": id_ratio_ind,
                "unit": "ratio",
                "seed": s,
                "model": mtype,
                "cohort": "fracture_positive_107",
                "commit": commit_sha,
                "description": f"Mean individual sample ratio (a)/(b) with 95% bootstrap CI ({mtype}, seed {s})"
            })
            ratio_ind_item["value"] = float(entry["ratio_a_over_b_individual"]["mean"])
            ratio_ind_item["ci_low"] = float(entry["ratio_a_over_b_individual"]["ci_95"][0])
            ratio_ind_item["ci_high"] = float(entry["ratio_a_over_b_individual"]["ci_95"][1])
            ratio_ind_item["timestamp"] = now_iso
            ratio_ind_item["script"] = "scripts/run_task_d5_lesion_dependence.py"
            reg_map[id_ratio_ind] = ratio_ind_item

    # Re-save registry as list
    updated_registry = list(reg_map.values())
    with open(reg_file, "w", encoding="utf-8") as f:
        json.dump(updated_registry, f, indent=2)

    print(f"Updated registry written to {reg_file} ({len(updated_registry)} items).")

if __name__ == "__main__":
    update_registry()
