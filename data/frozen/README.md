# BoneTrace Track B Frozen Manifests

**Generation Date:** 2026-10-19  
**Split Method:** `StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)`  
**Cohort Source:** `outputs/stage1/canonical_manifest.csv` (4,068 unique records)  
**Exclusions:** Four disputed label clone groups (`CANON_0206`, `CANON_0678`, `CANON_1668`, `CANON_3339`) excised from all partitions, yielding 4,064 clean benchmark records.  
**Grouping Key:** `future_split_group_id` (enforcing strict isolation of duplicate contents and confirmed pixel clones).  
**Stratification Key:** Binary fracture label (`fractured_label` / `label`).  

## Partition Summary

| Split | Total Images | Positive (Fractured) | Negative (Non-Fractured) | Prevalence | Folds Allocated |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | **2846** | 499 | 2347 | 17.53% | Folds 0–13 (70%) |
| **Validation** | **609** | 107 | 502 | 17.57% | Folds 14–16 (15%) |
| **Test** | **609** | 107 | 502 | 17.57% | Folds 17–19 (15%) |
| **Total Cohort** | **4064** | **713** | **3351** | **17.54%** | 20 Folds (100%) |

## Class Weight (Training Split)
- **pos_weight:** `4.70340681` (2347 / 499)

## Integrity Checksums (SHA-256)
```
3921741fb7e73b6c5cca486e470e6780369c6f8c5beb854bd2559a82ca2ace9e  train.csv
04ac4fe37b35568a596bccb0ce7884b7ea82112eca9384e343fb8d0fbd432bf9  val.csv
6a32f8dcb20e8d081e8c017a04b8fc849bb2856f025d4f4554c92207ae61c089  test.csv
```
