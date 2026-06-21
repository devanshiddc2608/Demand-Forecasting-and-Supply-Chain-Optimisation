# src/build_waterfall_excel.py
# ============================================================
# Builds the 4-row Excel file needed for the Page 3 waterfall
# chart in Power BI. Run this once from the project root:
#
#   python src/build_waterfall_excel.py
#
# It reads the already-exported inventory policy CSV and writes
# a ready-to-import Excel file into outputs/powerbi/.
# ============================================================

import pandas as pd
from src.config import POWERBI_EXPORT_DIR

def build_waterfall_table():
    # ---- Load the inventory policy data you already exported ----
    input_path = f"{POWERBI_EXPORT_DIR}/pb_inventory_policy.csv"
    store_stats = pd.read_csv(input_path)
    print(f"Loaded: {input_path}")
    print(f"  {len(store_stats):,} stores found.")

    # ---- Step 1: total current cost across ALL stores ----
    # This is the starting point of the waterfall.
    current_total = store_stats["Current_SS_Cost"].sum()

    # ---- Step 2: total saving contributed by each ABC class ----
    # groupby splits the data into three buckets (A, B, C) and
    # sums Annual_Cost_Saving within each bucket separately.
    saving_by_class = store_stats.groupby("ABC_Class")["Annual_Cost_Saving"].sum()

    print("\nCurrent total SS cost:", round(current_total, 0))
    print("\nSaving by class:")
    print(saving_by_class.round(0))

    # ---- Step 3: build the 4-row table the waterfall chart needs ----
    # Savings are entered as NEGATIVE — they represent a reduction
    # from the starting total. The waterfall chart reads negative
    # values as "this step brought the total down."
    waterfall_df = pd.DataFrame({
        "StepOrder": [1, 2, 3, 4],
        "StepName": [
            "Current SS Cost",
            "A-Class Saving",
            "B-Class Saving",
            "C-Class Saving",
        ],
        "StepValue": [
            round(current_total, 0),
            -round(saving_by_class.get("A", 0), 0),
            -round(saving_by_class.get("B", 0), 0),
            -round(saving_by_class.get("C", 0), 0),
        ],
    })

    print("\nFinal waterfall table:")
    print(waterfall_df.to_string(index=False))

    # ---- Step 4: save as Excel in outputs/powerbi ----
    output_path = f"{POWERBI_EXPORT_DIR}/pb_waterfall_steps.xlsx"
    waterfall_df.to_excel(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return waterfall_df


if __name__ == "__main__":
    build_waterfall_table()