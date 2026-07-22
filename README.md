# example3

import pandas as pd
from pathlib import Path

# ==========================================================
# Configuration
# ==========================================================

input_folder = Path(
    r"C:\Users\JMende95\OneDrive - JNJ\Desktop\ard_data"
)

input_file = input_folder / "54781532UC02001_jak_uc_ard_20260622.xlsx"

output_folder = input_folder / "csv"
output_folder.mkdir(exist_ok=True)

# ==========================================================
# Validate input file
# ==========================================================

if not input_file.exists():
    raise FileNotFoundError(
        f"Excel file not found:\n{input_file}"
    )

print(f"Processing: {input_file.name}")

# ==========================================================
# Convert selected ARD Excel file
# ==========================================================

try:
    # Read worksheets
    ard_df = pd.read_excel(
        input_file,
        sheet_name="ARD"
    )

    dict_df = pd.read_excel(
        input_file,
        sheet_name="PARAMCD_DICT"
    )

    # Base filename without extension
    base_name = input_file.stem

    # Output filenames
    ard_output = output_folder / f"{base_name}_ARD.csv"
    dict_output = output_folder / f"{base_name}_PARAMCD_DICT.csv"

    # Export CSV files
    ard_df.to_csv(
        ard_output,
        index=False,
        encoding="utf-8-sig"
    )

    dict_df.to_csv(
        dict_output,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nConversion completed successfully.")
    print(f"ARD CSV            : {ard_output}")
    print(f"PARAMCD dictionary : {dict_output}")

except ValueError as error:
    print("\nConversion failed.")
    print(
        "Confirm that the Excel file contains the worksheets "
        "'ARD' and 'PARAMCD_DICT'."
    )
    print(f"Details: {error}")

except Exception as error:
    print("\nConversion failed.")
    print(f"Details: {error}")
