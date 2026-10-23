#!/usr/bin/env python3
"""02_analyze_video_coverage.py
Read-only offline analysis of Thea.View video inventory CSVs.
Outputs: visit counts, cumulative scheduled-visit patient coverage, QA and HTML report.

Install: python -m pip install pandas
Run: python 02_analyze_video_coverage.py
     python 02_analyze_video_coverage.py --input video_inventory_output/all_video_inventory.csv

No API access, credentials, or video downloads are needed.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REQUIRED = {
    "study", "data_type", "video_class", "site", "patient_id", "visit",
    "filename", "item_id", "examination_date", "anatomical_region"
}
SCHEDULED = ["Screening", "I-12", "I-24", "M-44", "M-92", "M-140", "M-188", "M-236"]
SCHEDULED_ORDER = {v: i for i, v in enumerate(SCHEDULED)}
UNSCHEDULED = ["Clinical Flare", "Clinical Flare 2", "Clinical Flare 3",
               "ET-I", "ET-M", "Final Efficacy", "Test"]
OUTPUT_FILES = [
    "01_video_count_by_visit.csv", "02_patient_coverage_by_visit.csv",
    "03_raw_vs_processed_coverage.csv", "04_recording_level_inventory.csv",
    "05_quality_review.csv", "06_processor_summary.csv",
    "07_study_summary.csv", "08_analysis_methodology.md",
    "09_video_coverage_report.html", "10_analysis_summary.json",
]

def clean(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def canonical_name(filename: str) -> str:
    """Same base name for .avi/.mp4 and transcoded variants; not a content hash."""
    base = re.sub(r"\.[^.]+$", "", clean(filename), flags=re.I)
    base = re.sub(r"(?i)(?:[._ -]+transcoded)+$", "", base)
    return re.sub(r"\s+", " ", base).strip().casefold()

def normalized_visit(x: str) -> str:
    x = re.sub(r"\s+", " ", clean(x)).strip()
    lookup = {v.casefold(): v for v in SCHEDULED + UNSCHEDULED}
    return lookup.get(x.casefold(), x)

def csv_out(df: pd.DataFrame, dest: Path) -> None:
    df.to_csv(dest, index=False, encoding="utf-8-sig")

def fmt_int(x: int) -> str:
    return f"{int(x):,}"

def html_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    view = df.head(max_rows).copy().fillna("")
    for c in view.columns:
        if pd.api.types.is_numeric_dtype(view[c]) and pd.api.types.is_integer_dtype(view[c]):
            view[c] = view[c].map(lambda x: fmt_int(x) if pd.notna(x) else "")
    return view.to_html(index=False, border=0, escape=True, classes="report-table")

def main() -> int:
    p = argparse.ArgumentParser(description="Offline clinical video coverage analysis")
    p.add_argument("--input", default="video_inventory_output/all_video_inventory.csv")
    p.add_argument("--output-dir", default="video_coverage_output")
    args = p.parse_args()
    source = Path(args.input)
    if not source.is_file():
        p.error(f"Input not found: {source.resolve()}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        p.error("Missing required columns: " + ", ".join(missing))
    if df.empty:
        p.error("Input CSV contains no records.")

    for c in ["study", "data_type", "video_class", "site", "patient_id", "visit",
              "filename", "item_id", "examination_date", "anatomical_region",
              "batch", "processor", "relative_path", "needs_review"]:
        if c not in df:
            df[c] = ""
        df[c] = df[c].map(clean)

    df["visit"] = df["visit"].map(normalized_visit)
    df["data_type"] = df["data_type"].str.lower()
    df["patient_key"] = df["site"] + "::" + df["patient_id"]
    df["recording_name"] = df["filename"].map(canonical_name)
    df["scheduled_visit"] = df["visit"].isin(SCHEDULED)
    df["review_reason"] = ""

    def reasons(r: pd.Series) -> str:
        flags = []
        if not r["site"] or not r["patient_id"]:
            flags.append("MISSING_PATIENT_OR_SITE")
        if not r["visit"]:
            flags.append("MISSING_VISIT")
        if r["visit"] not in SCHEDULED and r["visit"] not in UNSCHEDULED:
            flags.append("UNKNOWN_VISIT")
        if not r["scheduled_visit"]:
            flags.append("NON_SCHEDULED_VISIT_NO_CUMULATIVE")
        if r["video_class"] == "RAW_UNCLASSIFIED":
            flags.append("RAW_UNCLASSIFIED_PROVISIONAL")
        if r["needs_review"].casefold() in {"true", "1", "yes"}:
            flags.append("SOURCE_FLAGGED_REVIEW")
        return ";".join(flags)

    df["review_reason"] = df.apply(reasons, axis=1)
    # Technical identity based on study/site/patient/visit/date/anatomy/filename stem.
    # Never deduplicate by patient+visit alone: a patient may have multiple recordings.
    key = ["study", "data_type", "site", "patient_id", "visit",
           "examination_date", "anatomical_region", "recording_name"]
    df["technical_key"] = df[key].fillna("").agg("|".join, axis=1)

    # For raw, prefer originals, then unclassified, then transcoded.
    # For processed, one representative per technical key, retaining processor details in QA.
    priority = {"RAW_ORIGINAL": 0, "RAW_UNCLASSIFIED": 1,
                "RAW_TRANSCODED": 2, "PROCESSED": 0}
    df["_priority"] = df["video_class"].map(priority).fillna(3)
    df = df.sort_values(["study", "data_type", "technical_key", "_priority", "item_id"],
                        kind="stable").reset_index(drop=True)
    df["representative"] = ~df.duplicated("technical_key", keep="first")
    rep = df[df["representative"]].copy()

    # Distinguish candidate original-only from provisional raw unclassified and
    # transcoded-only records. This is an estimated technical recording count.
    rep["count_status"] = rep["video_class"].map({
        "RAW_ORIGINAL": "ORIGINAL",
        "RAW_UNCLASSIFIED": "PROVISIONAL_UNCLASSIFIED",
        "RAW_TRANSCODED": "TRANSCODED_ONLY",
        "PROCESSED": "PROCESSED_TECHNICAL",
    }).fillna("REVIEW")

    duplicate_counts = df.groupby("technical_key", dropna=False).size()
    rep["files_per_technical_key"] = rep["technical_key"].map(duplicate_counts)
    rep["has_patient_key"] = (rep["site"] != "") & (rep["patient_id"] != "")
    rep["patient_key"] = rep["site"] + "::" + rep["patient_id"]

    visit_cols = ["study", "data_type", "visit"]
    file_counts = df.groupby(visit_cols).size().rename("video_file_items")
    recordings = rep.groupby(visit_cols).size().rename("technical_recordings")
    patient_counts = (rep[rep["has_patient_key"]].groupby(visit_cols)["patient_key"]
                      .nunique().rename("patients_at_visit"))
    original_counts = (rep[rep["video_class"] == "RAW_ORIGINAL"]
                       .groupby(visit_cols).size().rename("raw_original_recordings"))
    provisional_counts = (rep[rep["video_class"] == "RAW_UNCLASSIFIED"]
                          .groupby(visit_cols).size().rename("raw_unclassified_provisional"))
    visit = pd.concat([file_counts, recordings, patient_counts, original_counts,
                       provisional_counts], axis=1).fillna(0).astype(int).reset_index()
    visit["visit_type"] = visit["visit"].map(
        lambda v: "SCHEDULED" if v in SCHEDULED_ORDER else "UNSCHEDULED_OR_UNKNOWN")
    visit["visit_order"] = visit["visit"].map(SCHEDULED_ORDER)
    visit["_sort"] = visit["visit_order"].fillna(9999)
    visit = visit.sort_values(["study", "data_type", "_sort", "visit"]).drop(columns="_sort")
    csv_out(visit, out / OUTPUT_FILES[0])

    # Cumulative patient counts only for predefined scheduled visits.
    # Unscheduled events are separately reported and NEVER silently assigned to a time point.
    cov_rows = []
    for (study, dtype), grp in rep.groupby(["study", "data_type"]):
        scheduled = grp[grp["scheduled_visit"] & grp["has_patient_key"]]
        seen = set()
        for v in SCHEDULED:
            at_visit = set(scheduled.loc[scheduled["visit"] == v, "patient_key"])
            new = at_visit - seen
            seen |= at_visit
            sub = visit[(visit.study == study) & (visit.data_type == dtype) & (visit.visit == v)]
            n_files = int(sub["video_file_items"].iloc[0]) if len(sub) else 0
            n_recordings = int(sub["technical_recordings"].iloc[0]) if len(sub) else 0
            cov_rows.append({
                "study": study, "data_type": dtype, "visit": v,
                "visit_order": SCHEDULED_ORDER[v] + 1,
                "video_file_items": n_files,
                "technical_recordings": n_recordings,
                "patients_at_visit": len(at_visit),
                "new_patients_at_visit": len(new),
                "cumulative_distinct_patients": len(seen),
                "coverage_definition": "distinct site+patient with >=1 video in this or earlier scheduled visit",
            })
    coverage = pd.DataFrame(cov_rows)
    csv_out(coverage, out / OUTPUT_FILES[1])

    wide = coverage.pivot_table(
        index=["study", "visit", "visit_order"], columns="data_type",
        values=["technical_recordings", "patients_at_visit", "cumulative_distinct_patients"],
        aggfunc="first", fill_value=0
    )
    wide.columns = [f"{measure}_{dtype}" for measure, dtype in wide.columns]
    comparison = wide.reset_index().sort_values(["study", "visit_order"])
    for measure in ["technical_recordings", "patients_at_visit", "cumulative_distinct_patients"]:
        a, b = f"{measure}_rawdata", f"{measure}_processed"
        if a not in comparison: comparison[a] = 0
        if b not in comparison: comparison[b] = 0
        comparison[f"{measure}_processed_minus_raw"] = comparison[b] - comparison[a]
    csv_out(comparison, out / OUTPUT_FILES[2])

    rec_cols = ["study", "data_type", "video_class", "count_status", "site", "patient_id",
                "patient_key", "visit", "examination_date", "anatomical_region",
                "recording_name", "filename", "processor", "batch", "item_id",
                "files_per_technical_key", "review_reason", "relative_path"]
    csv_out(rep[rec_cols], out / OUTPUT_FILES[3])

    review = df[df["review_reason"] != ""].copy()
    review = review[["study", "data_type", "video_class", "site", "patient_id", "visit",
                     "filename", "item_id", "review_reason", "relative_path"]]
    csv_out(review, out / OUTPUT_FILES[4])

    processor = (df[df["data_type"] == "processed"]
                 .groupby(["study", "processor"], dropna=False)
                 .agg(file_items=("item_id", "size"),
                      patients=("patient_key", "nunique"),
                      visits=("visit", "nunique"))
                 .reset_index())
    csv_out(processor, out / OUTPUT_FILES[5])

    study_summary = (df.groupby(["study", "data_type"])
                     .agg(file_items=("item_id", "size"),
                          patients=("patient_key", "nunique"),
                          visits=("visit", "nunique"),
                          unique_technical_keys=("technical_key", "nunique"))
                     .reset_index())
    study_summary["raw_unclassified_files"] = study_summary.apply(
        lambda r: int(((df.study == r["study"]) & (df.data_type == r["data_type"]) &
                       (df.video_class == "RAW_UNCLASSIFIED")).sum()), axis=1)
    csv_out(study_summary, out / OUTPUT_FILES[6])

    method = """# Video coverage — methodology and limitations

## Data source
Thea.View / Girder video inventory CSV, analyzed offline. This analysis does not
read clinical SDTM/ADaM datasets, verify that a video is playable, or inspect video contents.

## Definitions
- **Video file items**: rows/items in the Girder inventory. Includes technical copies.
- **Technical recordings (estimated)**: distinct combination of study, data type,
  site, patient ID, visit, examination date, anatomical region and normalized
  filename stem (extension and trailing `transcoded` suffix removed).
- **Raw deduplication**: retain original preferentially, then unclassified, then
  transcoded-only. Unclassified raw items are PROVISIONAL, not validated originals.
- **Processed deduplication**: collapse matching technical identities across
  processor folders; different names or dates may still refer to the same clinical recording.
- **Patient identifier**: site + patient_id within each study; confirm against study metadata.
- **Patients at visit**: unique site+patient with >=1 retained technical recording.
- **Cumulative patients**: union of patients with >=1 recording at or before the
  specified SCHEDULED visit, separately for rawdata and processed. It is NOT the
  sum of visit counts. Counts include patients with an earlier video even if
  no video exists at the current visit.
- **Visit order**: Screening, I-12, I-24, M-44, M-92, M-140, M-188, M-236.
  This is a provisional QUASAR ordering and MUST be validated against the
  protocol/visit metadata before presenting as official.
- **Unscheduled visits**: Clinical Flare, ET-I, ET-M, Final Efficacy, Test etc.
  shown in visit file-count table but EXCLUDED from cumulative series because
  their clinical ordering cannot be inferred from labels alone.
- **No percentage coverage** is calculated because a denominator of expected
  patients/visits was not supplied.
- **Not proof of video availability for every visit**: metadata-based proxy only.

## Important review requirements
1. Confirm official study visit ordering, especially for other studies.
2. Review RAW_UNCLASSIFIED, missing identifiers and unusual visit labels.
3. Confirm whether distinct filename stems always mean distinct recordings.
4. Confirm processed data variants and cross-processor duplicates.
5. Confirm extraction_summary.json reports COMPLETE for both roots and no errors.
6. Do not share patient-level inventory outside approved J&J channels.
"""
    (out / OUTPUT_FILES[7]).write_text(method, encoding="utf-8")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cards = [
        ("Video file items", len(df)),
        ("Estimated technical recordings", len(rep)),
        ("Raw original files", int((df.video_class == "RAW_ORIGINAL").sum())),
        ("Raw unclassified files", int((df.video_class == "RAW_UNCLASSIFIED").sum())),
        ("Distinct site+patient IDs", int(df.groupby("study")["patient_key"].nunique().sum())),
        ("Rows flagged for review", len(review)),
    ]
    card_html = "".join(
        f'<div class="metric"><div class="number">{fmt_int(v)}</div>'
        f'<div class="label">{html.escape(k)}</div></div>' for k, v in cards
    )
    summary = {
        "generated_utc": timestamp, "input_file": str(source),
        "total_video_file_items": len(df), "estimated_technical_recordings": len(rep),
        "rows_flagged_for_review": len(review),
        "raw_unclassified_file_items": int((df.video_class == "RAW_UNCLASSIFIED").sum()),
        "scheduled_visit_order": SCHEDULED,
        "warnings": [
            "Clinical recording counts are estimates based on metadata, not video content.",
            "Visit order is provisional for QUASAR and not validated for other studies.",
            "Unscheduled visits excluded from cumulative series.",
            "No expected-patient denominator; not a completeness percentage.",
            "Check extraction_summary.json separately for full scan completeness."
        ],
        "outputs": OUTPUT_FILES
    }
    (out / OUTPUT_FILES[9]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    stylesheet = """body{font-family:Segoe UI,Arial,sans-serif;margin:36px auto;max-width:1220px;color:#182638;padding:0 24px}
h1{font-size:30px;margin-bottom:5px}h2{margin-top:38px;border-bottom:2px solid #dbe3ed;padding-bottom:8px}
.sub{color:#596879}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:26px 0}
.metric{background:#f3f6fa;border:1px solid #dbe3ed;border-radius:10px;padding:18px}
.number{font-size:28px;font-weight:700;color:#153c64}.label{color:#526377;margin-top:4px}
table{border-collapse:collapse;width:100%;font-size:12px;margin:15px 0 30px}
th{background:#163e62;color:white;text-align:left;padding:10px;position:sticky;top:0}
td{padding:8px 10px;border-bottom:1px solid #e5eaf0}tr:nth-child(even){background:#f6f8fb}
.note{background:#fff7e8;border-left:4px solid #d6a144;padding:13px 17px;line-height:1.6}
.small{color:#607086;font-size:12px}.scroll{overflow-x:auto}"""
    report = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Video Coverage | Thea.View</title><style>{stylesheet}</style></head><body>
<h1>Thea.View — Video Inventory & Patient Coverage</h1>
<p class="sub">Offline metadata analysis | Generated {html.escape(timestamp)} | Internal working draft</p>
<div class="metrics">{card_html}</div>
<div class="note"><strong>Interpretation:</strong> Technical recordings are estimated from metadata.
Cumulative patients are calculated for a provisional scheduled-visit sequence only.
Unscheduled visits are excluded from cumulative counts. This report does not establish
protocol-compliant video availability or percentage coverage.</div>
<h2>Executive summary by study and source</h2><div class="scroll">{html_table(study_summary)}</div>
<h2>Scheduled visits — cumulative distinct patients</h2><div class="scroll">{html_table(coverage, 500)}</div>
<h2>Raw vs processed — comparison</h2><div class="scroll">{html_table(comparison, 500)}</div>
<h2>All visits — file and recording counts</h2><div class="scroll">{html_table(visit, 500)}</div>
<h2>Processed data — processor overview</h2><div class="scroll">{html_table(processor)}</div>
<h2>Quality assurance and interpretation</h2>
<p>Flagged inventory rows: <strong>{fmt_int(len(review))}</strong>.
See <code>05_quality_review.csv</code> and <code>08_analysis_methodology.md</code>
for review details, deduplication definitions and limitations.</p>
<p class="small">Patient-level identifiers intentionally excluded from this HTML report.
Store CSV outputs only in approved corporate locations.</p>
</body></html>"""
    (out / OUTPUT_FILES[8]).write_text(report, encoding="utf-8")

    print(f"\nSUCCESS: {len(df):,} file items analyzed")
    print(f"Estimated technical recordings: {len(rep):,}")
    print(f"Flagged rows for review: {len(review):,}")
    print(f"Results: {out.resolve()}")
    for name in OUTPUT_FILES:
        print(f" - {name}")
    print("\nIMPORTANT: validate visit order and raw unclassified items before reporting.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
