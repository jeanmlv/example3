# example3

# =============================================================================
# CLINICAL VARIABLE MAPPING PIPELINE — MVP v1
# Pilot study: UNIFI-JR / CNTO1275PUC3001
#
# INPUTS
#   1. Gap Analysis Excel
#   2. define.pdf
#   3. acrf.pdf
#
# WORKFLOW
#   Variables of Interest (A/B)
#          ↓
#   Filter G == "N"
#          ↓
#   Search Define + aCRF
#          ↓
#   Exact / token / fuzzy / semantic-like matching
#          ↓
#   Candidate ranking
#          ↓
#   Human-review Excel report
#
# IMPORTANT:
#   This script DOES NOT automatically change G from N to Y.
#   AI findings are suggestions requiring review.
# =============================================================================

from pathlib import Path
import re
import math
import pandas as pd

from pypdf import PdfReader
from rapidfuzz import fuzz


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # ---- Input files ---------------------------------------------------------
    "excel_file": Path("uc_variables_gap_analysis.xlsx"),
    "define_pdf": Path("define.pdf"),
    "acrf_pdf": Path("acrf.pdf"),

    # ---- Excel ---------------------------------------------------------------
    "sheet_name": "UNIFI-JR",

    # Excel columns:
    # A = Variable of Interest
    # B = Variable Description
    # C = ARD Variable
    # D = ARD Variable Description
    # E = Statistical Approach
    # G = Found in ARD? (Y/N)
    #
    # We use positional indexes so the script does not depend on exact
    # column header spelling.
    "voi_col": 0,           # A
    "voi_desc_col": 1,      # B
    "ard_var_col": 2,       # C
    "ard_desc_col": 3,      # D
    "stat_approach_col": 4, # E
    "found_col": 6,         # G

    # ---- Search --------------------------------------------------------------
    "top_k": 5,

    # How many lines around a hit to return as evidence
    "context_window": 4,

    # Minimum score for inclusion in report
    "minimum_score": 25,

    # ---- Output --------------------------------------------------------------
    "output_file": Path("UNIFI_JR_AI_variable_mapping_report.xlsx"),
}


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

STOPWORDS = {
    "a", "an", "and", "or", "of", "the", "for", "to", "in", "on",
    "with", "without", "from", "by", "at", "as", "is", "are",
    "analysis", "approach", "primary", "alternative", "estimand"
}


def normalize_text(value):
    """Normalize text for comparison."""
    if pd.isna(value):
        return ""

    value = str(value).upper()

    # Replace punctuation with spaces
    value = re.sub(r"[^A-Z0-9]+", " ", value)

    # Collapse spaces
    value = re.sub(r"\s+", " ", value).strip()

    return value


def tokenize(value):
    """Create meaningful search tokens."""
    text = normalize_text(value)

    tokens = []

    for token in text.split():

        if token.lower() in STOPWORDS:
            continue

        if len(token) < 3:
            continue

        tokens.append(token)

    return tokens


# =============================================================================
# PDF EXTRACTION
# =============================================================================

def extract_pdf_pages(pdf_path):
    """
    Extract PDF text page by page.

    Returns
    -------
    list[dict]

    Example:
        {
            "page": 74,
            "text": "..."
        }
    """

    print(f"\nReading PDF: {pdf_path}")

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = re.sub(r"\s+", " ", text).strip()

        pages.append({
            "page": page_number,
            "text": text
        })

    print(f"  Pages extracted: {len(pages)}")

    return pages


# =============================================================================
# BUILD SEARCH CHUNKS
# =============================================================================

def build_chunks(pages, source_name):
    """
    Split PDF pages into smaller searchable chunks.

    A page is divided by sentences / text fragments.
    """

    chunks = []

    for page in pages:

        page_number = page["page"]
        text = page["text"]

        if not text:
            continue

        # Split text into reasonably sized fragments
        fragments = re.split(
            r"(?<=[\.\:\;])\s+|\s{2,}",
            text
        )

        fragments = [
            x.strip()
            for x in fragments
            if len(x.strip()) >= 20
        ]

        # Sliding context
        for i, fragment in enumerate(fragments):

            start = max(0, i - CONFIG["context_window"])
            end = min(
                len(fragments),
                i + CONFIG["context_window"] + 1
            )

            context = " ".join(fragments[start:end])

            chunks.append({
                "source": source_name,
                "page": page_number,
                "text": fragment,
                "context": context,
                "normalized": normalize_text(context)
            })

    return chunks


# =============================================================================
# SEARCH SCORING
# =============================================================================

def token_overlap_score(query_tokens, candidate_text):
    """
    Percentage of query tokens found in candidate text.
    """

    if not query_tokens:
        return 0

    candidate = normalize_text(candidate_text)

    matches = sum(
        token in candidate
        for token in query_tokens
    )

    return 100 * matches / len(query_tokens)


def fuzzy_score(query, candidate):
    """RapidFuzz similarity."""

    query = normalize_text(query)
    candidate = normalize_text(candidate)

    if not query or not candidate:
        return 0

    return fuzz.token_set_ratio(query, candidate)


def exact_bonus(variable_code, candidate):
    """
    Large bonus if VOI code itself occurs in the candidate.
    """

    code = normalize_text(variable_code)
    candidate = normalize_text(candidate)

    if not code:
        return 0

    # Word boundary search
    pattern = rf"\b{re.escape(code)}\b"

    if re.search(pattern, candidate):
        return 40

    return 0


def calculate_score(
    variable_code,
    description,
    candidate_text
):
    """
    Hybrid ranking score.

    Components:
      - exact variable code
      - token overlap
      - fuzzy similarity
    """

    description_tokens = tokenize(description)

    overlap = token_overlap_score(
        description_tokens,
        candidate_text
    )

    fuzzy = fuzzy_score(
        description,
        candidate_text
    )

    bonus = exact_bonus(
        variable_code,
        candidate_text
    )

    # Weighted hybrid score
    score = (
        0.55 * overlap +
        0.45 * fuzzy +
        bonus
    )

    return round(score, 2)


# =============================================================================
# SEARCH DOCUMENT
# =============================================================================

def search_chunks(
    variable_code,
    description,
    chunks,
    top_k=5
):

    results = []

    for chunk in chunks:

        score = calculate_score(
            variable_code,
            description,
            chunk["context"]
        )

        if score >= CONFIG["minimum_score"]:

            results.append({
                "score": score,
                "source": chunk["source"],
                "page": chunk["page"],
                "evidence": chunk["context"]
            })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# =============================================================================
# CLASSIFICATION
# =============================================================================

def classify_result(results):
    """
    Conservative classification.

    IMPORTANT:
    High score does not mean confirmed ARD mapping.
    """

    if not results:
        return "NO EVIDENCE"

    best = results[0]["score"]

    if best >= 90:
        return "STRONG CANDIDATE"

    if best >= 70:
        return "CANDIDATE"

    if best >= 50:
        return "POSSIBLE"

    return "WEAK EVIDENCE"


def review_action(classification):

    mapping = {
        "STRONG CANDIDATE": "REVIEW",
        "CANDIDATE": "REVIEW",
        "POSSIBLE": "REVIEW",
        "WEAK EVIDENCE": "ESCALATE",
        "NO EVIDENCE": "ESCALATE"
    }

    return mapping[classification]


# =============================================================================
# FORMAT RESULTS
# =============================================================================

def format_candidate(results, position):

    if len(results) < position:
        return {
            "score": None,
            "source": None,
            "page": None,
            "evidence": None
        }

    return results[position - 1]


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():

    print("=" * 80)
    print("CLINICAL VARIABLE MAPPING PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Validate files
    # -------------------------------------------------------------------------

    for key in [
        "excel_file",
        "define_pdf",
        "acrf_pdf"
    ]:

        file = CONFIG[key]

        if not file.exists():
            raise FileNotFoundError(
                f"File not found: {file}"
            )

    # -------------------------------------------------------------------------
    # 2. Read Gap Analysis
    # -------------------------------------------------------------------------

    print(
        f"\nReading Excel: "
        f"{CONFIG['excel_file']}"
    )

    df = pd.read_excel(
        CONFIG["excel_file"],
        sheet_name=CONFIG["sheet_name"]
    )

    print(f"Rows in worksheet: {len(df):,}")

    # Get actual column names by position
    voi_column = df.columns[CONFIG["voi_col"]]
    desc_column = df.columns[CONFIG["voi_desc_col"]]
    ard_var_column = df.columns[CONFIG["ard_var_col"]]
    ard_desc_column = df.columns[CONFIG["ard_desc_col"]]
    stat_column = df.columns[CONFIG["stat_approach_col"]]
    found_column = df.columns[CONFIG["found_col"]]

    print("\nColumn mapping:")
    print(f"A -> {voi_column}")
    print(f"B -> {desc_column}")
    print(f"C -> {ard_var_column}")
    print(f"D -> {ard_desc_column}")
    print(f"E -> {stat_column}")
    print(f"G -> {found_column}")

    # -------------------------------------------------------------------------
    # 3. Filter G == N
    # -------------------------------------------------------------------------

    found_status = (
        df[found_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    missing_df = df[
        found_status.eq("N")
    ].copy()

    print(
        f"\nVariables with G = N: "
        f"{len(missing_df):,}"
    )

    # -------------------------------------------------------------------------
    # 4. Extract PDFs
    # -------------------------------------------------------------------------

    define_pages = extract_pdf_pages(
        CONFIG["define_pdf"]
    )

    acrf_pages = extract_pdf_pages(
        CONFIG["acrf_pdf"]
    )

    # -------------------------------------------------------------------------
    # 5. Build searchable corpus
    # -------------------------------------------------------------------------

    print("\nBuilding search index...")

    define_chunks = build_chunks(
        define_pages,
        "DEFINE"
    )

    acrf_chunks = build_chunks(
        acrf_pages,
        "aCRF"
    )

    all_chunks = (
        define_chunks +
        acrf_chunks
    )

    print(
        f"Searchable chunks: "
        f"{len(all_chunks):,}"
    )

    # -------------------------------------------------------------------------
    # 6. Analyze each missing VOI
    # -------------------------------------------------------------------------

    report_rows = []

    for counter, (idx, row) in enumerate(
        missing_df.iterrows(),
        start=1
    ):

        variable_code = row[voi_column]
        description = row[desc_column]

        print(
            f"\n[{counter}/{len(missing_df)}] "
            f"{variable_code} | {description}"
        )

        results = search_chunks(
            variable_code=variable_code,
            description=description,
            chunks=all_chunks,
            top_k=CONFIG["top_k"]
        )

        classification = classify_result(
            results
        )

        candidate_1 = format_candidate(
            results,
            1
        )

        candidate_2 = format_candidate(
            results,
            2
        )

        candidate_3 = format_candidate(
            results,
            3
        )

        report_rows.append({

            # -------------------------------------------------------------
            # ORIGINAL DATA
            # -------------------------------------------------------------

            "VOI": variable_code,

            "VOI_DESCRIPTION":
                description,

            "CURRENT_ARD_VARIABLE":
                row[ard_var_column],

            "CURRENT_ARD_DESCRIPTION":
                row[ard_desc_column],

            "CURRENT_STATISTICAL_APPROACH":
                row[stat_column],

            "CURRENT_FOUND_STATUS":
                row[found_column],

            # -------------------------------------------------------------
            # AI RESULT
            # -------------------------------------------------------------

            "AI_CLASSIFICATION":
                classification,

            "AI_RECOMMENDATION":
                review_action(classification),

            # -------------------------------------------------------------
            # TOP CANDIDATE
            # -------------------------------------------------------------

            "AI_TOP_SCORE":
                candidate_1["score"],

            "AI_TOP_SOURCE":
                candidate_1["source"],

            "AI_TOP_PAGE":
                candidate_1["page"],

            "AI_TOP_EVIDENCE":
                candidate_1["evidence"],

            # -------------------------------------------------------------
            # CANDIDATE 2
            # -------------------------------------------------------------

            "AI_CANDIDATE_2_SCORE":
                candidate_2["score"],

            "AI_CANDIDATE_2_SOURCE":
                candidate_2["source"],

            "AI_CANDIDATE_2_PAGE":
                candidate_2["page"],

            "AI_CANDIDATE_2_EVIDENCE":
                candidate_2["evidence"],

            # -------------------------------------------------------------
            # CANDIDATE 3
            # -------------------------------------------------------------

            "AI_CANDIDATE_3_SCORE":
                candidate_3["score"],

            "AI_CANDIDATE_3_SOURCE":
                candidate_3["source"],

            "AI_CANDIDATE_3_PAGE":
                candidate_3["page"],

            "AI_CANDIDATE_3_EVIDENCE":
                candidate_3["evidence"],

            # -------------------------------------------------------------
            # HUMAN REVIEW
            # -------------------------------------------------------------

            "REVIEW_DECISION":
                "",

            "CONFIRMED_ARD_VARIABLE":
                "",

            "CONFIRMED_ARD_DESCRIPTION":
                "",

            "CONFIRMED_STATISTICAL_APPROACH":
                "",

            "REVIEW_NOTES":
                ""
        })

    # -------------------------------------------------------------------------
    # 7. Create report
    # -------------------------------------------------------------------------

    report_df = pd.DataFrame(
        report_rows
    )

    # Sort strongest evidence first
    report_df = report_df.sort_values(
        by="AI_TOP_SCORE",
        ascending=False,
        na_position="last"
    )

    # -------------------------------------------------------------------------
    # 8. Export Excel
    # -------------------------------------------------------------------------

    output = CONFIG["output_file"]

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        # Original G=N records
        missing_df.to_excel(
            writer,
            sheet_name="01_GAP_VARIABLES",
            index=False
        )

        # AI investigation
        report_df.to_excel(
            writer,
            sheet_name="02_AI_MAPPING",
            index=False
        )

        # Simple summary
        summary = (
            report_df[
                "AI_CLASSIFICATION"
            ]
            .value_counts(dropna=False)
            .rename_axis("CLASSIFICATION")
            .reset_index(name="COUNT")
        )

        summary.to_excel(
            writer,
            sheet_name="03_SUMMARY",
            index=False
        )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

    print(
        f"\nOutput created:\n"
        f"{output.resolve()}"
    )

    print("\nClassification summary:")
    print(
        report_df[
            "AI_CLASSIFICATION"
        ].value_counts()
    )


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()
