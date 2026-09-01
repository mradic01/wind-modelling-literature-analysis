from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WoS_classified_filtrirano_s_graform_POLAZNI_PODATCI.xlsx"
)

# Traženi sheet. Skripta tolerira razliku u završnim razmacima.
INPUT_SHEET = "Podatci - gruba podjela"

DICTIONARY_FILE = (
    PROJECT_ROOT
    / "config"
    / "ai_methods.json"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"

CLASSIFIED_OUTPUT = (
    PROCESSED_DIR
    / "WoS_ai_methods_classified.xlsx"
)

METHOD_COUNTS_CSV = (
    TABLES_DIR
    / "ai_method_counts.csv"
)

METHOD_COUNTS_XLSX = (
    TABLES_DIR
    / "ai_method_counts.xlsx"
)

FAMILY_COUNTS_CSV = (
    TABLES_DIR
    / "ai_family_counts.csv"
)

FAMILY_COUNTS_XLSX = (
    TABLES_DIR
    / "ai_family_counts.xlsx"
)

FIGURE_OUTPUT_PNG = (
    FIGURES_DIR
    / "ai_methods_frequency.png"
)

FIGURE_OUTPUT_JPG = (
    FIGURES_DIR
    / "ai_methods_frequency.jpg"
)


# ============================================================================
# SOURCE COLUMNS
# ============================================================================

TITLE_COLUMN = "Article Title"
ABSTRACT_COLUMN = "Abstract"
AUTHOR_KEYWORDS_COLUMN = "Author Keywords"
KEYWORDS_PLUS_COLUMN = "Keywords Plus"
YEAR_COLUMN = "Publication Year"

SEARCH_COLUMNS = [
    TITLE_COLUMN,
    ABSTRACT_COLUMN,
    AUTHOR_KEYWORDS_COLUMN,
    KEYWORDS_PLUS_COLUMN,
]


# ============================================================================
# FIGURE SETTINGS
# ============================================================================

# Minimalan broj radova potreban da se metoda prikaže na grafu.
# Postavi na 1 za prikaz svih pronađenih metoda.
MIN_DOCUMENT_COUNT_FOR_FIGURE = 1

# Ako želiš samo najčešćih N metoda, postavi npr. 30.
# None znači da se prikazuju sve metode iznad minimalnog broja.
TOP_N_METHODS: int | None = None

FAMILY_COLOURS = {
    "Classical Machine Learning": "#4C78A8",
    "Artificial Neural Networks": "#F28E2B",
    "Deep Learning": "#E15759",
    "Physics-Informed and Hybrid AI": "#59A14F",
}


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def normalize_text(value: Any) -> str:
    """
    Normalize bibliographic text before matching.

    Normalization includes:
    - missing-value handling;
    - lower-casing;
    - unification of dash characters;
    - removal of repeated whitespace.
    """
    if pd.isna(value):
        return ""

    text = str(value).casefold()

    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("−", "-")
    text = text.replace("-", "-")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def combine_search_columns(df: pd.DataFrame) -> pd.Series:
    """
    Combine Title, Abstract, Author Keywords and Keywords Plus into one
    normalized text field.
    """
    available_columns = [
        column
        for column in SEARCH_COLUMNS
        if column in df.columns
    ]

    if TITLE_COLUMN not in available_columns:
        raise ValueError(
            f"Required column is missing: '{TITLE_COLUMN}'"
        )

    if ABSTRACT_COLUMN not in available_columns:
        raise ValueError(
            f"Required column is missing: '{ABSTRACT_COLUMN}'"
        )

    normalized_fields = []

    for column in available_columns:
        normalized_fields.append(
            df[column]
            .fillna("")
            .astype(str)
            .map(normalize_text)
        )

    combined = normalized_fields[0]

    for field in normalized_fields[1:]:
        combined = combined + " " + field

    return combined.str.replace(
        r"\s+",
        " ",
        regex=True,
    ).str.strip()


# ============================================================================
# DICTIONARY LOADING AND VALIDATION
# ============================================================================

def load_dictionary() -> dict[str, dict[str, Any]]:
    """
    Load and validate config/ai_methods.json.
    """
    if not DICTIONARY_FILE.exists():
        raise FileNotFoundError(
            "AI methods dictionary was not found:\n"
            f"{DICTIONARY_FILE}"
        )

    try:
        with DICTIONARY_FILE.open(
            "r",
            encoding="utf-8-sig",
        ) as file:
            dictionary = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            "The AI methods dictionary is not valid JSON.\n"
            f"File: {DICTIONARY_FILE}\n"
            f"Line: {error.lineno}\n"
            f"Column: {error.colno}\n"
            f"Error: {error.msg}"
        ) from error

    if not isinstance(dictionary, dict) or not dictionary:
        raise ValueError(
            "The AI methods dictionary must be a non-empty JSON object."
        )

    family_labels: set[str] = set()
    method_labels: set[str] = set()

    for family_key, family_data in dictionary.items():
        if not isinstance(family_data, dict):
            raise ValueError(
                f"Family '{family_key}' must contain a JSON object."
            )

        family_label = family_data.get("label")
        methods = family_data.get("methods")

        if not isinstance(family_label, str) or not family_label.strip():
            raise ValueError(
                f"Family '{family_key}' has no valid 'label'."
            )

        family_label = family_label.strip()

        if family_label in family_labels:
            raise ValueError(
                "Duplicate family label in dictionary: "
                f"'{family_label}'"
            )

        family_labels.add(family_label)

        if not isinstance(methods, dict) or not methods:
            raise ValueError(
                f"Family '{family_key}' has no valid 'methods' object."
            )

        for method_key, method_data in methods.items():
            if not isinstance(method_data, dict):
                raise ValueError(
                    f"Method '{method_key}' must contain a JSON object."
                )

            method_label = method_data.get("label")
            terms = method_data.get("terms")

            if not isinstance(method_label, str) or not method_label.strip():
                raise ValueError(
                    f"Method '{method_key}' has no valid 'label'."
                )

            method_label = method_label.strip()

            # Labels must be globally unique because labels become
            # Excel column names.
            if method_label in method_labels:
                raise ValueError(
                    "Duplicate method label in dictionary: "
                    f"'{method_label}'"
                )

            method_labels.add(method_label)

            if not isinstance(terms, list) or not terms:
                raise ValueError(
                    f"Method '{method_key}' must contain a non-empty "
                    "'terms' list."
                )

            cleaned_terms = []

            for term in terms:
                if not isinstance(term, str):
                    raise ValueError(
                        f"Non-string term found in method '{method_key}'."
                    )

                cleaned_term = normalize_text(term)

                if cleaned_term:
                    cleaned_terms.append(cleaned_term)

            if not cleaned_terms:
                raise ValueError(
                    f"Method '{method_key}' contains no usable terms."
                )

            # Remove duplicate terms while preserving order.
            method_data["terms"] = list(
                dict.fromkeys(cleaned_terms)
            )

    return dictionary


# ============================================================================
# MATCHING
# ============================================================================

def compile_term_pattern(term: str) -> re.Pattern[str]:
    """
    Compile one complete-word or complete-phrase matching expression.

    Boundaries prevent abbreviations such as ANN, GRU, GNN or SVM from
    matching inside unrelated longer words.
    """
    return re.compile(
        r"(?<![a-z0-9])"
        + re.escape(normalize_text(term))
        + r"(?![a-z0-9])",
        flags=re.IGNORECASE,
    )


def build_method_definitions(
    dictionary: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Flatten the hierarchical JSON dictionary into a list of methods with
    precompiled term patterns.
    """
    definitions: list[dict[str, Any]] = []

    for family_key, family_data in dictionary.items():
        family_label = family_data["label"]

        for method_key, method_data in family_data["methods"].items():
            terms = method_data["terms"]

            definitions.append(
                {
                    "family_key": family_key,
                    "family_label": family_label,
                    "method_key": method_key,
                    "method_label": method_data["label"],
                    "terms": terms,
                    "term_patterns": [
                        (
                            term,
                            compile_term_pattern(term),
                        )
                        for term in terms
                    ],
                }
            )

    return definitions


def find_matched_terms(
    text: str,
    term_patterns: list[tuple[str, re.Pattern[str]]],
) -> list[str]:
    """
    Return all terms from one method entry that occur in the publication.
    """
    return [
        term
        for term, pattern in term_patterns
        if pattern.search(text)
    ]


# ============================================================================
# CLASSIFICATION
# ============================================================================

def classify_dataset(
    analysis_text: pd.Series,
    dictionary: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Classify all publications.

    Returns:
    1. DataFrame containing only newly generated AI columns;
    2. flattened method definitions used for counting.

    Keeping generated results separate prevents duplicate-column errors.
    """
    method_definitions = build_method_definitions(dictionary)

    result_df = pd.DataFrame(
        index=analysis_text.index
    )

    method_detected_columns: dict[str, pd.Series] = {}
    method_terms_columns: dict[str, pd.Series] = {}

    family_method_labels: dict[str, list[str]] = {
        family_data["label"]: []
        for family_data in dictionary.values()
    }

    # ------------------------------------------------------------------
    # Detect individual methods
    # ------------------------------------------------------------------

    for definition in method_definitions:
        family_label = definition["family_label"]
        method_label = definition["method_label"]
        term_patterns = definition["term_patterns"]

        matched_terms = analysis_text.map(
            lambda text: find_matched_terms(
                text=text,
                term_patterns=term_patterns,
            )
        )

        detected = matched_terms.map(bool)

        method_detected_columns[method_label] = detected
        method_terms_columns[method_label] = matched_terms

        family_method_labels[family_label].append(method_label)

        result_df[f"{method_label} Detected"] = detected

        result_df[f"{method_label} Matched Terms"] = (
            matched_terms.map(
                lambda terms: "; ".join(terms)
            )
        )

    # ------------------------------------------------------------------
    # Detect families from detected methods
    # ------------------------------------------------------------------

    family_detected_columns: dict[str, pd.Series] = {}

    for family_label, method_labels in family_method_labels.items():
        family_matrix = pd.concat(
            [
                method_detected_columns[method_label]
                for method_label in method_labels
            ],
            axis=1,
        )

        family_detected = family_matrix.any(axis=1)

        family_detected_columns[family_label] = family_detected

        result_df[f"{family_label} Detected"] = family_detected

    # ------------------------------------------------------------------
    # Human-readable summary columns
    # ------------------------------------------------------------------

    method_labels_in_order = [
        definition["method_label"]
        for definition in method_definitions
    ]

    family_labels_in_order = [
        family_data["label"]
        for family_data in dictionary.values()
    ]

    result_df["AI Method Detected"] = pd.concat(
        list(method_detected_columns.values()),
        axis=1,
    ).any(axis=1)

    result_df["AI Methods"] = result_df.apply(
        lambda row: "; ".join(
            method_label
            for method_label in method_labels_in_order
            if bool(row[f"{method_label} Detected"])
        ),
        axis=1,
    )

    result_df["AI Method Count"] = result_df[
        [
            f"{method_label} Detected"
            for method_label in method_labels_in_order
        ]
    ].sum(axis=1).astype(int)

    result_df["AI Families"] = result_df.apply(
        lambda row: "; ".join(
            family_label
            for family_label in family_labels_in_order
            if bool(row[f"{family_label} Detected"])
        ),
        axis=1,
    )

    result_df["AI Family Count"] = result_df[
        [
            f"{family_label} Detected"
            for family_label in family_labels_in_order
        ]
    ].sum(axis=1).astype(int)

    def collect_all_terms(row_index: int) -> str:
        terms: list[str] = []

        for method_label in method_labels_in_order:
            terms.extend(
                method_terms_columns[method_label].loc[row_index]
            )

        return "; ".join(
            dict.fromkeys(terms)
        )

    result_df["AI Matched Terms"] = [
        collect_all_terms(index)
        for index in result_df.index
    ]

    def create_evidence(row_index: int) -> str:
        evidence: list[str] = []

        for method_label in method_labels_in_order:
            terms = method_terms_columns[method_label].loc[row_index]

            if terms:
                evidence.append(
                    f"{method_label}: {', '.join(terms)}"
                )

        return " | ".join(evidence)

    result_df["AI Classification Evidence"] = [
        create_evidence(index)
        for index in result_df.index
    ]

    # Put summary columns first.
    summary_columns = [
        "AI Method Detected",
        "AI Families",
        "AI Family Count",
        "AI Methods",
        "AI Method Count",
        "AI Matched Terms",
        "AI Classification Evidence",
    ]

    remaining_columns = [
        column
        for column in result_df.columns
        if column not in summary_columns
    ]

    result_df = result_df[
        summary_columns + remaining_columns
    ]

    return result_df, method_definitions


# ============================================================================
# STATISTICS
# ============================================================================

def create_method_counts(
    result_df: pd.DataFrame,
    method_definitions: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Count how many unique publications mention each AI method.

    Counts are calculated from result_df, not from the combined Excel
    output. Therefore input-column duplicates cannot affect statistics.
    """
    total_documents = len(result_df)
    rows: list[dict[str, Any]] = []

    for definition in method_definitions:
        family_key = definition["family_key"]
        family_label = definition["family_label"]
        method_key = definition["method_key"]
        method_label = definition["method_label"]

        detected_column = f"{method_label} Detected"

        count = int(
            result_df[detected_column]
            .fillna(False)
            .astype(bool)
            .sum()
        )

        percentage = (
            100 * count / total_documents
            if total_documents > 0
            else 0
        )

        rows.append(
            {
                "Family Key": family_key,
                "AI Family": family_label,
                "Method Key": method_key,
                "AI Method": method_label,
                "Document Count": count,
                "Percentage of Analysed Publications": round(
                    percentage,
                    3,
                ),
            }
        )

    counts_df = pd.DataFrame(rows)

    counts_df = counts_df.sort_values(
        by=[
            "Document Count",
            "AI Family",
            "AI Method",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    counts_df.to_csv(
        METHOD_COUNTS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    counts_df.to_excel(
        METHOD_COUNTS_XLSX,
        index=False,
        engine="openpyxl",
    )

    return counts_df


def create_family_counts(
    result_df: pd.DataFrame,
    dictionary: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """
    Count publications containing at least one method from each family.
    """
    total_documents = len(result_df)
    rows: list[dict[str, Any]] = []

    for family_key, family_data in dictionary.items():
        family_label = family_data["label"]
        detected_column = f"{family_label} Detected"

        count = int(
            result_df[detected_column]
            .fillna(False)
            .astype(bool)
            .sum()
        )

        percentage = (
            100 * count / total_documents
            if total_documents > 0
            else 0
        )

        rows.append(
            {
                "Family Key": family_key,
                "AI Family": family_label,
                "Document Count": count,
                "Percentage of Analysed Publications": round(
                    percentage,
                    3,
                ),
            }
        )

    counts_df = pd.DataFrame(rows)

    counts_df = counts_df.sort_values(
        by="Document Count",
        ascending=False,
    ).reset_index(drop=True)

    counts_df.to_csv(
        FAMILY_COUNTS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    counts_df.to_excel(
        FAMILY_COUNTS_XLSX,
        index=False,
        engine="openpyxl",
    )

    return counts_df


# ============================================================================
# HISTOGRAM / BAR CHART
# ============================================================================

def create_method_figure(
    method_counts: pd.DataFrame,
) -> None:
    """
    Create a horizontal bar chart of detected AI methods.

    Each bar is coloured according to one of the four AI families.
    """
    plot_df = method_counts.loc[
        method_counts["Document Count"]
        >= MIN_DOCUMENT_COUNT_FOR_FIGURE
    ].copy()

    if TOP_N_METHODS is not None:
        plot_df = plot_df.head(TOP_N_METHODS).copy()

    if plot_df.empty:
        print(
            "\nNo methods satisfy the figure threshold. "
            "The figure was not created."
        )
        return

    # Horizontal bar charts display the last row at the top, so reverse
    # the descending frequency table.
    plot_df = plot_df.sort_values(
        by="Document Count",
        ascending=True,
    )

    colours = plot_df["AI Family"].map(
        FAMILY_COLOURS
    ).fillna("#999999")

    figure_height = max(
        8,
        0.38 * len(plot_df),
    )

    fig, ax = plt.subplots(
        figsize=(13, figure_height)
    )

    bars = ax.barh(
        plot_df["AI Method"],
        plot_df["Document Count"],
        color=colours,
        edgecolor="none",
    )

    ax.set_xlabel("Number of publications")
    ax.set_ylabel("AI method")

    ax.set_title(
        "Frequency of AI methods used in wind modelling studies"
    )

    ax.grid(
        axis="x",
        alpha=0.20,
    )

    ax.set_axisbelow(True)

    maximum_count = int(
        plot_df["Document Count"].max()
    )

    label_offset = max(
        0.5,
        maximum_count * 0.006,
    )

    for bar in bars:
        count = int(bar.get_width())

        ax.text(
            count + label_offset,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            ha="left",
            fontsize=8,
        )

    legend_handles = [
        Patch(
            facecolor=colour,
            label=family,
        )
        for family, colour in FAMILY_COLOURS.items()
    ]

    ax.legend(
        handles=legend_handles,
        title="AI family",
        loc="lower right",
        frameon=True,
    )

    ax.set_xlim(
        left=0,
        right=maximum_count * 1.12,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        FIGURE_OUTPUT_JPG,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# EXCEL UTILITIES
# ============================================================================

def resolve_sheet_name(
    input_file: Path,
    requested_sheet: str,
) -> str:
    """
    Find the requested sheet while ignoring leading/trailing whitespace
    and differences in letter case.
    """
    excel_file = pd.ExcelFile(
        input_file,
        engine="openpyxl",
    )

    normalized_requested = requested_sheet.strip().casefold()

    for sheet_name in excel_file.sheet_names:
        if sheet_name.strip().casefold() == normalized_requested:
            return sheet_name

    raise ValueError(
        "Requested sheet was not found.\n"
        f"Requested: '{requested_sheet}'\n"
        f"Available sheets: {excel_file.sheet_names}"
    )


def remove_previous_ai_columns(
    source_df: pd.DataFrame,
    result_columns: list[str],
) -> pd.DataFrame:
    """
    Remove AI columns from an earlier execution before adding new ones.

    This makes the script safe to run on an already classified workbook.
    """
    columns_to_remove = [
        column
        for column in result_columns
        if column in source_df.columns
    ]

    if columns_to_remove:
        print(
            "\nRemoving AI columns from a previous run: "
            f"{len(columns_to_remove):,}"
        )

        source_df = source_df.drop(
            columns=columns_to_remove,
            errors="ignore",
        )

    # Also remove duplicated input columns, preserving the first copy.
    if source_df.columns.duplicated().any():
        duplicated_names = sorted(
            set(
                source_df.columns[
                    source_df.columns.duplicated(
                        keep=False
                    )
                ]
            )
        )

        print("\nDuplicated source columns detected:")
        for column in duplicated_names:
            print(f"  - {column}")

        print(
            "Keeping the first occurrence of each duplicated "
            "source column."
        )

        source_df = source_df.loc[
            :,
            ~source_df.columns.duplicated(
                keep="first"
            ),
        ].copy()

    return source_df


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    # ------------------------------------------------------------------
    # Check files and create folders
    # ------------------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Input Excel file was not found:\n"
            f"{INPUT_FILE}\n\n"
            "Check INPUT_FILE at the top of the script."
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------

    dictionary = load_dictionary()

    resolved_sheet = resolve_sheet_name(
        input_file=INPUT_FILE,
        requested_sheet=INPUT_SHEET,
    )

    print("Reading:")
    print(INPUT_FILE)

    print("\nSheet:")
    print(resolved_sheet)

    # ------------------------------------------------------------------
    # Read source data
    # ------------------------------------------------------------------

    source_df = pd.read_excel(
        INPUT_FILE,
        sheet_name=resolved_sheet,
        engine="openpyxl",
    )

    if TITLE_COLUMN not in source_df.columns:
        raise ValueError(
            f"Required column is missing: '{TITLE_COLUMN}'"
        )

    if ABSTRACT_COLUMN not in source_df.columns:
        raise ValueError(
            f"Required column is missing: '{ABSTRACT_COLUMN}'"
        )

    for optional_column in [
        AUTHOR_KEYWORDS_COLUMN,
        KEYWORDS_PLUS_COLUMN,
    ]:
        if optional_column not in source_df.columns:
            source_df[optional_column] = ""

    print(
        f"\nPublications analysed: {len(source_df):,}"
    )

    # ------------------------------------------------------------------
    # Prepare text and classify publications
    # ------------------------------------------------------------------

    analysis_text = combine_search_columns(
        source_df
    )

    result_df, method_definitions = classify_dataset(
        analysis_text=analysis_text,
        dictionary=dictionary,
    )

    # Remove previous copies of generated columns before concatenation.
    clean_source_df = remove_previous_ai_columns(
        source_df=source_df,
        result_columns=list(result_df.columns),
    )

    classified_df = pd.concat(
        [
            clean_source_df.reset_index(drop=True),
            result_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Verify that the final file has no duplicate columns.
    duplicated_output_columns = classified_df.columns[
        classified_df.columns.duplicated()
    ]

    if len(duplicated_output_columns) > 0:
        raise RuntimeError(
            "Duplicate columns remain after classification:\n"
            f"{sorted(set(duplicated_output_columns))}"
        )

    # ------------------------------------------------------------------
    # Save full classified dataset
    # ------------------------------------------------------------------

    classified_df.to_excel(
        CLASSIFIED_OUTPUT,
        index=False,
        engine="openpyxl",
    )

    # ------------------------------------------------------------------
    # Statistics and figure
    # ------------------------------------------------------------------

    method_counts = create_method_counts(
        result_df=result_df,
        method_definitions=method_definitions,
    )

    family_counts = create_family_counts(
        result_df=result_df,
        dictionary=dictionary,
    )

    create_method_figure(
        method_counts=method_counts,
    )

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    detected_count = int(
        result_df["AI Method Detected"]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    undetected_count = (
        len(result_df) - detected_count
    )

    detected_percentage = (
        100 * detected_count / len(result_df)
        if len(result_df) > 0
        else 0
    )

    print("\nAI-method classification completed.")

    print(
        "\nPublications with at least one explicitly detected "
        f"AI method: {detected_count:,} "
        f"({detected_percentage:.2f}%)"
    )

    print(
        "Publications without an explicitly detected AI method: "
        f"{undetected_count:,}"
    )

    print("\nAI family distribution:")

    print(
        family_counts[
            [
                "AI Family",
                "Document Count",
                "Percentage of Analysed Publications",
            ]
        ].to_string(index=False)
    )

    print("\nMost frequently detected AI methods:")

    nonzero_methods = method_counts.loc[
        method_counts["Document Count"] > 0,
        [
            "AI Family",
            "AI Method",
            "Document Count",
            "Percentage of Analysed Publications",
        ],
    ]

    if nonzero_methods.empty:
        print("No AI methods were detected.")
    else:
        print(
            nonzero_methods
            .head(40)
            .to_string(index=False)
        )

    print("\nSaved classified dataset:")
    print(CLASSIFIED_OUTPUT)

    print("\nSaved AI-method tables:")
    print(METHOD_COUNTS_CSV)
    print(METHOD_COUNTS_XLSX)

    print("\nSaved AI-family tables:")
    print(FAMILY_COUNTS_CSV)
    print(FAMILY_COUNTS_XLSX)

    print("\nSaved figures:")
    print(FIGURE_OUTPUT_PNG)
    print(FIGURE_OUTPUT_JPG)


if __name__ == "__main__":
    main()