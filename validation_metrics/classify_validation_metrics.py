from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WoS_classified_filtrirano_s_graform_POLAZNI_PODATCI.xlsx"
)

INPUT_SHEET = "Podatci - gruba podjela "

DICTIONARY_FILE = (
    PROJECT_ROOT
    / "config"
    / "validation_metrics.json"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"

CLASSIFIED_OUTPUT = (
    PROCESSED_DIR
    / "WoS_validation_metrics_classified.xlsx"
)

ANALYSIS_OUTPUT = (
    TABLES_DIR
    / "validation_metrics_analysis.xlsx"
)


# ---------------------------------------------------------------------
# COLUMN NAMES
# ---------------------------------------------------------------------

TITLE_COLUMN = "Article Title"
ABSTRACT_COLUMN = "Abstract"
AUTHOR_KEYWORDS_COLUMN = "Author Keywords"
KEYWORDS_PLUS_COLUMN = "Keywords Plus"
YEAR_COLUMN = "Publication Year"
CATEGORY_COLUMN = "Final Category"

CATEGORY_ORDER = [
    "Physics-based",
    "Statistical",
    "Machine Learning",
    "Hybrid",
    "Unclassified",
]


# ---------------------------------------------------------------------
# TEXT PROCESSING
# ---------------------------------------------------------------------

def normalize_text(value: Any) -> str:
    """
    Normalize text before dictionary matching.
    """
    if pd.isna(value):
        return ""

    text = str(value).casefold()

    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("−", "-")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def phrase_present(text: str, phrase: str) -> bool:
    """
    Match complete words and phrases.

    This avoids matching short abbreviations such as MSE or MAE
    inside unrelated longer words.
    """
    normalized_phrase = normalize_text(phrase)

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(normalized_phrase)
        + r"(?![a-z0-9])"
    )

    return re.search(pattern, text) is not None


def combine_text_fields(row: pd.Series) -> str:
    """
    Combine all bibliographic fields used for classification.
    """
    values = [
        row.get(TITLE_COLUMN, ""),
        row.get(ABSTRACT_COLUMN, ""),
        row.get(AUTHOR_KEYWORDS_COLUMN, ""),
        row.get(KEYWORDS_PLUS_COLUMN, ""),
    ]

    return " ".join(
        normalize_text(value)
        for value in values
        if not pd.isna(value) and str(value).strip()
    )


# ---------------------------------------------------------------------
# DICTIONARY
# ---------------------------------------------------------------------

def load_dictionary() -> dict[str, dict[str, Any]]:
    """
    Load and validate the validation-metric dictionary.
    """
    if not DICTIONARY_FILE.exists():
        raise FileNotFoundError(
            "Validation metric dictionary was not found:\n"
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
            "The validation metric dictionary is not valid JSON.\n"
            f"Line: {error.lineno}, column: {error.colno}\n"
            f"Error: {error.msg}"
        ) from error

    for metric_key, metric_data in dictionary.items():
        if "label" not in metric_data:
            raise ValueError(
                f"Metric '{metric_key}' is missing 'label'."
            )

        if "terms" not in metric_data:
            raise ValueError(
                f"Metric '{metric_key}' is missing 'terms'."
            )

        if not isinstance(metric_data["terms"], list):
            raise ValueError(
                f"Metric '{metric_key}' must contain a list of terms."
            )

    return dictionary


# ---------------------------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------------------------

def find_matches(
    text: str,
    terms: list[str],
) -> list[str]:
    """
    Return all dictionary terms found in one publication.
    """
    return sorted(
        {
            term
            for term in terms
            if phrase_present(text, term)
        }
    )


def classify_publication(
    text: str,
    dictionary: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Detect all validation metrics mentioned in one publication.

    Classification is multi-label: one publication may contain
    RMSE, MAE and R² simultaneously.
    """
    detected_labels: list[str] = []
    detected_keys: list[str] = []
    all_matched_terms: list[str] = []

    result: dict[str, Any] = {}

    for metric_key, metric_data in dictionary.items():
        label = metric_data["label"]
        matches = find_matches(
            text=text,
            terms=metric_data["terms"],
        )

        detected = bool(matches)

        if detected:
            detected_keys.append(metric_key)
            detected_labels.append(label)
            all_matched_terms.extend(matches)

        result[f"{label} Detected"] = detected
        result[f"{label} Matched Terms"] = "; ".join(matches)

    result.update(
        {
            "Validation Metric Detected": bool(detected_labels),
            "Validation Metrics": "; ".join(detected_labels),
            "Validation Metric Keys": "; ".join(detected_keys),
            "Validation Matched Terms": "; ".join(
                sorted(set(all_matched_terms))
            ),
            "Validation Metric Count": len(detected_labels),
        }
    )

    return result


# ---------------------------------------------------------------------
# DATASET VALIDATION
# ---------------------------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Verify that required columns are present.
    """
    required_columns = {
        TITLE_COLUMN,
        ABSTRACT_COLUMN,
        YEAR_COLUMN,
        CATEGORY_COLUMN,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "The input dataset is missing these columns:\n"
            f"{sorted(missing_columns)}"
        )


# ---------------------------------------------------------------------
# OUTPUT TABLES
# ---------------------------------------------------------------------

def create_overall_counts(
    classified_df: pd.DataFrame,
    dictionary: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """
    Count how many publications contain each validation metric.
    """
    rows: list[dict[str, Any]] = []

    total_publications = len(classified_df)

    for metric_data in dictionary.values():
        label = metric_data["label"]
        detected_column = f"{label} Detected"

        count = int(
            classified_df[detected_column]
            .fillna(False)
            .astype(bool)
            .sum()
        )

        rows.append(
            {
                "Validation Metric": label,
                "Document Count": count,
                "Percentage of All Publications": round(
                    100 * count / total_publications,
                    3,
                ),
            }
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "Document Count",
        ascending=False,
    )


    return result


def create_counts_matrix(
    classified_df: pd.DataFrame,
    dictionary: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """
    Create the Metric × Modelling Category count matrix.
    """
    rows: list[dict[str, Any]] = []

    for metric_data in dictionary.values():
        label = metric_data["label"]
        detected_column = f"{label} Detected"

        row: dict[str, Any] = {
            "Validation Metric": label
        }

        for category in CATEGORY_ORDER:
            category_mask = (
                classified_df[CATEGORY_COLUMN]
                .fillna("Unclassified")
                .eq(category)
            )

            metric_mask = (
                classified_df[detected_column]
                .fillna(False)
                .astype(bool)
            )

            row[category] = int(
                (category_mask & metric_mask).sum()
            )

        row["Total"] = sum(
            row[category]
            for category in CATEGORY_ORDER
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "Total",
        ascending=False,
    )


    return result


def create_percentage_matrix(
    counts_matrix: pd.DataFrame,
    classified_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert metric counts into within-category percentages.

    Example:
    RMSE Physics-based (%) =
    Physics-based papers using RMSE /
    total Physics-based papers.
    """
    category_totals = (
        classified_df[CATEGORY_COLUMN]
        .fillna("Unclassified")
        .value_counts()
        .to_dict()
    )

    rows: list[dict[str, Any]] = []

    for _, metric_row in counts_matrix.iterrows():
        row: dict[str, Any] = {
            "Validation Metric": metric_row["Validation Metric"]
        }

        for category in CATEGORY_ORDER:
            total_in_category = int(
                category_totals.get(category, 0)
            )

            metric_count = int(metric_row[category])

            if total_in_category > 0:
                percentage = (
                    100 * metric_count / total_in_category
                )
            else:
                percentage = 0

            row[category] = round(percentage, 3)

        total_documents = len(classified_df)
        total_metric_count = int(metric_row["Total"])

        row["Overall Percentage"] = round(
            100 * total_metric_count / total_documents,
            3,
        )

        rows.append(row)

    result = pd.DataFrame(rows)


    return result


def create_unmatched_publications(
    classified_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return publications for which no metric was detected.
    """
    unmatched_df = classified_df.loc[
        ~classified_df["Validation Metric Detected"]
        .fillna(False)
        .astype(bool)
    ].copy()

    preferred_columns = [
        YEAR_COLUMN,
        TITLE_COLUMN,
        ABSTRACT_COLUMN,
        CATEGORY_COLUMN,
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in unmatched_df.columns
    ]

    return unmatched_df[available_columns]


# ---------------------------------------------------------------------
# EXCEL OUTPUT
# ---------------------------------------------------------------------

def autosize_worksheet_columns(
    worksheet: Any,
    max_width: int = 60,
) -> None:
    """
    Set practical Excel column widths based on cell contents.
    """
    for column_cells in worksheet.columns:
        column_letter = column_cells[0].column_letter
        longest_value = 0

        for cell in column_cells:
            value = "" if cell.value is None else str(cell.value)
            longest_value = max(longest_value, len(value))

        worksheet.column_dimensions[column_letter].width = min(
            longest_value + 2,
            max_width,
        )


def save_analysis_workbook(
    overall_counts: pd.DataFrame,
    counts_matrix: pd.DataFrame,
    percentage_matrix: pd.DataFrame,
    unmatched_df: pd.DataFrame,
) -> None:
    """
    Save all summary outputs to one Excel workbook, one table per sheet.
    """
    with pd.ExcelWriter(
        ANALYSIS_OUTPUT,
        engine="openpyxl",
    ) as writer:
        overall_counts.to_excel(
            writer,
            sheet_name="Overall Counts",
            index=False,
        )

        counts_matrix.to_excel(
            writer,
            sheet_name="Counts Matrix",
            index=False,
        )

        percentage_matrix.to_excel(
            writer,
            sheet_name="Percentages",
            index=False,
        )

        unmatched_df.to_excel(
            writer,
            sheet_name="Unmatched Publications",
            index=False,
        )

        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            autosize_worksheet_columns(worksheet)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Input Excel file was not found:\n"
            f"{INPUT_FILE}\n\n"
            "Change INPUT_FILE at the top of the script if your file "
            "is located somewhere else."
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dictionary = load_dictionary()

    print(f"Reading input file:\n{INPUT_FILE}")
    print(f"Reading sheet:\n{INPUT_SHEET}\n")

    df = pd.read_excel(
        INPUT_FILE,
        sheet_name=INPUT_SHEET,
        engine="openpyxl",
    )

    validate_dataset(df)

    for optional_column in [
        AUTHOR_KEYWORDS_COLUMN,
        KEYWORDS_PLUS_COLUMN,
    ]:
        if optional_column not in df.columns:
            df[optional_column] = ""

    df[CATEGORY_COLUMN] = (
        df[CATEGORY_COLUMN]
        .fillna("Unclassified")
        .astype(str)
        .str.strip()
    )

    print(f"Publications loaded: {len(df):,}")

    df["Validation Analysis Text"] = df.apply(
        combine_text_fields,
        axis=1,
    )

    classification_results = df[
        "Validation Analysis Text"
    ].apply(
        lambda text: pd.Series(
            classify_publication(
                text=text,
                dictionary=dictionary,
            )
        )
    )

    classified_df = pd.concat(
        [
            df.reset_index(drop=True),
            classification_results.reset_index(drop=True),
        ],
        axis=1,
    )

    classified_df.to_excel(
        CLASSIFIED_OUTPUT,
        index=False,
        engine="openpyxl",
    )

    overall_counts = create_overall_counts(
        classified_df=classified_df,
        dictionary=dictionary,
    )

    counts_matrix = create_counts_matrix(
        classified_df=classified_df,
        dictionary=dictionary,
    )

    percentage_matrix = create_percentage_matrix(
        counts_matrix=counts_matrix,
        classified_df=classified_df,
    )

    unmatched_df = create_unmatched_publications(classified_df)

    save_analysis_workbook(
        overall_counts=overall_counts,
        counts_matrix=counts_matrix,
        percentage_matrix=percentage_matrix,
        unmatched_df=unmatched_df,
    )

    detected_count = int(
        classified_df[
            "Validation Metric Detected"
        ].sum()
    )

    print("\nValidation-metric classification completed.")

    print(
        "\nPublications with at least one metric detected: "
        f"{detected_count:,} "
        f"({100 * detected_count / len(classified_df):.2f}%)"
    )

    print(
        "Publications without a detected metric: "
        f"{len(classified_df) - detected_count:,}"
    )

    print("\nMetric distribution:")

    print(
        overall_counts[
            [
                "Validation Metric",
                "Document Count",
                "Percentage of All Publications",
            ]
        ].to_string(index=False)
    )

    print("\nSaved classified dataset:")
    print(CLASSIFIED_OUTPUT)

    print("\nSaved Excel analysis workbook:")
    print(ANALYSIS_OUTPUT)

    print(
        "Sheets: Overall Counts, Counts Matrix, "
        "Percentages, Unmatched Publications"
    )


if __name__ == "__main__":
    main()