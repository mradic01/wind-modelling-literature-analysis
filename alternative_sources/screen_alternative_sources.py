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
    PROJECT_ROOT / "data" / "processed" / "WoS_measurement_sources_classified.xlsx"
)

DICTIONARY_FILE = PROJECT_ROOT / "config" / "alternative_source_screening.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "WoS_alternative_source_candidates.xlsx"


# ---------------------------------------------------------------------
# COLUMN NAMES
# ---------------------------------------------------------------------

TITLE_COLUMN = "Article Title"
ABSTRACT_COLUMN = "Abstract"
AUTHOR_KEYWORDS_COLUMN = "Author Keywords"
KEYWORDS_PLUS_COLUMN = "Keywords Plus"
YEAR_COLUMN = "Publication Year"
DOI_COLUMN = "DOI"
FINAL_CATEGORY_COLUMN = "Final Category"


# ---------------------------------------------------------------------
# TEXT PROCESSING
# ---------------------------------------------------------------------


def normalize_text(value: Any) -> str:
    """
    Normalize publication text before matching.
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
    Match a complete term or phrase.

    This prevents short terms such as PWS, IoT and WSN from matching
    inside unrelated longer words.
    """
    normalized_phrase = normalize_text(phrase)

    pattern = r"(?<![a-z0-9])" + re.escape(normalized_phrase) + r"(?![a-z0-9])"

    return re.search(pattern, text) is not None


def combine_text_fields(row: pd.Series) -> str:
    """
    Combine the bibliographic text fields used for screening.
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


def find_matches(
    text: str,
    terms: list[str],
) -> list[str]:
    """
    Return all dictionary terms found in a publication.
    """
    return sorted({term for term in terms if phrase_present(text, term)})


# ---------------------------------------------------------------------
# INPUT LOADING
# ---------------------------------------------------------------------


def load_dictionary() -> list[str]:
    """
    Load the alternative-source screening terms.
    """
    if not DICTIONARY_FILE.exists():
        raise FileNotFoundError("Dictionary file was not found:\n" f"{DICTIONARY_FILE}")

    try:
        with DICTIONARY_FILE.open(
            "r",
            encoding="utf-8-sig",
        ) as file:
            dictionary = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            "The dictionary is not valid JSON.\n"
            f"File: {DICTIONARY_FILE}\n"
            f"Line: {error.lineno}\n"
            f"Column: {error.colno}\n"
            f"Error: {error.msg}"
        ) from error

    required_key = "alternative_source_terms"

    if required_key not in dictionary:
        raise ValueError(f"The dictionary must contain the key '{required_key}'.")

    terms = dictionary[required_key]

    if not isinstance(terms, list):
        raise ValueError(f"'{required_key}' must contain a JSON list.")

    cleaned_terms = sorted({str(term).strip() for term in terms if str(term).strip()})

    if not cleaned_terms:
        raise ValueError("The alternative-source dictionary contains no terms.")

    return cleaned_terms


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Verify essential input columns.
    """
    required_columns = {
        TITLE_COLUMN,
        ABSTRACT_COLUMN,
        YEAR_COLUMN,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "The input dataset is missing these columns:\n" f"{sorted(missing_columns)}"
        )


# ---------------------------------------------------------------------
# SCREENING
# ---------------------------------------------------------------------


def screen_publication(
    text: str,
    terms: list[str],
) -> dict[str, Any]:
    """
    Screen one publication for possible alternative data-source terms.
    """
    matched_terms = find_matches(
        text=text,
        terms=terms,
    )

    return {
        "Selected for Manual Review": bool(matched_terms),
        "Matched Terms": "; ".join(matched_terms),
        "Matched Term Count": len(matched_terms),
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Input Excel file was not found:\n"
            f"{INPUT_FILE}\n\n"
            "Change INPUT_FILE at the top of the script if necessary."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    terms = load_dictionary()

    print(f"Reading input file:\n{INPUT_FILE}\n")

    df = pd.read_excel(
        INPUT_FILE,
        engine="openpyxl",
    )

    validate_dataset(df)

    for optional_column in [
        AUTHOR_KEYWORDS_COLUMN,
        KEYWORDS_PLUS_COLUMN,
    ]:
        if optional_column not in df.columns:
            df[optional_column] = ""

    print(f"Publications loaded: {len(df):,}")
    print(f"Dictionary terms loaded: {len(terms):,}")

    df["_alternative_screening_text"] = df.apply(
        combine_text_fields,
        axis=1,
    )

    screening_results = df["_alternative_screening_text"].apply(
        lambda text: pd.Series(
            screen_publication(
                text=text,
                terms=terms,
            )
        )
    )

    screened_df = pd.concat(
        [
            df.reset_index(drop=True),
            screening_results.reset_index(drop=True),
        ],
        axis=1,
    )

    review_df = screened_df.loc[
        screened_df["Selected for Manual Review"].fillna(False).astype(bool)
    ].copy()

    review_df["Final Decision"] = ""
    review_df["Manual Review Notes"] = ""

    preferred_columns = [
        "Final Decision",
        "Manual Review Notes",
        "Matched Terms",
        "Matched Term Count",
        YEAR_COLUMN,
        TITLE_COLUMN,
        ABSTRACT_COLUMN,
        AUTHOR_KEYWORDS_COLUMN,
        KEYWORDS_PLUS_COLUMN,
        DOI_COLUMN,
        FINAL_CATEGORY_COLUMN,
        "Measurement Source Categories",
        "Measurement Matched Terms",
    ]

    available_columns = [
        column for column in preferred_columns if column in review_df.columns
    ]

    review_df = review_df[available_columns]

    sort_columns: list[str] = [
        "Matched Term Count",
    ]

    ascending: list[bool] = [
        False,
    ]

    if YEAR_COLUMN in review_df.columns:
        sort_columns.append(YEAR_COLUMN)
        ascending.append(False)

    review_df = review_df.sort_values(
        by=sort_columns,
        ascending=ascending,
    )

    review_df.to_excel(
        OUTPUT_FILE,
        index=False,
        engine="openpyxl",
    )

    print("\nScreening completed.")

    print("Publications selected for manual review: " f"{len(review_df):,}")

    if not review_df.empty:
        preview_columns = [
            column
            for column in [
                YEAR_COLUMN,
                TITLE_COLUMN,
                "Matched Terms",
            ]
            if column in review_df.columns
        ]

        print("\nSelected publications:")
        print(review_df[preview_columns].to_string(index=False))

    print("\nSaved review workbook:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
