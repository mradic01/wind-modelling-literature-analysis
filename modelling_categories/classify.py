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

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "WoS.xlsx"
CONFIG_FILE = PROJECT_ROOT / "config" / "categories.json"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "output" / "tables"

CLASSIFIED_OUTPUT = PROCESSED_DIR / "WoS_classified.xlsx"
CLASSIFIED_CSV_OUTPUT = PROCESSED_DIR / "WoS_classified.csv"

YEARLY_COUNTS_OUTPUT = OUTPUT_TABLES_DIR / "category_counts_by_year.csv"

YEARLY_SHARES_OUTPUT = OUTPUT_TABLES_DIR / "category_shares_by_year.csv"

DL_COUNTS_OUTPUT = OUTPUT_TABLES_DIR / "deep_learning_by_year.csv"

CLASSIFICATION_SUMMARY_OUTPUT = OUTPUT_TABLES_DIR / "classification_summary.csv"


# ---------------------------------------------------------------------
# INPUT COLUMNS
# ---------------------------------------------------------------------

TITLE_COLUMN = "Article Title"
ABSTRACT_COLUMN = "Abstract"
AUTHOR_KEYWORDS_COLUMN = "Author Keywords"
KEYWORDS_PLUS_COLUMN = "Keywords Plus"
YEAR_COLUMN = "Publication Year"
DOCUMENT_TYPE_COLUMN = "Document Type"
DOI_COLUMN = "DOI"
WOS_ID_COLUMN = "UT (Unique WOS ID)"


# ---------------------------------------------------------------------
# TEXT PROCESSING
# ---------------------------------------------------------------------


def normalize_text(value: Any) -> str:
    """
    Convert a value into normalized lowercase text suitable for matching.
    """
    if pd.isna(value):
        return ""

    text = str(value).casefold()

    # Normalize different dash characters.
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("−", "-")

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def phrase_present(text: str, phrase: str) -> bool:
    """
    Search for a complete term rather than a substring.

    This prevents abbreviations such as 'gru' or 'svm' from being found
    inside unrelated longer words.
    """
    normalized_phrase = normalize_text(phrase)

    pattern = r"(?<![a-z0-9])" + re.escape(normalized_phrase) + r"(?![a-z0-9])"

    return re.search(pattern, text) is not None


def join_nonempty(values: list[Any]) -> str:
    """
    Join non-empty text fields into a single string.
    """
    cleaned_values = [
        normalize_text(value)
        for value in values
        if not pd.isna(value) and str(value).strip()
    ]

    return " ".join(cleaned_values)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


def load_categories() -> dict[str, list[str]]:
    """
    Load classification terms from config/categories.json.
    """
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Configuration file was not found:\n{CONFIG_FILE}")

    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        categories = json.load(file)

    required_categories = {
        "physics_based",
        "statistical",
        "machine_learning",
        "deep_learning",
        "explicit_hybrid",
    }

    missing_categories = required_categories.difference(categories)

    if missing_categories:
        raise ValueError(
            "The configuration is missing these categories: "
            f"{sorted(missing_categories)}"
        )

    return categories


# ---------------------------------------------------------------------
# MATCHING
# ---------------------------------------------------------------------


def find_matches(
    text: str,
    terms: list[str],
) -> list[str]:
    """
    Return all category terms found in a text.
    """
    matches = [term for term in terms if phrase_present(text, term)]

    return sorted(set(matches))


def calculate_field_score(
    title: str,
    abstract: str,
    author_keywords: str,
    keywords_plus: str,
    terms: list[str],
) -> tuple[int, list[str]]:
    """
    Calculate a category score.

    Terms found in the title receive more weight than terms found only
    in the abstract.

    Title:            weight 3
    Author keywords:  weight 2
    Keywords Plus:    weight 2
    Abstract:         weight 1
    """
    title_matches = find_matches(title, terms)
    author_keyword_matches = find_matches(author_keywords, terms)
    keywords_plus_matches = find_matches(keywords_plus, terms)
    abstract_matches = find_matches(abstract, terms)

    score = (
        3 * len(title_matches)
        + 2 * len(author_keyword_matches)
        + 2 * len(keywords_plus_matches)
        + len(abstract_matches)
    )

    all_matches = sorted(
        set(
            title_matches
            + author_keyword_matches
            + keywords_plus_matches
            + abstract_matches
        )
    )

    return score, all_matches


# ---------------------------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------------------------


def classify_record(
    row: pd.Series,
    categories: dict[str, list[str]],
) -> dict[str, Any]:
    """
    Classify one publication into a principal modelling category.

    Deep learning is treated as part of machine learning but is also
    recorded separately through a Boolean flag.

    Hybrid is assigned when:
    1. an explicit hybrid term is found, or
    2. physics-based and ML/DL evidence are both present.

    Statistical and ML terms occurring together are not automatically
    treated as Hybrid unless the publication explicitly describes the
    approach as hybrid. This reduces false positives in comparative
    studies and review papers.
    """
    title = normalize_text(row.get(TITLE_COLUMN, ""))
    abstract = normalize_text(row.get(ABSTRACT_COLUMN, ""))
    author_keywords = normalize_text(row.get(AUTHOR_KEYWORDS_COLUMN, ""))
    keywords_plus = normalize_text(row.get(KEYWORDS_PLUS_COLUMN, ""))

    physics_score, physics_terms = calculate_field_score(
        title=title,
        abstract=abstract,
        author_keywords=author_keywords,
        keywords_plus=keywords_plus,
        terms=categories["physics_based"],
    )

    statistical_score, statistical_terms = calculate_field_score(
        title=title,
        abstract=abstract,
        author_keywords=author_keywords,
        keywords_plus=keywords_plus,
        terms=categories["statistical"],
    )

    ml_score, ml_terms = calculate_field_score(
        title=title,
        abstract=abstract,
        author_keywords=author_keywords,
        keywords_plus=keywords_plus,
        terms=categories["machine_learning"],
    )

    dl_score, dl_terms = calculate_field_score(
        title=title,
        abstract=abstract,
        author_keywords=author_keywords,
        keywords_plus=keywords_plus,
        terms=categories["deep_learning"],
    )

    hybrid_score, hybrid_terms = calculate_field_score(
        title=title,
        abstract=abstract,
        author_keywords=author_keywords,
        keywords_plus=keywords_plus,
        terms=categories["explicit_hybrid"],
    )

    # Deep learning belongs to the broader ML paradigm.
    combined_ml_score = ml_score + dl_score
    combined_ml_terms = sorted(set(ml_terms + dl_terms))

    physics_detected = physics_score > 0
    statistical_detected = statistical_score > 0
    ml_detected = combined_ml_score > 0
    dl_detected = dl_score > 0
    explicit_hybrid_detected = hybrid_score > 0

    # A physics + ML combination is treated as hybrid even where the
    # abstract does not explicitly use the word "hybrid".
    physics_ml_combination = physics_detected and ml_detected

    if explicit_hybrid_detected or physics_ml_combination:
        final_category = "Hybrid"

        matched_terms = sorted(set(hybrid_terms + physics_terms + combined_ml_terms))

        if hybrid_score >= 3 or (physics_score >= 3 and combined_ml_score >= 3):
            confidence = "High"
        else:
            confidence = "Medium"

        if explicit_hybrid_detected and physics_ml_combination:
            reason = (
                "Explicit hybrid terminology and simultaneous "
                "physics-based and machine-learning evidence"
            )
        elif explicit_hybrid_detected:
            reason = "Explicit hybrid terminology"
        else:
            reason = "Simultaneous physics-based and " "machine-learning evidence"

    else:
        paradigm_scores = {
            "Physics-based": physics_score,
            "Statistical": statistical_score,
            "Machine Learning": combined_ml_score,
        }

        ranked_scores = sorted(
            paradigm_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        highest_category, highest_score = ranked_scores[0]
        second_score = ranked_scores[1][1]

        if highest_score == 0:
            final_category = "Unclassified"
            confidence = "Low"
            matched_terms = []
            reason = "No configured modelling terms were detected"

        else:
            final_category = highest_category

            category_term_map = {
                "Physics-based": physics_terms,
                "Statistical": statistical_terms,
                "Machine Learning": combined_ml_terms,
            }

            matched_terms = category_term_map[final_category]

            difference = highest_score - second_score

            if highest_score >= 6 and difference >= 3:
                confidence = "High"
            elif difference >= 1:
                confidence = "Medium"
            else:
                confidence = "Low"

            reason = f"Highest weighted score: {final_category} " f"({highest_score})"

    all_detected_categories: list[str] = []

    if physics_detected:
        all_detected_categories.append("Physics-based")

    if statistical_detected:
        all_detected_categories.append("Statistical")

    if ml_detected:
        all_detected_categories.append("Machine Learning")

    if dl_detected:
        all_detected_categories.append("Deep Learning")

    if explicit_hybrid_detected:
        all_detected_categories.append("Explicit Hybrid")

    return {
        "Final Category": final_category,
        "Classification Confidence": confidence,
        "Classification Reason": reason,
        "Detected Categories": "; ".join(all_detected_categories),
        "Matched Terms": "; ".join(matched_terms),
        "Physics-based Detected": physics_detected,
        "Statistical Detected": statistical_detected,
        "Machine Learning Detected": ml_detected,
        "Deep Learning Detected": dl_detected,
        "Explicit Hybrid Detected": explicit_hybrid_detected,
        "Physics-based Terms": "; ".join(physics_terms),
        "Statistical Terms": "; ".join(statistical_terms),
        "Machine Learning Terms": "; ".join(ml_terms),
        "Deep Learning Terms": "; ".join(dl_terms),
        "Hybrid Terms": "; ".join(hybrid_terms),
        "Physics-based Score": physics_score,
        "Statistical Score": statistical_score,
        "Machine Learning Score": combined_ml_score,
        "Deep Learning Score": dl_score,
        "Explicit Hybrid Score": hybrid_score,
    }


# ---------------------------------------------------------------------
# DATASET VALIDATION AND CLEANING
# ---------------------------------------------------------------------


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Verify that all essential WoS columns are present.
    """
    required_columns = {
        TITLE_COLUMN,
        ABSTRACT_COLUMN,
        YEAR_COLUMN,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "The input file is missing these required columns: "
            f"{sorted(missing_columns)}"
        )


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate WoS records, preferably using the WoS unique ID.

    The original row count is reported to the terminal.
    """
    original_count = len(df)

    if WOS_ID_COLUMN in df.columns:
        has_wos_id = df[WOS_ID_COLUMN].notna()

        with_id = df.loc[has_wos_id].drop_duplicates(
            subset=[WOS_ID_COLUMN],
            keep="first",
        )

        without_id = df.loc[~has_wos_id]

        df = pd.concat(
            [with_id, without_id],
            ignore_index=True,
        )

    elif DOI_COLUMN in df.columns:
        has_doi = df[DOI_COLUMN].notna()

        with_doi = df.loc[has_doi].drop_duplicates(
            subset=[DOI_COLUMN],
            keep="first",
        )

        without_doi = df.loc[~has_doi]

        df = pd.concat(
            [with_doi, without_doi],
            ignore_index=True,
        )

    removed_count = original_count - len(df)

    print(f"Duplicate records removed: {removed_count:,}")

    return df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean publication years and create a combined analysis text column.
    """
    df = df.copy()

    df[YEAR_COLUMN] = pd.to_numeric(
        df[YEAR_COLUMN],
        errors="coerce",
    ).astype("Int64")

    text_columns = [
        TITLE_COLUMN,
        ABSTRACT_COLUMN,
        AUTHOR_KEYWORDS_COLUMN,
        KEYWORDS_PLUS_COLUMN,
    ]

    for column in text_columns:
        if column not in df.columns:
            df[column] = ""

        df[column] = df[column].fillna("")

    df["Analysis Text"] = df.apply(
        lambda row: join_nonempty(
            [
                row[TITLE_COLUMN],
                row[ABSTRACT_COLUMN],
                row[AUTHOR_KEYWORDS_COLUMN],
                row[KEYWORDS_PLUS_COLUMN],
            ]
        ),
        axis=1,
    )

    return df


# ---------------------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------------------


def create_yearly_statistics(
    classified_df: pd.DataFrame,
) -> None:
    """
    Create publication counts and category shares by year.
    """
    valid_df = classified_df.dropna(subset=[YEAR_COLUMN]).copy()

    valid_df[YEAR_COLUMN] = valid_df[YEAR_COLUMN].astype(int)

    category_order = [
        "Physics-based",
        "Statistical",
        "Machine Learning",
        "Hybrid",
        "Unclassified",
    ]

    yearly_counts = (
        valid_df.groupby([YEAR_COLUMN, "Final Category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=category_order, fill_value=0)
        .sort_index()
    )

    yearly_counts["Total Publications"] = yearly_counts.sum(axis=1)

    yearly_counts.to_csv(
        YEARLY_COUNTS_OUTPUT,
        encoding="utf-8-sig",
    )

    yearly_shares = (
        yearly_counts[category_order].div(
            yearly_counts["Total Publications"],
            axis=0,
        )
        * 100
    )

    yearly_shares = yearly_shares.round(3)

    yearly_shares.to_csv(
        YEARLY_SHARES_OUTPUT,
        encoding="utf-8-sig",
    )

    dl_by_year = valid_df.groupby(YEAR_COLUMN)["Deep Learning Detected"].agg(
        Deep_Learning_Publications="sum",
        Total_Publications="count",
    )

    dl_by_year["Deep Learning Share (%)"] = (
        100
        * dl_by_year["Deep_Learning_Publications"]
        / dl_by_year["Total_Publications"]
    ).round(3)

    dl_by_year.to_csv(
        DL_COUNTS_OUTPUT,
        encoding="utf-8-sig",
    )

    classification_summary = (
        classified_df["Final Category"]
        .value_counts(dropna=False)
        .rename_axis("Category")
        .reset_index(name="Document Count")
    )

    classification_summary["Percentage"] = (
        100 * classification_summary["Document Count"] / len(classified_df)
    ).round(3)

    classification_summary.to_csv(
        CLASSIFICATION_SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"The input Excel file was not found:\n{INPUT_FILE}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading:\n{INPUT_FILE}\n")

    df = pd.read_excel(
        INPUT_FILE,
        engine="openpyxl",
    )

    validate_dataset(df)

    print(f"Records loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns):,}")

    df = remove_duplicates(df)
    df = prepare_dataset(df)

    categories = load_categories()

    classification_results = df.apply(
        lambda row: pd.Series(classify_record(row, categories)),
        axis=1,
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

    classified_df.to_csv(
        CLASSIFIED_CSV_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    create_yearly_statistics(classified_df)

    print("\nClassification completed.\n")

    print("Final category distribution:")
    print(classified_df["Final Category"].value_counts().to_string())

    print("\nClassification confidence:")
    print(classified_df["Classification Confidence"].value_counts().to_string())

    print(
        "\nDeep learning publications detected: "
        f"{classified_df['Deep Learning Detected'].sum():,}"
    )

    print("\nSaved classified Excel:")
    print(CLASSIFIED_OUTPUT)

    print("\nSaved classified CSV:")
    print(CLASSIFIED_CSV_OUTPUT)

    print("\nSaved yearly counts:")
    print(YEARLY_COUNTS_OUTPUT)

    print("\nSaved yearly category shares:")
    print(YEARLY_SHARES_OUTPUT)

    print("\nSaved deep-learning statistics:")
    print(DL_COUNTS_OUTPUT)


if __name__ == "__main__":
    main()
