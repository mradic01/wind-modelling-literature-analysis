from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "WoS_classified_filtrirano_s_graform_POLAZNI_PODATCI_OBRISANI_unclassified.xlsx"
)

DICTIONARY_FILE = PROJECT_ROOT / "config" / "measurement_sources.json"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"

CLASSIFIED_EXCEL_OUTPUT = PROCESSED_DIR / "WoS_measurement_sources_classified.xlsx"

CLASSIFIED_CSV_OUTPUT = PROCESSED_DIR / "WoS_measurement_sources_classified.csv"

CATEGORY_COUNTS_OUTPUT = TABLES_DIR / "measurement_source_counts.csv"

CATEGORY_BY_METHOD_OUTPUT = TABLES_DIR / "measurement_sources_by_modelling_category.csv"

UNSPECIFIED_OUTPUT = TABLES_DIR / "measurement_sources_unspecified.csv"

FIGURE_OUTPUT = FIGURES_DIR / "measurement_source_categories.jpg"


# ---------------------------------------------------------------------
# COLUMN NAMES
# ---------------------------------------------------------------------

TITLE_COLUMN = "Article Title"
ABSTRACT_COLUMN = "Abstract"
AUTHOR_KEYWORDS_COLUMN = "Author Keywords"
KEYWORDS_PLUS_COLUMN = "Keywords Plus"
YEAR_COLUMN = "Publication Year"
FINAL_CATEGORY_COLUMN = "Final Category"


# Categories that represent concrete measurement/data sources.
# Generic observational data is used only as a fallback.
PRECISE_CATEGORIES = [
    "ground_based_measurements",
    "ground_based_remote_sensing",
    "upper_air_measurements",
    "satellite_observations",
    "marine_measurements",
    "airborne_and_uav_measurements",
    "laboratory_and_wind_tunnel_measurements",
    "reanalysis_data",
    "numerical_model_data",
]

GENERIC_CATEGORY = "generic_observational_data"


# ---------------------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------------------


def normalize_text(value: Any) -> str:
    """
    Normalize text for reliable term matching.
    """
    if pd.isna(value):
        return ""

    text = str(value).casefold()

    # Normalize dash variants.
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("−", "-")

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def phrase_present(text: str, phrase: str) -> bool:
    """
    Search for a complete phrase while preventing partial matches.

    Example:
    'uav' should not be detected inside another longer word.
    """
    normalized_phrase = normalize_text(phrase)

    pattern = r"(?<![a-z0-9])" + re.escape(normalized_phrase) + r"(?![a-z0-9])"

    return re.search(pattern, text) is not None


def combine_text_fields(row: pd.Series) -> str:
    """
    Combine all available bibliographic text fields.
    """
    values = [
        row.get(TITLE_COLUMN, ""),
        row.get(ABSTRACT_COLUMN, ""),
        row.get(AUTHOR_KEYWORDS_COLUMN, ""),
        row.get(KEYWORDS_PLUS_COLUMN, ""),
    ]

    normalized_values = [
        normalize_text(value)
        for value in values
        if not pd.isna(value) and str(value).strip()
    ]

    return " ".join(normalized_values)


# ---------------------------------------------------------------------
# DICTIONARY
# ---------------------------------------------------------------------


def load_dictionary() -> dict[str, dict[str, Any]]:
    """
    Load the measurement source dictionary.
    """
    if not DICTIONARY_FILE.exists():
        raise FileNotFoundError(
            "Measurement source dictionary was not found:\n" f"{DICTIONARY_FILE}"
        )

    with DICTIONARY_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        dictionary = json.load(file)

    required_categories = set(PRECISE_CATEGORIES + [GENERIC_CATEGORY])
    missing_categories = required_categories.difference(dictionary)

    if missing_categories:
        raise ValueError(
            "The dictionary is missing these categories:\n"
            f"{sorted(missing_categories)}"
        )

    for category_key, category_data in dictionary.items():
        if "label" not in category_data:
            raise ValueError(f"Category '{category_key}' has no 'label'.")

        if "terms" not in category_data:
            raise ValueError(f"Category '{category_key}' has no 'terms' list.")

    return dictionary


# ---------------------------------------------------------------------
# TERM MATCHING
# ---------------------------------------------------------------------


def find_matches(
    text: str,
    terms: list[str],
) -> list[str]:
    """
    Return all dictionary terms found in the publication text.
    """
    matches = [term for term in terms if phrase_present(text, term)]

    return sorted(set(matches))


def classify_measurement_sources(
    text: str,
    dictionary: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Perform multi-label source classification.

    A publication may belong to several precise categories.

    Generic observational data is assigned only when no precise source
    category is found.
    """
    precise_matches: dict[str, list[str]] = {}

    for category_key in PRECISE_CATEGORIES:
        terms = dictionary[category_key]["terms"]
        matches = find_matches(text, terms)

        if matches:
            precise_matches[category_key] = matches

    generic_matches = find_matches(
        text,
        dictionary[GENERIC_CATEGORY]["terms"],
    )

    if precise_matches:
        selected_category_keys = list(precise_matches.keys())
        fallback_generic = False

    elif generic_matches:
        selected_category_keys = [GENERIC_CATEGORY]
        fallback_generic = True

    else:
        selected_category_keys = []
        fallback_generic = False

    selected_labels = [
        dictionary[category_key]["label"] for category_key in selected_category_keys
    ]

    all_matched_terms: list[str] = []

    for matches in precise_matches.values():
        all_matched_terms.extend(matches)

    if fallback_generic:
        all_matched_terms.extend(generic_matches)

    result: dict[str, Any] = {
        "Measurement Source Identified": bool(selected_category_keys),
        "Measurement Source Count": len(selected_category_keys),
        "Measurement Source Categories": "; ".join(selected_labels),
        "Measurement Source Keys": "; ".join(selected_category_keys),
        "Measurement Matched Terms": "; ".join(sorted(set(all_matched_terms))),
        "Generic Observational Fallback": fallback_generic,
        "No Measurement Source Identified": not bool(selected_category_keys),
    }

    # Add one Boolean and one term column for each category.
    for category_key, category_data in dictionary.items():
        label = category_data["label"]

        if category_key in precise_matches:
            detected = True
            matches = precise_matches[category_key]

        elif category_key == GENERIC_CATEGORY and fallback_generic:
            detected = True
            matches = generic_matches

        else:
            detected = False
            matches = []

        result[f"{label} Detected"] = detected
        result[f"{label} Terms"] = "; ".join(matches)

    return result


# ---------------------------------------------------------------------
# DATA VALIDATION
# ---------------------------------------------------------------------


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Verify essential input columns.
    """
    required_columns = {
        TITLE_COLUMN,
        ABSTRACT_COLUMN,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "The input dataset is missing these columns:\n" f"{sorted(missing_columns)}"
        )


# ---------------------------------------------------------------------
# SUMMARY TABLES
# ---------------------------------------------------------------------


def create_category_counts(
    classified_df: pd.DataFrame,
    dictionary: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """
    Count the number of publications assigned to every source category.

    Because classification is multi-label, totals can exceed the number
    of publications.
    """
    rows: list[dict[str, Any]] = []

    total_documents = len(classified_df)

    ordered_keys = PRECISE_CATEGORIES + [GENERIC_CATEGORY]

    for category_key in ordered_keys:
        label = dictionary[category_key]["label"]
        detected_column = f"{label} Detected"

        count = int(classified_df[detected_column].fillna(False).astype(bool).sum())

        rows.append(
            {
                "Category Key": category_key,
                "Measurement Source Category": label,
                "Document Count": count,
                "Percentage of Documents": round(
                    100 * count / total_documents,
                    3,
                ),
            }
        )

    counts_df = pd.DataFrame(rows)

    counts_df = counts_df.sort_values(
        "Document Count",
        ascending=False,
    ).reset_index(drop=True)

    counts_df.to_csv(
        CATEGORY_COUNTS_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    return counts_df


def create_category_by_modelling_method(
    classified_df: pd.DataFrame,
    dictionary: dict[str, dict[str, Any]],
) -> None:
    """
    Cross-tabulate measurement sources and modelling categories.
    """
    if FINAL_CATEGORY_COLUMN not in classified_df.columns:
        print(
            "\nFinal Category column was not found. " "Skipping source-by-method table."
        )
        return

    rows: list[dict[str, Any]] = []

    modelling_categories = sorted(
        classified_df[FINAL_CATEGORY_COLUMN].dropna().astype(str).unique()
    )

    ordered_keys = PRECISE_CATEGORIES + [GENERIC_CATEGORY]

    for category_key in ordered_keys:
        label = dictionary[category_key]["label"]
        detected_column = f"{label} Detected"

        row: dict[str, Any] = {"Measurement Source Category": label}

        for modelling_category in modelling_categories:
            mask = classified_df[detected_column].fillna(False).astype(bool) & (
                classified_df[FINAL_CATEGORY_COLUMN] == modelling_category
            )

            row[modelling_category] = int(mask.sum())

        row["Total"] = int(
            classified_df[detected_column].fillna(False).astype(bool).sum()
        )

        rows.append(row)

    result_df = pd.DataFrame(rows)

    result_df = result_df.sort_values(
        "Total",
        ascending=False,
    )

    result_df.to_csv(
        CATEGORY_BY_METHOD_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )


def save_unspecified_publications(
    classified_df: pd.DataFrame,
) -> None:
    """
    Save publications for which no source was identified.
    """
    unspecified_df = classified_df.loc[
        classified_df["No Measurement Source Identified"].fillna(False).astype(bool)
    ].copy()

    preferred_columns = [
        TITLE_COLUMN,
        ABSTRACT_COLUMN,
        YEAR_COLUMN,
        FINAL_CATEGORY_COLUMN,
        "Measurement Source Categories",
        "Measurement Matched Terms",
    ]

    available_columns = [
        column for column in preferred_columns if column in unspecified_df.columns
    ]

    unspecified_df[available_columns].to_csv(
        UNSPECIFIED_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )


# ---------------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------------


def create_figure(
    counts_df: pd.DataFrame,
) -> None:
    """
    Create a horizontal bar chart of measurement-source categories.
    """
    plot_df = counts_df.copy()

    # Generic fallback is useful for QA, but it is usually better not to
    # show it as a principal scientific category in the final graph.
    plot_df = plot_df.loc[plot_df["Category Key"] != GENERIC_CATEGORY].copy()

    plot_df = plot_df.sort_values(
        "Document Count",
        ascending=True,
    )

    plt.figure(figsize=(11, 7))

    bars = plt.barh(
        plot_df["Measurement Source Category"],
        plot_df["Document Count"],
    )

    plt.xlabel("Number of publications")
    plt.ylabel("Meteorological data source category")
    plt.title("Meteorological data sources used in wind modelling studies")

    # Add values at the end of bars.
    for bar in bars:
        width = bar.get_width()

        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f" {int(width)}",
            va="center",
        )

    plt.tight_layout()

    plt.savefig(
        FIGURE_OUTPUT,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Input Excel file was not found:\n"
            f"{INPUT_FILE}\n\n"
            "Change INPUT_FILE at the top of the script if your file "
            "has a different name or location."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dictionary = load_dictionary()

    print(f"Reading input file:\n{INPUT_FILE}\n")

    df = pd.read_excel(
        INPUT_FILE,
        engine="openpyxl",
    )

    validate_dataset(df)

    print(f"Publications loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns):,}")

    # Ensure optional text fields exist.
    for column in [
        AUTHOR_KEYWORDS_COLUMN,
        KEYWORDS_PLUS_COLUMN,
    ]:
        if column not in df.columns:
            df[column] = ""

    df["Measurement Analysis Text"] = df.apply(
        combine_text_fields,
        axis=1,
    )

    classification_results = df["Measurement Analysis Text"].apply(
        lambda text: pd.Series(
            classify_measurement_sources(
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
        CLASSIFIED_EXCEL_OUTPUT,
        index=False,
        engine="openpyxl",
    )

    classified_df.to_csv(
        CLASSIFIED_CSV_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    counts_df = create_category_counts(
        classified_df=classified_df,
        dictionary=dictionary,
    )

    create_category_by_modelling_method(
        classified_df=classified_df,
        dictionary=dictionary,
    )

    save_unspecified_publications(classified_df)

    create_figure(counts_df)

    identified_count = int(classified_df["Measurement Source Identified"].sum())

    unspecified_count = int(classified_df["No Measurement Source Identified"].sum())

    multi_source_count = int((classified_df["Measurement Source Count"] > 1).sum())

    generic_fallback_count = int(classified_df["Generic Observational Fallback"].sum())

    print("\nMeasurement source classification completed.")

    print(
        f"\nPublications with at least one source identified: "
        f"{identified_count:,} "
        f"({100 * identified_count / len(classified_df):.1f}%)"
    )

    print(f"Publications with multiple source categories: " f"{multi_source_count:,}")

    print(
        f"Publications assigned only through generic observational "
        f"terms: {generic_fallback_count:,}"
    )

    print(
        f"Publications with no source identified: "
        f"{unspecified_count:,} "
        f"({100 * unspecified_count / len(classified_df):.1f}%)"
    )

    print("\nCategory distribution:")
    print(
        counts_df[
            [
                "Measurement Source Category",
                "Document Count",
                "Percentage of Documents",
            ]
        ].to_string(index=False)
    )

    print("\nSaved classified Excel:")
    print(CLASSIFIED_EXCEL_OUTPUT)

    print("\nSaved summary table:")
    print(CATEGORY_COUNTS_OUTPUT)

    print("\nSaved source-by-method table:")
    print(CATEGORY_BY_METHOD_OUTPUT)

    print("\nSaved unspecified publications:")
    print(UNSPECIFIED_OUTPUT)

    print("\nSaved figure:")
    print(FIGURE_OUTPUT)


if __name__ == "__main__":
    main()
