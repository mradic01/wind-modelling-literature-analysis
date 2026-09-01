from __future__ import annotations

import gzip
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "station_density"
OUTPUT_DIR = PROJECT_ROOT / "output" / "station_density"
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"

METEOSTAT_CACHE = RAW_DIR / "meteostat_active_stations_lite.json.gz"
AREA_CACHE = RAW_DIR / "world_bank_country_areas.json"

CSV_OUTPUT = OUTPUT_DIR / "station_density.csv"
EXCEL_OUTPUT = OUTPUT_DIR / "station_density.xlsx"
LATEX_OUTPUT = OUTPUT_DIR / "station_density_table.tex"
METADATA_OUTPUT = OUTPUT_DIR / "station_density_metadata.txt"
FIGURE_OUTPUT = FIGURES_DIR / "station_density.png"


# =============================================================================
# SOURCES
# =============================================================================

METEOSTAT_STATIONS_URL = (
    "https://bulk.meteostat.net/v2/stations/lite.json.gz"
)

WORLD_BANK_AREA_URL = (
    "https://api.worldbank.org/v2/country/"
    "{iso3}/indicator/AG.LND.TOTL.K2"
    "?format=json&per_page=100"
)


# =============================================================================
# COUNTRIES
# =============================================================================

# alpha2 is used by Meteostat.
# alpha3 is used by the World Bank API.
COUNTRIES = [
    {
        "country": "Austria",
        "alpha2": "AT",
        "alpha3": "AUT",
        "service": "GeoSphere Austria",
    },
    {
        "country": "Australia",
        "alpha2": "AU",
        "alpha3": "AUS",
        "service": "Bureau of Meteorology",
    },
    {
        "country": "China",
        "alpha2": "CN",
        "alpha3": "CHN",
        "service": "China Meteorological Administration",
    },
    {
        "country": "Croatia",
        "alpha2": "HR",
        "alpha3": "HRV",
        "service": "DHMZ",
    },
    {
        "country": "Denmark",
        "alpha2": "DK",
        "alpha3": "DNK",
        "service": "DMI",
    },
    {
        "country": "France",
        "alpha2": "FR",
        "alpha3": "FRA",
        "service": "Météo-France",
    },
    {
        "country": "Germany",
        "alpha2": "DE",
        "alpha3": "DEU",
        "service": "DWD",
    },
    {
        "country": "Italy",
        "alpha2": "IT",
        "alpha3": "ITA",
        "service": "Servizio Meteorologico Italiano",
    },
    {
        "country": "Japan",
        "alpha2": "JP",
        "alpha3": "JPN",
        "service": "Japan Meteorological Agency",
    },
    {
        "country": "Netherlands",
        "alpha2": "NL",
        "alpha3": "NLD",
        "service": "KNMI",
    },
    {
        "country": "Norway",
        "alpha2": "NO",
        "alpha3": "NOR",
        "service": "MET Norway",
    },
    {
        "country": "Spain",
        "alpha2": "ES",
        "alpha3": "ESP",
        "service": "AEMET",
    },
    {
        "country": "Switzerland",
        "alpha2": "CH",
        "alpha3": "CHE",
        "service": "MeteoSwiss",
    },
    {
        "country": "United Kingdom",
        "alpha2": "GB",
        "alpha3": "GBR",
        "service": "Met Office",
    },
    {
        "country": "United States",
        "alpha2": "US",
        "alpha3": "USA",
        "service": "NOAA",
    },
        {
        "country": "Poland",
        "alpha2": "PL",
        "alpha3": "POL",
        "service": "IMGW-PIB",
    },
    {
        "country": "Sweden",
        "alpha2": "SE",
        "alpha3": "SWE",
        "service": "SMHI",
    },
    {
        "country": "Finland",
        "alpha2": "FI",
        "alpha3": "FIN",
        "service": "Finnish Meteorological Institute",
    },
    {
        "country": "Belgium",
        "alpha2": "BE",
        "alpha3": "BEL",
        "service": "Royal Meteorological Institute of Belgium",
    },
    {
        "country": "Portugal",
        "alpha2": "PT",
        "alpha3": "PRT",
        "service": "IPMA",
    },
]


# =============================================================================
# GENERAL SETTINGS
# =============================================================================

DOWNLOAD_TIMEOUT_SECONDS = 90
DOWNLOAD_RETRIES = 3

# Ako je True, ponovno preuzima podatke iako cache postoji.
FORCE_DOWNLOAD = False

# Prikaz na grafu.
FIGURE_COUNT_COLUMN = "Active Meteostat Stations per 1,000 km²"

# Mean Earth radius recommended by the International Union of Geodesy and
# Geophysics. BallTree returns angular Haversine distances in radians.
EARTH_RADIUS_KM = 6371.0088


# =============================================================================
# DOWNLOAD UTILITIES
# =============================================================================

def download_file(
    url: str,
    destination: Path,
    force: bool = False,
) -> None:
    """
    Download a file with retries and save it locally.

    Existing cached files are reused unless force=True.
    """
    if destination.exists() and not force:
        print(f"Using cached file: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Wind-Modelling-MetaAnalysis/1.0 "
                "(academic research)"
            )
        },
    )

    last_error: Exception | None = None

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            print(
                f"Downloading {url}\n"
                f"Attempt {attempt}/{DOWNLOAD_RETRIES}"
            )

            with urllib.request.urlopen(
                request,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            ) as response:
                content = response.read()

            if not content:
                raise RuntimeError(
                    f"Downloaded file is empty: {url}"
                )

            destination.write_bytes(content)

            print(
                f"Saved {len(content):,} bytes to:\n"
                f"{destination}"
            )
            return

        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            RuntimeError,
        ) as error:
            last_error = error

            if attempt < DOWNLOAD_RETRIES:
                wait_seconds = attempt * 2
                print(
                    f"Download failed: {error}\n"
                    f"Retrying in {wait_seconds} seconds..."
                )
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Unable to download:\n{url}\n\n"
        f"Last error: {last_error}"
    )


def download_json(url: str) -> Any:
    """Download and decode a JSON response."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Wind-Modelling-MetaAnalysis/1.0 "
                "(academic research)"
            )
        },
    )

    last_error: Exception | None = None

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            with urllib.request.urlopen(
                request,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            ) as response:
                content = response.read()

            return json.loads(
                content.decode("utf-8")
            )

        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            json.JSONDecodeError,
        ) as error:
            last_error = error

            if attempt < DOWNLOAD_RETRIES:
                time.sleep(attempt * 2)

    raise RuntimeError(
        f"Unable to download JSON:\n{url}\n\n"
        f"Last error: {last_error}"
    )


# =============================================================================
# METEOSTAT DATA
# =============================================================================

def load_meteostat_stations() -> list[dict[str, Any]]:
    """
    Load Meteostat's compressed active-station dump.

    The function accepts both:
    - a JSON list of station dictionaries;
    - a JSON dictionary keyed by station ID.
    """
    download_file(
        url=METEOSTAT_STATIONS_URL,
        destination=METEOSTAT_CACHE,
        force=FORCE_DOWNLOAD,
    )

    try:
        with gzip.open(
            METEOSTAT_CACHE,
            mode="rt",
            encoding="utf-8",
        ) as file:
            raw_data = json.load(file)

    except (gzip.BadGzipFile, json.JSONDecodeError) as error:
        raise RuntimeError(
            "The downloaded Meteostat station file could not "
            "be decoded. Delete the cached file and run the "
            "script again:\n"
            f"{METEOSTAT_CACHE}"
        ) from error

    stations: list[dict[str, Any]] = []

    if isinstance(raw_data, list):
        stations = [
            station
            for station in raw_data
            if isinstance(station, dict)
        ]

    elif isinstance(raw_data, dict):
        # Possible structures:
        # {"id": {...}, "id2": {...}}
        # or {"data": [{...}, {...}]}
        for possible_key in (
            "data",
            "stations",
            "results",
        ):
            possible_value = raw_data.get(possible_key)

            if isinstance(possible_value, list):
                stations = [
                    station
                    for station in possible_value
                    if isinstance(station, dict)
                ]
                break

        if not stations:
            for station_id, station_data in raw_data.items():
                if not isinstance(station_data, dict):
                    continue

                station = station_data.copy()
                station.setdefault("id", station_id)
                stations.append(station)

    else:
        raise ValueError(
            "Unsupported Meteostat JSON structure: "
            f"{type(raw_data).__name__}"
        )

    if not stations:
        raise ValueError(
            "No weather stations were found in the "
            "Meteostat station dump."
        )

    print(
        f"\nActive Meteostat stations loaded: "
        f"{len(stations):,}"
    )

    return stations


def get_station_country(station: dict[str, Any]) -> str:
    """Extract a station's ISO alpha-2 country code."""
    country = station.get("country")

    if isinstance(country, str):
        return country.strip().upper()

    # Defensive handling in case the source schema changes.
    if isinstance(country, dict):
        for key in ("code", "iso2", "alpha2"):
            value = country.get(key)

            if isinstance(value, str):
                return value.strip().upper()

    return ""


def contains_wmo_identifier(value: Any) -> bool:
    """
    Recursively inspect a metadata object for a non-empty WMO identifier.
    """
    if isinstance(value, dict):
        for key, nested_value in value.items():
            normalized_key = str(key).casefold().replace("-", "_")

            if "wmo" in normalized_key:
                if nested_value not in (
                    None,
                    "",
                    [],
                    {},
                ):
                    return True

            if contains_wmo_identifier(nested_value):
                return True

    elif isinstance(value, list):
        return any(
            contains_wmo_identifier(item)
            for item in value
        )

    return False


def station_has_wmo_identifier(
    station: dict[str, Any],
) -> bool:
    """Return True when station metadata contain a WMO identifier."""
    identifiers = station.get("identifiers", {})

    if contains_wmo_identifier(identifiers):
        return True

    # Defensive fallback for flatter schemas.
    for key, value in station.items():
        normalized_key = str(key).casefold().replace("-", "_")

        if "wmo" in normalized_key and value not in (
            None,
            "",
            [],
            {},
        ):
            return True

    return False


def get_station_coordinates(
    station: dict[str, Any],
) -> tuple[float, float] | None:
    """
    Extract and validate latitude and longitude from Meteostat metadata.

    The current lite dump stores coordinates inside ``location``, while the
    fallback also supports a possible flat schema.
    """
    location = station.get("location")

    if isinstance(location, dict):
        latitude = location.get("latitude", location.get("lat"))
        longitude = location.get("longitude", location.get("lon"))
    else:
        latitude = station.get("latitude", station.get("lat"))
        longitude = station.get("longitude", station.get("lon"))

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError):
        return None

    if not (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
    ):
        return None

    return latitude, longitude


# =============================================================================
# COUNTRY AREAS
# =============================================================================

def load_area_cache() -> dict[str, dict[str, Any]]:
    """Read cached World Bank country areas."""
    if not AREA_CACHE.exists():
        return {}

    try:
        with AREA_CACHE.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        return data if isinstance(data, dict) else {}

    except json.JSONDecodeError:
        return {}


def save_area_cache(
    cache: dict[str, dict[str, Any]],
) -> None:
    """Save World Bank area data locally."""
    AREA_CACHE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with AREA_CACHE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            cache,
            file,
            ensure_ascii=False,
            indent=2,
        )


def fetch_country_area(
    alpha3: str,
    cache: dict[str, dict[str, Any]],
) -> tuple[float, int]:
    """
    Retrieve the latest non-null land area from the World Bank.

    Returns:
        area_km2, source_year
    """
    if alpha3 in cache and not FORCE_DOWNLOAD:
        cached = cache[alpha3]

        return (
            float(cached["area_km2"]),
            int(cached["year"]),
        )

    url = WORLD_BANK_AREA_URL.format(
        iso3=alpha3
    )

    response = download_json(url)

    if (
        not isinstance(response, list)
        or len(response) < 2
        or not isinstance(response[1], list)
    ):
        raise ValueError(
            f"Unexpected World Bank response for {alpha3}."
        )

    observations = response[1]

    for observation in observations:
        if not isinstance(observation, dict):
            continue

        value = observation.get("value")
        year = observation.get("date")

        if value is None or year is None:
            continue

        area_km2 = float(value)
        source_year = int(year)

        cache[alpha3] = {
            "area_km2": area_km2,
            "year": source_year,
        }

        save_area_cache(cache)

        return area_km2, source_year

    raise ValueError(
        f"No valid World Bank land-area value found for {alpha3}."
    )


# =============================================================================
# ANALYSIS
# =============================================================================

def estimate_equivalent_spacing(
    area_km2: float,
    station_count: int,
) -> float | None:
    """
    Approximate characteristic square-grid spacing.

    sqrt(area / number_of_stations) is not the observed nearest-neighbour
    distance. It is only an intuitive equivalent-grid spacing.
    """
    if station_count <= 0:
        return None

    return math.sqrt(
        area_km2 / station_count
    )


def calculate_mean_nearest_station_distance(
    stations: list[dict[str, Any]],
) -> tuple[float | None, int]:
    """
    Calculate the observed mean nearest-station distance in kilometres.

    A Haversine BallTree is queried for two neighbours because the closest
    result is the station itself. The second result is its nearest station.

    Returns:
        mean distance in km (or None), number of stations with valid coordinates
    """
    coordinates = [
        coordinate
        for station in stations
        if (coordinate := get_station_coordinates(station)) is not None
    ]

    valid_coordinate_count = len(coordinates)

    if valid_coordinate_count < 2:
        return None, valid_coordinate_count

    coordinates_radians = np.radians(
        np.asarray(coordinates, dtype=float)
    )

    tree = BallTree(
        coordinates_radians,
        metric="haversine",
    )

    angular_distances, _ = tree.query(
        coordinates_radians,
        k=2,
    )

    nearest_distances_km = (
        angular_distances[:, 1] * EARTH_RADIUS_KM
    )

    return (
        float(np.mean(nearest_distances_km)),
        valid_coordinate_count,
    )


def calculate_station_density(
    stations: list[dict[str, Any]],
) -> pd.DataFrame:
    """Calculate counts and density indicators for selected countries."""
    area_cache = load_area_cache()
    rows: list[dict[str, Any]] = []

    for country_info in COUNTRIES:
        country = country_info["country"]
        alpha2 = country_info["alpha2"]
        alpha3 = country_info["alpha3"]
        service = country_info["service"]

        print(f"Processing {country}...")

        country_stations = [
            station
            for station in stations
            if get_station_country(station) == alpha2
        ]

        active_count = len(country_stations)

        wmo_count = sum(
            station_has_wmo_identifier(station)
            for station in country_stations
        )

        area_km2, area_year = fetch_country_area(
            alpha3=alpha3,
            cache=area_cache,
        )

        active_density = (
            active_count / area_km2 * 1000
            if area_km2 > 0
            else 0
        )

        wmo_density = (
            wmo_count / area_km2 * 1000
            if area_km2 > 0
            else 0
        )

        equivalent_spacing = estimate_equivalent_spacing(
            area_km2=area_km2,
            station_count=active_count,
        )

        (
            mean_nearest_distance,
            stations_with_coordinates,
        ) = calculate_mean_nearest_station_distance(
            country_stations
        )

        rows.append(
            {
                "Country": country,
                "National Meteorological Service": service,
                "ISO2": alpha2,
                "ISO3": alpha3,
                "Land Area (km²)": round(area_km2),
                "Area Reference Year": area_year,
                "Active Meteostat Stations": active_count,
                "WMO-Identified Stations": wmo_count,
                (
                    "Active Meteostat Stations "
                    "per 1,000 km²"
                ): round(active_density, 4),
                (
                    "WMO-Identified Stations "
                    "per 1,000 km²"
                ): round(wmo_density, 4),
                (
                    "Equivalent Grid Spacing (km)"
                ): (
                    round(equivalent_spacing, 2)
                    if equivalent_spacing is not None
                    else None
                ),
                "Stations with Valid Coordinates": (
                    stations_with_coordinates
                ),
                "Mean Nearest-Station Distance (km)": (
                    round(mean_nearest_distance, 2)
                    if mean_nearest_distance is not None
                    else None
                ),
            }
        )

    result = pd.DataFrame(rows)

    return result.sort_values(
        FIGURE_COUNT_COLUMN,
        ascending=False,
    ).reset_index(drop=True)


# =============================================================================
# OUTPUT
# =============================================================================

def save_results(result: pd.DataFrame) -> None:
    """Save CSV, Excel and LaTeX outputs."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        CSV_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    result.to_excel(
        EXCEL_OUTPUT,
        index=False,
        engine="openpyxl",
    )

    latex_columns = [
        "Country",
        "National Meteorological Service",
        "Land Area (km²)",
        "Active Meteostat Stations",
        "Active Meteostat Stations per 1,000 km²",
        "Mean Nearest-Station Distance (km)",
    ]

    latex_df = result[latex_columns].copy()

    latex_table = latex_df.to_latex(
        index=False,
        escape=True,
        caption=(
            "Density and mean nearest-neighbour distance of active publicly "
            "indexed meteorological stations in selected countries."
        ),
        label="tab:meteorological_station_density",
        float_format=lambda value: f"{value:.3f}",
        column_format="llrrrr",
    )

    LATEX_OUTPUT.write_text(
        latex_table,
        encoding="utf-8",
    )


def create_figure(result: pd.DataFrame) -> None:
    """Create a horizontal station-density bar chart."""
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_df = result.sort_values(
        FIGURE_COUNT_COLUMN,
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    bars = ax.barh(
        plot_df["Country"],
        plot_df[FIGURE_COUNT_COLUMN],
    )

    ax.set_xlabel(
        "Active Meteostat stations per 1,000 km²"
    )

    ax.set_ylabel("Country")

    ax.set_title(
        "Density of active publicly indexed "
        "meteorological stations"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    ax.set_axisbelow(True)

    maximum = float(
        plot_df[FIGURE_COUNT_COLUMN].max()
    )

    offset = max(
        maximum * 0.01,
        0.001,
    )

    for bar in bars:
        value = float(bar.get_width())

        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8,
        )

    ax.set_xlim(
        0,
        maximum * 1.15 if maximum > 0 else 1,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_OUTPUT,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_metadata(result: pd.DataFrame) -> None:
    """Write methodology and source notes beside the outputs."""
    area_years = sorted(
        result["Area Reference Year"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    metadata = f"""Station-density analysis

Station source:
{METEOSTAT_STATIONS_URL}

Meteostat dataset:
Lite dump containing active stations listed in the Meteostat public
weather-station directory.

Area source:
World Bank indicator AG.LND.TOTL.K2 (Land area, km²).

Countries analysed:
{", ".join(result["Country"].tolist())}

Area reference years found:
{", ".join(map(str, area_years))}

Important interpretation:
'Active Meteostat Stations' represents active publicly indexed stations
available in the Meteostat directory. It must not be interpreted as an
exhaustive count of all stations operated by each national meteorological
service.

'WMO-Identified Stations' is a narrower subset whose Meteostat metadata
contain a WMO identifier. This may exclude official stations without a WMO
identifier and therefore should not be treated as the complete national
official network.

Density formula:
station count / land area in km² * 1,000

Equivalent grid spacing:
sqrt(land area / station count)

The equivalent spacing is an intuitive square-grid approximation and is
not an observed mean nearest-neighbour distance.

Mean nearest-station distance:
For each active Meteostat station with valid coordinates, the nearest other
station within the same country is found using a BallTree with the Haversine
metric. Angular distances are converted to kilometres using an Earth radius
of {EARTH_RADIUS_KM} km, and the arithmetic mean is reported by country.

National meteorological service:
The service name is descriptive country-level context. Active station counts
refer to the Meteostat directory and not exclusively to stations operated by
the listed national meteorological service.
"""

    METADATA_OUTPUT.write_text(
        metadata,
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("Starting meteorological station-density analysis.\n")

    stations = load_meteostat_stations()

    result = calculate_station_density(
        stations=stations,
    )

    save_results(result)
    create_figure(result)
    save_metadata(result)

    print("\nAnalysis completed.\n")

    print(
        result[
            [
                "Country",
                "National Meteorological Service",
                "Land Area (km²)",
                "Active Meteostat Stations",
                (
                    "Active Meteostat Stations "
                    "per 1,000 km²"
                ),
                "Mean Nearest-Station Distance (km)",
            ]
        ].to_string(index=False)
    )

    print("\nSaved CSV:")
    print(CSV_OUTPUT)

    print("\nSaved Excel:")
    print(EXCEL_OUTPUT)

    print("\nSaved LaTeX table:")
    print(LATEX_OUTPUT)

    print("\nSaved figure:")
    print(FIGURE_OUTPUT)

    print("\nSaved methodology notes:")
    print(METADATA_OUTPUT)


if __name__ == "__main__":
    main()