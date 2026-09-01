#!/usr/bin/env python3
"""
Usporedba kopnene i otočne morske obale za odabrane europske države.

Izvor:
    Eurostat GISCO, Countries 2024, mjerilo 1:1 000 000, EPSG:4326

Metoda:
    - preuzimaju se službeni poligoni država i posebno izdvojene obalne linije;
    - zadržavaju se samo europski teritoriji;
    - najveći povezani poligon države definira se kao mainland;
    - svi ostali morski poligoni definiraju se kao islands;
    - obalne linije povezuju se s odgovarajućim poligonom uz toleranciju 250 m;
    - duljina se računa geodetski na elipsoidu WGS84.

Pokretanje:
    pip install geopandas shapely pyproj pandas openpyxl requests
    python coastline_analysis.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from pyproj import Geod
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


COUNTRIES = {
    "HRV": "Croatia",
    "SVN": "Slovenia",
    "ITA": "Italy",
    "MNE": "Montenegro",
    "ALB": "Albania",
    "GRC": "Greece",
    "NOR": "Norway",
    "ESP": "Spain",
    "FRA": "France",
    "DNK": "Denmark",
    "GBR": "United Kingdom",
}

GISCO_BASE = (
    "https://gisco-services.ec.europa.eu/distribution/v2/"
    "countries/geojson"
)
REGIONS_URL = f"{GISCO_BASE}/CNTR_RG_01M_2024_4326.geojson"
COASTLINES_URL = f"{GISCO_BASE}/CNTR_BN_01M_2024_4326_COASTL.geojson"

# Približna granica Europe služi samo za uklanjanje udaljenih/prekomorskih
# teritorija. Dodatna pravila ispod uklanjaju Kanare i Svalbard.
EUROPE_BOUNDS = (-25.0, 34.0, 45.0, 72.0)  # min lon, min lat, max lon, max lat
MATCH_TOLERANCE_M = 250.0
GEOD = Geod(ellps="WGS84")


def download(url: str, destination: Path, force: bool = False) -> Path:
    """Preuzmi datoteku ako već nije spremljena lokalno."""
    if destination.exists() and not force:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {destination.name} ...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def polygon_parts(geometry) -> list[Polygon]:
    """Vrati sve neprazne poligone iz Polygon/MultiPolygon geometrije."""
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [part for part in geometry.geoms if not part.is_empty]
    return []


def is_european_part(part: Polygon, iso3: str) -> bool:
    """Zadrži europski teritorij, a ukloni udaljene teritorije."""
    point = part.representative_point()
    min_lon, min_lat, max_lon, max_lat = EUROPE_BOUNDS

    if not (min_lon <= point.x <= max_lon and min_lat <= point.y <= max_lat):
        return False

    # Kanarski otoci te španjolski teritoriji na afričkoj obali.
    if iso3 == "ESP" and (point.y < 35.0 or point.x < -10.0):
        return False

    # Svalbard i Jan Mayen nisu dio kontinentalne europske usporedbe.
    if iso3 == "NOR" and point.y > 72.0:
        return False

    return True


def country_coastline(coastlines: gpd.GeoDataFrame, iso3: str):
    """Izdvoji sve GISCO obalne segmente koji pripadaju državi."""
    mask = (coastlines["LEFT_URI"] == iso3) | (
        coastlines["RIGHT_URI"] == iso3
    )
    selected = coastlines.loc[mask, "geometry"]
    if selected.empty:
        raise ValueError(f"No coastline found for {iso3}.")
    return unary_union(selected.to_list())


def geodesic_length_km(geometry) -> float:
    """Geodetska duljina geometrije na elipsoidu WGS84."""
    if geometry.is_empty:
        return 0.0
    return abs(GEOD.geometry_length(geometry)) / 1000.0


def analyse_country(
    regions: gpd.GeoDataFrame,
    coastlines: gpd.GeoDataFrame,
    iso3: str,
    country_name: str,
) -> dict:
    """Izračunaj kopnenu, otočnu i ukupnu morsku obalu jedne države."""
    country_rows = regions.loc[regions["ISO3_CODE"] == iso3]
    if country_rows.empty:
        raise ValueError(f"Country polygon not found for {iso3}.")

    geometry = unary_union(country_rows.geometry.to_list())
    parts = [
        part for part in polygon_parts(geometry)
        if is_european_part(part, iso3)
    ]
    if not parts:
        raise ValueError(f"No European polygon parts retained for {iso3}.")

    # Jednaka europska projekcija u metrima služi samo za prostorno povezivanje.
    parts_gdf = gpd.GeoDataFrame(
        {"part_id": range(len(parts))},
        geometry=parts,
        crs="EPSG:4326",
    ).to_crs("EPSG:3035")

    coast = country_coastline(coastlines, iso3)
    coast_metric = gpd.GeoSeries([coast], crs="EPSG:4326").to_crs(
        "EPSG:3035"
    ).iloc[0]

    # Mainland se određuje prema najvećoj površini u projekciji jednakih površina.
    mainland_id = int(parts_gdf.geometry.area.idxmax())

    mainland_coast_km = 0.0
    island_coast_km = 0.0
    for idx, part_metric in parts_gdf.geometry.items():
        # Uzmemo samo onaj službeni obalni segment koji je uz rub ovog poligona.
        matching_coast = coast_metric.intersection(
            part_metric.boundary.buffer(MATCH_TOLERANCE_M)
        )
        if matching_coast.is_empty:
            continue

        matching_coast_wgs84 = gpd.GeoSeries(
            [matching_coast], crs="EPSG:3035"
        ).to_crs("EPSG:4326").iloc[0]
        length_km = geodesic_length_km(matching_coast_wgs84)

        if idx == mainland_id:
            mainland_coast_km += length_km
        else:
            island_coast_km += length_km

    total_coast_km = mainland_coast_km + island_coast_km
    island_share = (
        island_coast_km / total_coast_km * 100.0
        if total_coast_km > 0
        else float("nan")
    )

    return {
        "Country": country_name,
        "Mainland Coastline (km)": mainland_coast_km,
        "Island Coastline (km)": island_coast_km,
        "Total Coastline (km)": total_coast_km,
        "Island Coastline Share (%)": island_share,
    }


def latex_table(df: pd.DataFrame, destination: Path) -> None:
    """Spremi tablicu spremnu za uključivanje u LaTeX dokument."""
    display = df[
        [
            "Country",
            "Mainland Coastline (km)",
            "Island Coastline (km)",
            "Total Coastline (km)",
            "Island Coastline Share (%)",
        ]
    ].copy()

    rows = []
    for row in display.itertuples(index=False, name=None):
        country, mainland, islands, total, share = row
        rows.append(
            f"{country} & \\num{{{mainland:.1f}}} & \\num{{{islands:.1f}}} "
            f"& \\num{{{total:.1f}}} & \\num{{{share:.1f}}} \\\\"
        )

    latex = "\n".join(
        [
            r"\begin{table}[htbp]",
            r"    \centering",
            r"    \caption{Comparison of mainland and island marine coastline "
            r"lengths calculated from harmonised Eurostat GISCO 2024 spatial data.}",
            r"    \label{tab:european_mainland_island_coastline}",
            r"    \begin{tabular}{lrrrr}",
            r"        \toprule",
            r"        Country & Mainland coastline (km) & Island coastline (km) "
            r"& Total coastline (km) & Island share (\%) \\",
            r"        \midrule",
            *[f"        {row}" for row in rows],
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    destination.write_text(latex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for CSV, Excel and LaTeX results.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/gisco"),
        help="Directory for downloaded GISCO files.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download source files again even if they already exist.",
    )
    args = parser.parse_args()

    region_path = download(
        REGIONS_URL,
        args.data_dir / "CNTR_RG_01M_2024_4326.geojson",
        args.force_download,
    )
    coastline_path = download(
        COASTLINES_URL,
        args.data_dir / "CNTR_BN_01M_2024_4326_COASTL.geojson",
        args.force_download,
    )

    print("Reading Eurostat GISCO data ...")
    regions = gpd.read_file(region_path)
    coastlines = gpd.read_file(coastline_path)

    rows = []
    for iso3, country_name in COUNTRIES.items():
        print(f"Analysing {country_name} ...")
        rows.append(
            analyse_country(
                regions, coastlines, iso3, country_name
            )
        )

    result = pd.DataFrame(rows).sort_values(
        "Island Coastline Share (%)", ascending=False
    )

    numeric_columns = result.select_dtypes(include="number").columns
    result[numeric_columns] = result[numeric_columns].round(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "european_coastline_comparison.csv"
    xlsx_path = args.output_dir / "european_coastline_comparison.xlsx"
    tex_path = args.output_dir / "european_coastline_comparison.tex"

    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    result.to_excel(xlsx_path, index=False, sheet_name="Coastline comparison")
    latex_table(result, tex_path)

    print("\nAnalysis completed.\n")
    print(result.to_string(index=False))
    print("\nSaved:")
    for path in (csv_path, xlsx_path, tex_path):
        print(f"  {path}")
    print(
        "\nMethodological note: values are reproducible calculations from "
        "harmonised GISCO geometry, not national official coastline totals. "
        "Results depend on the 1:1,000,000 source scale."
    )


if __name__ == "__main__":
    main()