from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# POSTAVKE
# ============================================================

EXCEL_FILE = Path("WoS_ai_methods_classified.xlsx")

# Upiši naziv sheeta na kojem se nalazi pripremljena tablica.
SHEET_NAME = "Heatmap"

METHOD_COLUMN = "Row Labels"

PERIOD_COLUMNS = [
    "2005-2009",
    "2010-2014",
    "2015-2019",
    "2020-2025",
]

# Koliko metoda prikazati:
# None = sve metode
# 20 = Top 20 metoda prema ukupnom broju pojavljivanja
TOP_N = 25

# Ne prikazuj metode koje ukupno imaju manje od ovog broja pojavljivanja.
MIN_TOTAL = 1

OUTPUT_NAME = "ai_methods_heatmap"


# ============================================================
# UČITAVANJE PODATAKA
# ============================================================

if not EXCEL_FILE.exists():
    raise FileNotFoundError(
        f"Datoteka nije pronađena: {EXCEL_FILE.resolve()}"
    )

df = pd.read_excel(
    EXCEL_FILE,
    sheet_name=SHEET_NAME,
)

required_columns = [METHOD_COLUMN, *PERIOD_COLUMNS]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "U Excel tablici nedostaju stupci: "
        + ", ".join(missing_columns)
    )

# Zadržavanje samo potrebnih stupaca
df = df[required_columns].copy()

# Uklanjanje praznih redaka i eventualnog Grand Total retka
df = df.dropna(subset=[METHOD_COLUMN])

df[METHOD_COLUMN] = (
    df[METHOD_COLUMN]
    .astype(str)
    .str.replace(" Detected", "", regex=False)
    .str.strip()
)

df = df[
    ~df[METHOD_COLUMN].str.lower().isin(
        ["grand total", "row labels", "nan"]
    )
].copy()

# Pretvaranje vrijednosti u brojeve
for column in PERIOD_COLUMNS:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(0)

# Ukupan broj pojavljivanja po metodi
df["Total"] = df[PERIOD_COLUMNS].sum(axis=1)

# Uklanjanje metoda bez ili s vrlo malo pojavljivanja
df = df[df["Total"] >= MIN_TOTAL]

# Sortiranje prema ukupnoj zastupljenosti
df = df.sort_values(
    "Total",
    ascending=False,
)

if TOP_N is not None:
    df = df.head(TOP_N)


# ============================================================
# PODACI ZA BOJU I OZNAKE
# ============================================================

original_values = df[PERIOD_COLUMNS].to_numpy(dtype=float)

# Logaritamska transformacija koristi se samo za boju.
# log1p(x) znači ln(x + 1), pa se može koristiti i kada je x = 0.
colour_values = np.log1p(original_values)

methods = df[METHOD_COLUMN].tolist()


# ============================================================
# IZRADA HEATMAPE
# ============================================================

figure_height = max(7, len(methods) * 0.38)

fig, ax = plt.subplots(
    figsize=(9.5, figure_height),
)

heatmap = ax.imshow(
    colour_values,
    cmap="Blues",
    aspect="auto",
    interpolation="nearest",
)

# Nazivi vremenskih razdoblja
ax.set_xticks(np.arange(len(PERIOD_COLUMNS)))
ax.set_xticklabels(
    PERIOD_COLUMNS,
    fontsize=10,
)

# Nazivi metoda
ax.set_yticks(np.arange(len(methods)))
ax.set_yticklabels(
    methods,
    fontsize=9,
)

ax.set_xlabel(
    "Publication period",
    fontsize=11,
    labelpad=10,
)

ax.set_ylabel(
    "AI method",
    fontsize=11,
    labelpad=10,
)

ax.set_title(
    "Temporal distribution of AI methods in wind-modelling publications",
    fontsize=13,
    pad=16,
)

# Tanke bijele linije između ćelija
ax.set_xticks(
    np.arange(-0.5, len(PERIOD_COLUMNS), 1),
    minor=True,
)
ax.set_yticks(
    np.arange(-0.5, len(methods), 1),
    minor=True,
)

ax.grid(
    which="minor",
    linewidth=0.8,
    color="white",
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False,
)

ax.tick_params(
    axis="both",
    which="major",
    length=0,
)

# Ispis originalnih vrijednosti.
# Nule ostaju prazne radi veće čitljivosti.
maximum_colour_value = colour_values.max()

for row_index in range(original_values.shape[0]):
    for column_index in range(original_values.shape[1]):

        original_value = int(
            original_values[row_index, column_index]
        )

        if original_value == 0:
            continue

        transformed_value = colour_values[
            row_index,
            column_index,
        ]

        # Bijeli tekst na najtamnijim ćelijama
        text_colour = (
            "white"
            if transformed_value > maximum_colour_value * 0.68
            else "black"
        )

        ax.text(
            column_index,
            row_index,
            str(original_value),
            ha="center",
            va="center",
            fontsize=8.5,
            color=text_colour,
        )


# ============================================================
# LEGENDA
# ============================================================

colour_bar = fig.colorbar(
    heatmap,
    ax=ax,
    fraction=0.035,
    pad=0.025,
)

# Legenda prikazuje originalne vrijednosti, iako je skala logaritamska
legend_counts = [0, 1, 2, 5, 10, 20, 50, 100]

legend_counts = [
    count
    for count in legend_counts
    if count <= original_values.max()
]

legend_positions = np.log1p(legend_counts)

colour_bar.set_ticks(legend_positions)
colour_bar.set_ticklabels(
    [str(count) for count in legend_counts]
)

colour_bar.set_label(
    "Number of occurrences\n(logarithmic colour scale)",
    fontsize=9,
)

colour_bar.ax.tick_params(
    labelsize=8,
    length=0,
)


# Uklanjanje vanjskog okvira
for spine in ax.spines.values():
    spine.set_visible(False)

fig.tight_layout()


# ============================================================
# IZVOZ
# ============================================================

fig.savefig(
    f"{OUTPUT_NAME}.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    f"{OUTPUT_NAME}.pdf",
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    f"{OUTPUT_NAME}.svg",
    bbox_inches="tight",
    facecolor="white",
)

plt.show()

print("Izrađene datoteke:")
print(f"  {OUTPUT_NAME}.png")
print(f"  {OUTPUT_NAME}.pdf")
print(f"  {OUTPUT_NAME}.svg")