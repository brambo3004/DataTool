"""
Overzichtskaart voor rijstroken.

Deze module bouwt een Folium-kaart voor het tabblad "Overzicht".
De kaart is bewust alleen-lezen: hij visualiseert rijstroken per gekozen
attribuut en past geen iASSET-data aan.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any

import folium
import geopandas as gpd
import pandas as pd

from .config import OVERVIEW_ATTRIBUTE_ALIASES, OVERVIEW_POPUP_COLUMNS
from .utils import clean_display_value, normalize_text
from .trajectory import (
    TrajectoryQuantity as CentralTrajectoryQuantity,
    combine_trajectory_sources,
    trajectory_quantity_for_group,
)


UNKNOWN_LABEL = "Onbekend"

# Kandidaten voor hoeveelheidssamenvattingen in het Overzicht.
# iASSET-exports kunnen kolomnamen net anders schrijven. Daarom gebruiken we
# meerdere veilige kandidaten en vallen we terug op geometrie waar dat kan.
LENGTH_COLUMN_CANDIDATES = (
    "totale lengte",
    "Totale lengte",
    "lengte",
    "Lengte",
    "lengte_m",
    "Lengte (m)",
    "Lengte m",
)

WIDTH_COLUMN_CANDIDATES = (
    "Administratieve breedte",
    "administratieve breedte",
    "totale breedte",
    "Totale breedte",
    "breedte",
    "Breedte",
    "breedte2",
    "Breedte2",
)

AREA_COLUMN_CANDIDATES = (
    "Berekende lengte/oppervlakte",
    "Lengte_Oppervlakte",
    "Oppervlakte",
    "oppervlakte",
    "Oppervlakte (m2)",
    "Oppervlakte (m²)",
    "oppervlakte_m2",
    "oppervlakte_m²",
    "Area",
    "area",
)


TRAJECTORY_HM_COLUMN_CANDIDATES = (
    "Metrering",
    "metrering",
    "hectometrering",
    "ligging hectometrering 1",
)

TRAJECTORY_PROJECT_COLUMN_CANDIDATES = (
    "Onderhoudsproject",
    "onderhoudsproject",
    "Advies_Onderhoudsproject",
)


def _column_has_display_values(gdf: gpd.GeoDataFrame, column: str) -> bool:
    """
    Controleer of een kolom minstens één inhoudelijke waarde heeft.

    Dit voorkomt dat een lege alias-kolom, die door de loader is aangemaakt,
    een echte bronkolom met data overschaduwt.
    """
    if column not in gdf.columns:
        return False

    return any(clean_display_value(value) for value in gdf[column])


@dataclass(frozen=True)
class LegendItem:
    """Eén regel in de legenda, inclusief hoeveelheden voor v0.29."""

    label: str
    color: str
    object_count: int = 0
    # Objectlengte = som van alle rijstrookobjecten. Dit kan hoger zijn dan de
    # trajectlengte als er meerdere rijstroken/richtingen over hetzelfde stuk liggen.
    length_m: float = 0.0
    # Trajectlengte = lengte langs de weg/het onderhoudsdeel. Deze is bedoeld
    # voor de legenda en sluit beter aan bij begin- en eindmetrering.
    trajectory_length_m: float | None = None
    trajectory_segment_count: int = 0
    # Administratieve trajectlengte uit de projectnaam. Deze kan afwijken
    # doordat namen op hectometerpunten worden afgerond.
    trajectory_name_length_m: float | None = None
    trajectory_object_metrering_length_m: float | None = None
    trajectory_difference_m: float | None = None
    trajectory_warning: str = ""
    trajectory_source: str = "niet beschikbaar"
    trajectory_source_quality: str = "niet beschikbaar"
    trajectory_object_metrering_source: str = "niet beschikbaar"
    trajectory_object_metrering_quality: str = "niet beschikbaar"
    area_m2: float | None = None


@dataclass
class OverviewMapResult:
    """Resultaat van de overzichtskaart."""

    folium_map: folium.Map
    row_count: int
    legend_items: list[LegendItem]
    selected_column: str | None
    total_length_m: float = 0.0
    total_trajectory_length_m: float | None = None
    total_trajectory_segment_count: int = 0
    total_trajectory_name_length_m: float | None = None
    total_trajectory_difference_m: float | None = None
    total_area_m2: float | None = None
    length_source: str = "niet beschikbaar"
    trajectory_source: str = "niet beschikbaar"
    trajectory_source_quality: str = "niet beschikbaar"
    area_source: str = "niet beschikbaar"


def resolve_overview_attribute(gdf: gpd.GeoDataFrame, attribute_label: str) -> str | None:
    """
    Vertaal een gebruikerslabel naar de beste beschikbare kolom in de data.

    Waarom?
    De oude voorbeeldtool gebruikt bijvoorbeeld `Soort verharding_N`, terwijl
    onze huidige app vaak `verhardingssoort` gebruikt. Door aliases centraal te
    behandelen, blijft het tabblad bruikbaar bij licht afwijkende exports.

    We kiezen bij voorkeur een kolom met echte waarden. Dat is nodig omdat de
    loader ontbrekende verwachte kolommen soms als lege kolom aanmaakt.
    """
    candidates = OVERVIEW_ATTRIBUTE_ALIASES.get(attribute_label, [attribute_label])
    first_existing: str | None = None

    for column in candidates:
        if column not in gdf.columns:
            continue

        if first_existing is None:
            first_existing = column

        if _column_has_display_values(gdf, column):
            return column

    return first_existing


def available_overview_attributes(gdf: gpd.GeoDataFrame) -> list[str]:
    """
    Geef de attributen terug die daadwerkelijk gevisualiseerd kunnen worden.

    Een attribuut wordt alleen aangeboden als er op de rijstroken minstens één
    inhoudelijke waarde beschikbaar is.
    """
    rijstroken = _rijstroken_only(gdf)
    if rijstroken.empty:
        return []

    available: list[str] = []
    for label in OVERVIEW_ATTRIBUTE_ALIASES:
        column = resolve_overview_attribute(rijstroken, label)
        if column is not None and _column_has_display_values(rijstroken, column):
            available.append(label)

    return available


def _rijstroken_only(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Filter de data op rijstroken.

    We gebruiken `subthema_clean` als die bestaat; anders normaliseren we
    `subthema` zelf. Zo blijft de functie ook testbaar met kleine testframes.
    """
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=getattr(gdf, "crs", None))

    if "subthema_clean" in gdf.columns:
        mask = gdf["subthema_clean"].apply(normalize_text) == "rijstrook"
    elif "subthema" in gdf.columns:
        mask = gdf["subthema"].apply(normalize_text) == "rijstrook"
    else:
        mask = pd.Series(False, index=gdf.index)

    return gdf.loc[mask].copy()


def _display_value(value: Any) -> str:
    """Maak een attribuutwaarde geschikt voor legenda, tooltip en popup."""
    cleaned = clean_display_value(value)
    return cleaned if cleaned else UNKNOWN_LABEL


def _parse_positive_number(value: Any) -> float | None:
    """
    Parseer Nederlandse en Engelse getalnotaties veilig naar een positief getal.

    Voorbeelden die werken: ``"1.234,5"``, ``"1234.5"``, ``"40,96 m²"``.
    Niet-numerieke of negatieve waarden worden genegeerd.
    """
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        number = float(value)
        return number if number > 0 else None

    text = clean_display_value(value)
    if not text:
        return None

    text = text.replace("m²", "").replace("m2", "").replace("meter", "")
    text = re.sub(r"[^0-9,.\-]", "", text)
    if not text or text in {"-", ".", ","}:
        return None

    # Nederlandse notatie: 1.234,56 -> 1234.56
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")

    try:
        number = float(text)
    except ValueError:
        return None

    return number if number > 0 else None


def _first_numeric_column(gdf: gpd.GeoDataFrame, candidates: tuple[str, ...]) -> str | None:
    """Zoek de eerste kandidaatkolom met minstens één positief numeriek getal."""
    for column in candidates:
        if column not in gdf.columns:
            continue

        if any(_parse_positive_number(value) is not None for value in gdf[column]):
            return column

    return None


def _safe_projected_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Geef een kopie terug in meters.

    De normale loader zet geometrie al om naar EPSG:28992. Deze fallback maakt
    de hoeveelheidsberekening robuuster voor kleine tests of afwijkende exports.
    """
    projected = gdf.copy()

    try:
        crs = projected.crs
        if crs is not None and getattr(crs, "is_geographic", False):
            return projected.to_crs(epsg=28992)
    except Exception:
        return projected

    return projected


def _is_line_geometry(geometry: Any) -> bool:
    """Controleer of een geometrie een lijngeometrie is."""
    geom_type = getattr(geometry, "geom_type", "")
    return geom_type in {"LineString", "MultiLineString"}


def _is_polygon_geometry(geometry: Any) -> bool:
    """Controleer of een geometrie een vlakgeometrie is."""
    geom_type = getattr(geometry, "geom_type", "")
    return geom_type in {"Polygon", "MultiPolygon"}


def _geometry_length_series_m(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Bereken lijnlengte in meters, maar alleen voor lijngeometrieën.

    Voor polygonen gebruiken we bewust niet de perimeter als weglengte; dat zou
    te hoge en misleidende kilometers geven.
    """
    projected = _safe_projected_gdf(gdf)
    values: list[float] = []
    for geometry in projected.geometry:
        if geometry is None or getattr(geometry, "is_empty", True) or not _is_line_geometry(geometry):
            values.append(0.0)
            continue

        try:
            values.append(float(geometry.length))
        except Exception:
            values.append(0.0)

    return pd.Series(values, index=gdf.index, dtype="float64")


def _geometry_area_series_m2(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Bereken oppervlakte in m² voor polygonen en multipolygonen."""
    projected = _safe_projected_gdf(gdf)
    values: list[float] = []
    for geometry in projected.geometry:
        if geometry is None or getattr(geometry, "is_empty", True) or not _is_polygon_geometry(geometry):
            values.append(0.0)
            continue

        try:
            values.append(float(geometry.area))
        except Exception:
            values.append(0.0)

    return pd.Series(values, index=gdf.index, dtype="float64")


def _numeric_column_series(gdf: gpd.GeoDataFrame, column: str | None) -> pd.Series:
    """Zet een numerieke bronkolom om naar een meters/m²-serie."""
    if column is None or column not in gdf.columns:
        return pd.Series(0.0, index=gdf.index, dtype="float64")

    values = [_parse_positive_number(value) or 0.0 for value in gdf[column]]
    return pd.Series(values, index=gdf.index, dtype="float64")


def _quantity_sources(gdf: gpd.GeoDataFrame) -> tuple[pd.Series, str, pd.Series, str]:
    """
    Bepaal lengte en oppervlakte voor de Overzicht-legenda.

    Lengtebron:
    1. expliciete lengtekolom als die aanwezig is;
    2. lijngeometrie als de rijstrook als lijn is geëxporteerd;
    3. bij polygonen: oppervlakte / administratieve breedte, duidelijk als
       schatting gemarkeerd.

    Oppervlaktebron:
    1. polygongeometrie;
    2. expliciete oppervlaktekolom als geometrie geen vlakoppervlakte geeft.
    """
    length_column = _first_numeric_column(gdf, LENGTH_COLUMN_CANDIDATES)
    length_m = _numeric_column_series(gdf, length_column)
    length_source = f"kolom '{length_column}'" if length_column else "niet beschikbaar"

    if float(length_m.sum()) <= 0:
        geometry_length = _geometry_length_series_m(gdf)
        if float(geometry_length.sum()) > 0:
            length_m = geometry_length
            length_source = "lijngeometrie"

    area_m2 = _geometry_area_series_m2(gdf)
    area_source = "polygongeometrie" if float(area_m2.sum()) > 0 else "niet beschikbaar"

    if float(area_m2.sum()) <= 0:
        area_column = _first_numeric_column(gdf, AREA_COLUMN_CANDIDATES)
        area_m2 = _numeric_column_series(gdf, area_column)
        area_source = f"kolom '{area_column}'" if area_column else "niet beschikbaar"

    if float(length_m.sum()) <= 0 and float(area_m2.sum()) > 0:
        width_column = _first_numeric_column(gdf, WIDTH_COLUMN_CANDIDATES)
        widths = _numeric_column_series(gdf, width_column)
        estimated_values = []
        for area, width in zip(area_m2, widths):
            if area > 0 and width > 0:
                estimated_values.append(area / width)
            else:
                estimated_values.append(0.0)

        estimated_length = pd.Series(estimated_values, index=gdf.index, dtype="float64")
        if float(estimated_length.sum()) > 0:
            length_m = estimated_length
            length_source = f"geschat uit oppervlakte / kolom '{width_column}'"

    return length_m, length_source, area_m2, area_source



def _first_existing_column(gdf: gpd.GeoDataFrame, candidates: tuple[str, ...]) -> str | None:
    """Zoek de eerste kandidaatkolom die in de data bestaat."""
    for column in candidates:
        if column in gdf.columns:
            return column
    return None


def _parse_hm_value(value: Any) -> float | None:
    """
    Parseer een iASSET-metrering naar kilometers.

    Lege, negatieve of niet-numerieke waarden worden genegeerd. Dit is bewust
    strenger dan de sorteervariant: voor trajectlengte willen we geen fallback
    optellen alsof het echte data is.
    """
    number = _parse_positive_number(value)
    if number is None:
        return None
    return float(number)


@dataclass(frozen=True)
class _ProjectRange:
    """Herleid hm-bereik uit een onderhoudsprojectnaam."""

    start_km: float
    end_km: float

    @property
    def length_m(self) -> float:
        return max(0.0, (self.end_km - self.start_km) * 1000.0)


def _parse_project_range(value: Any) -> _ProjectRange | None:
    """
    Haal begin- en eindmetrering uit een onderhoudsprojectnaam.

    Voorbeelden:
    - N354-HRB-18.5-18.7
    - N398 HRB 00,1 00,4

    De functie is tolerant voor oude namen, maar geeft ``None`` terug als het
    patroon niet betrouwbaar herkend kan worden.
    """
    text = clean_display_value(value)
    if not text:
        return None

    text = text.upper().replace(",", ".")
    match = re.search(
        r"\bN\s*\d{3,4}\s*[-_/ ]+\s*[A-Z0-9]+\s*[-_/ ]+(\d+(?:\.\d+)?)\s*[-_/ ]+(\d+(?:\.\d+)?)\b",
        text,
    )
    if not match:
        return None

    start = _parse_hm_value(match.group(1))
    end = _parse_hm_value(match.group(2))
    if start is None or end is None or start == end:
        return None

    left, right = sorted((start, end))
    return _ProjectRange(start_km=left, end_km=right)


@dataclass(frozen=True)
class _TrajectoryQuantity:
    """Trajectlengte voor één legenda-item."""

    length_m: float | None
    segment_count: int
    source: str


def _hm_segments_length_m(hm_values: list[float], max_gap_km: float = 0.25) -> tuple[float | None, int]:
    """
    Benader trajectlengte uit losse metreringwaarden.

    Belangrijk:
    - We nemen niet simpelweg min/max over de hele groep, omdat hetzelfde
      legenda-item op twee losse wegdelen kan voorkomen.
    - Waarden worden in segmenten geknipt wanneer er een duidelijke sprong zit.
    - Een segment met maar één hm-punt krijgt geen lengte, omdat begin/einde dan
      niet betrouwbaar uit de data af te leiden is.

    De drempel van 0,25 km sluit aan bij hectometerwaarden die soms niet elk
    honderdmeterpunt bevatten. Een sprong groter dan 250 meter zien we als
    vermoedelijk nieuw los trajectdeel.
    """
    clean_values = sorted({round(float(value), 6) for value in hm_values if value is not None})
    if len(clean_values) < 2:
        return None, 0

    segments: list[list[float]] = [[clean_values[0]]]
    for value in clean_values[1:]:
        if value - segments[-1][-1] > max_gap_km:
            segments.append([value])
        else:
            segments[-1].append(value)

    total_km = 0.0
    usable_segments = 0
    for segment in segments:
        if len(segment) < 2:
            continue
        total_km += max(segment) - min(segment)
        usable_segments += 1

    if total_km <= 0:
        return None, 0

    return total_km * 1000.0, usable_segments


def _trajectory_quantity_for_group(group: gpd.GeoDataFrame, all_rijstroken: gpd.GeoDataFrame) -> _TrajectoryQuantity:
    """
    Bereken trajectlengte voor één legenda-item.

    Voorkeur:
    1. onderhoudsprojectnaam met hm-bereik, maar alleen als dat project binnen
       de volledige rijstrookset niet over meerdere legenda-items verdeeld is;
    2. metreringsegmenten als terugval.

    Waarom deze volgorde?
    De onderhoudsprojectnaam bevat vaak expliciet het beoogde trajectbereik
    (bijv. ``N398-HRB-00.1-00.4``). Dat sluit beter aan bij de beheerpraktijk dan
    het optellen van rijstrookobjecten. Als een onderhoudsproject meerdere
    legenda-items bevat, gebruiken we die projectrange niet voor elk item, want
    dat zou dubbeltellingen veroorzaken.
    """
    project_column = _first_existing_column(group, TRAJECTORY_PROJECT_COLUMN_CANDIDATES)
    length_from_projects_m = 0.0
    project_segments = 0

    if project_column is not None and "__overview_value" in all_rijstroken.columns:
        for project_name in group[project_column].dropna().unique():
            project_range = _parse_project_range(project_name)
            if project_range is None:
                continue

            same_project = all_rijstroken[all_rijstroken[project_column].astype(str) == str(project_name)]
            unique_values = {
                clean_display_value(value)
                for value in same_project.get("__overview_value", pd.Series(dtype=object))
                if clean_display_value(value)
            }
            if len(unique_values) > 1:
                # Het project bevat meerdere legenda-items. De projectrange is
                # dan te grof voor één specifieke legenda-categorie.
                continue

            length_from_projects_m += project_range.length_m
            project_segments += 1

    if length_from_projects_m > 0:
        return _TrajectoryQuantity(
            length_m=length_from_projects_m,
            segment_count=project_segments,
            source="onderhoudsprojectnaam",
        )

    hm_column = _first_existing_column(group, TRAJECTORY_HM_COLUMN_CANDIDATES)
    if hm_column is not None:
        hm_values = [
            hm
            for hm in (_parse_hm_value(value) for value in group[hm_column])
            if hm is not None
        ]
        hm_length_m, hm_segments = _hm_segments_length_m(hm_values)
        if hm_length_m is not None:
            return _TrajectoryQuantity(
                length_m=hm_length_m,
                segment_count=hm_segments,
                source=f"metreringkolom '{hm_column}'",
            )

    return _TrajectoryQuantity(length_m=None, segment_count=0, source="niet beschikbaar")


def _combine_trajectory_sources(sources: list[str]) -> str:
    """Vat de gebruikte bronnen voor trajectlengte samen."""
    real_sources = [source for source in sources if source and source != "niet beschikbaar"]
    if not real_sources:
        return "niet beschikbaar"

    unique_sources = list(dict.fromkeys(real_sources))
    return " + ".join(unique_sources)


def _format_km(length_m: float) -> str:
    """Formatteer meters als kilometers voor Nederlandse UI-tekst."""
    return f"{length_m / 1000:.2f}".replace(".", ",")


def _format_m2(area_m2: float | None) -> str:
    """Formatteer vierkante meters met punt als duizendtalseparator."""
    if area_m2 is None:
        return "n.v.t."

    return f"{area_m2:,.0f}".replace(",", ".")


def _numeric_sort_key(value: str) -> tuple[int, float | str]:
    """
    Sorteer legenda-items: numerieke waarden oplopend, onbekend achteraan.
    """
    if value == UNKNOWN_LABEL:
        return (1, value)

    number = pd.to_numeric(str(value).replace(",", "."), errors="coerce")
    if pd.notna(number):
        return (0, float(number))

    return (0, value.lower())


def _is_numeric_attribute(values: list[str]) -> bool:
    """
    Bepaal of een attribuut vooral numeriek is.

    Bij jaren en wegvaknummers willen we de legenda oplopend sorteren.
    """
    real_values = [value for value in values if value != UNKNOWN_LABEL]
    if not real_values:
        return False

    numeric_count = 0
    for value in real_values:
        number = pd.to_numeric(str(value).replace(",", "."), errors="coerce")
        if pd.notna(number):
            numeric_count += 1

    return numeric_count == len(real_values)


def _sort_legend_values(values: list[str]) -> list[str]:
    """Sorteer legenda-items numeriek als dat kan, anders alfabetisch."""
    unique_values = list(dict.fromkeys(values))

    if _is_numeric_attribute(unique_values):
        return sorted(unique_values, key=_numeric_sort_key)

    return sorted(unique_values, key=lambda value: (value == UNKNOWN_LABEL, value.lower()))


# Doorlopende kleurenschaal voor het Overzicht-tabblad.
# Lage/vroege waarden starten koel blauw, hoge/recente waarden eindigen warm rood.
# Dit voorkomt de "allegaartje"-legenda die ontstaat bij willekeurige categoriekleuren.
COLOR_RAMP_STOPS: tuple[tuple[float, str], ...] = (
    (0.00, "#2c7bb6"),  # blauw
    (0.20, "#00a6ca"),  # blauwgroen
    (0.35, "#00ccbc"),  # turquoise
    (0.50, "#ffff8c"),  # geel
    (0.65, "#f9d057"),  # geeloranje
    (0.80, "#f29e2e"),  # oranje
    (1.00, "#d7191c"),  # rood
)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    """Zet een hexkleur om naar RGB-componenten."""
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """Zet RGB-componenten om naar een hexkleur."""
    return "#" + "".join(f"{max(0, min(255, value)):02x}" for value in rgb)


def _interpolate_color(left: str, right: str, fraction: float) -> str:
    """
    Interpoleer lineair tussen twee hexkleuren.

    `fraction=0` geeft de linkerkleur, `fraction=1` de rechterkleur.
    """
    left_rgb = _hex_to_rgb(left)
    right_rgb = _hex_to_rgb(right)

    rgb = tuple(
        round(left_rgb[i] + (right_rgb[i] - left_rgb[i]) * fraction)
        for i in range(3)
    )
    return _rgb_to_hex(rgb)


def _color_from_ramp(ratio: float) -> str:
    """
    Geef een kleur uit de doorlopende kleurenschaal.

    De ratio ligt tussen 0 en 1. Waarden buiten dat bereik worden afgeknipt.
    """
    ratio = max(0.0, min(1.0, ratio))

    for stop_index in range(1, len(COLOR_RAMP_STOPS)):
        left_pos, left_color = COLOR_RAMP_STOPS[stop_index - 1]
        right_pos, right_color = COLOR_RAMP_STOPS[stop_index]

        if ratio <= right_pos:
            span = right_pos - left_pos
            if span <= 0:
                return right_color

            fraction = (ratio - left_pos) / span
            return _interpolate_color(left_color, right_color, fraction)

    return COLOR_RAMP_STOPS[-1][1]


def _color_for_index(index: int, total: int) -> str:
    """
    Geef een kleur terug op basis van de positie in de gesorteerde legenda.

    Waarom deze aanpak?
    Voor jaren en wegvaknummers wil je een visuele volgorde: oud/laag aan de
    koele kant van het spectrum en nieuw/hoog aan de warme kant. Voor tekstuele
    attributen gebruiken we dezelfde schaal op alfabetische volgorde; dat is
    niet inhoudelijk ordinaal, maar geeft wel een rustige legenda zonder
    willekeurige kleurensprongen.
    """
    if total <= 1:
        return _color_from_ramp(0.5)

    ratio = index / (total - 1)
    return _color_from_ramp(ratio)


def build_value_color_mapping(values: list[str]) -> tuple[dict[str, str], list[LegendItem]]:
    """
    Bouw kleurmapping en legenda-items voor de opgegeven attribuutwaarden.

    `Onbekend` krijgt bewust geen plek in de kleurenschaal, maar blijft grijs.
    Daardoor loopt de schaal altijd van de laagste/beginnende echte waarde naar
    de hoogste/eindigende echte waarde.
    """
    sorted_values = _sort_legend_values(values)
    known_values = [value for value in sorted_values if value != UNKNOWN_LABEL]
    total_known = len(known_values)

    color_by_value: dict[str, str] = {}
    legend_items: list[LegendItem] = []

    known_index = 0
    for value in sorted_values:
        if value == UNKNOWN_LABEL:
            color = "#bdbdbd"
        else:
            color = _color_for_index(known_index, total_known)
            known_index += 1

        color_by_value[value] = color
        legend_items.append(LegendItem(label=value, color=color))

    return color_by_value, legend_items


def _base_map(gdf_web: gpd.GeoDataFrame) -> folium.Map:
    """Maak een basiskaart die op de rijstroken wordt ingezoomd."""
    minx, miny, maxx, maxy = gdf_web.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap",
        max_zoom=22,
    )

    if gdf_web.total_bounds is not None:
        m.fit_bounds([[miny, minx], [maxy, maxx]])

    return m


def _build_popup_html(row: pd.Series, selected_label: str, selected_column: str) -> str:
    """Maak compacte popup-inhoud voor één rijstrook."""
    lines = []

    name = clean_display_value(row.get("naam", "")) or clean_display_value(row.get("nummer", ""))
    if name:
        lines.append(f"<b>Naam:</b> {html.escape(name)}")

    selected_value = _display_value(row.get(selected_column, ""))
    lines.append(f"<b>{html.escape(selected_label)}:</b> {html.escape(selected_value)}")

    for label, candidates in OVERVIEW_POPUP_COLUMNS.items():
        column = next((candidate for candidate in candidates if candidate in row.index), None)
        if column is None:
            continue

        value = _display_value(row.get(column, ""))
        lines.append(f"<b>{html.escape(label)}:</b> {html.escape(value)}")

    return "<br>".join(lines)


def _add_legend(m: folium.Map, title: str, legend_items: list[LegendItem]) -> None:
    """Voeg een Leaflet-achtige legenda linksonder toe."""
    if not legend_items:
        legend_html = "Geen data beschikbaar."
    else:
        rows = []
        for item in legend_items:
            quantity_parts = []
            if item.object_count:
                quantity_parts.append(f"{item.object_count} obj.")
            if item.trajectory_length_m is not None:
                segment_text = f", {item.trajectory_segment_count} deeltraject(en)" if item.trajectory_segment_count else ""
                quantity_parts.append(f"traject {_format_km(item.trajectory_length_m)} km{segment_text}")
            if (
                item.trajectory_name_length_m is not None
                and item.trajectory_length_m is not None
                and abs(item.trajectory_name_length_m - item.trajectory_length_m) >= 1
            ):
                quantity_parts.append(f"naam {_format_km(item.trajectory_name_length_m)} km")
            if item.area_m2 is not None:
                quantity_parts.append(f"{_format_m2(item.area_m2)} m²")
            if item.length_m:
                quantity_parts.append(f"object {_format_km(item.length_m)} km")

            quantity_html = (
                "<div style='font-size:11px;color:#666;margin-top:1px;'>"
                + html.escape(" · ".join(quantity_parts))
                + "</div>"
                if quantity_parts
                else ""
            )

            rows.append(
                "<div style='display:flex;align-items:flex-start;margin:4px 0;'>"
                f"<span style='background:{html.escape(item.color)};"
                "width:18px;height:18px;display:inline-block;margin-right:8px;"
                "opacity:0.8;border:1px solid #333;flex:0 0 18px;'></span>"
                "<span>"
                f"<span>{html.escape(item.label)}</span>"
                f"{quantity_html}"
                "</span>"
                "</div>"
            )
        legend_html = "\n".join(rows)

    html_block = f"""
    <div style="
        position: fixed;
        bottom: 18px;
        left: 18px;
        z-index: 9999;
        background: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
        max-height: 70vh;
        max-width: 280px;
        overflow-y: auto;
        font-family: sans-serif;
        font-size: 12px;
    ">
        <b>{html.escape(title)}</b><br>
        {legend_html}
    </div>
    """
    m.get_root().html.add_child(folium.Element(html_block))


def build_overview_map(
    road_gdf: gpd.GeoDataFrame,
    attribute_label: str,
) -> OverviewMapResult:
    """
    Bouw de alleen-lezen overzichtskaart voor rijstroken.

    Parameters
    ----------
    road_gdf:
        GeoDataFrame van de geselecteerde weg in RD New (EPSG:28992).
    attribute_label:
        Label zoals gekozen in de UI, bijvoorbeeld `Jaar deklaag`.
    """
    rijstroken = _rijstroken_only(road_gdf)

    if rijstroken.empty:
        # Fallbackkaart op Nederland, zodat Streamlit toch iets kan tonen.
        fallback_map = folium.Map(location=[52.2, 5.4], zoom_start=8, tiles="OpenStreetMap", max_zoom=22)
        _add_legend(fallback_map, attribute_label, [])
        return OverviewMapResult(
            folium_map=fallback_map,
            row_count=0,
            legend_items=[],
            selected_column=None,
        )

    selected_column = resolve_overview_attribute(rijstroken, attribute_label)
    if selected_column is None:
        fallback_map = folium.Map(location=[52.2, 5.4], zoom_start=8, tiles="OpenStreetMap", max_zoom=22)
        _add_legend(fallback_map, attribute_label, [])
        return OverviewMapResult(
            folium_map=fallback_map,
            row_count=len(rijstroken),
            legend_items=[],
            selected_column=None,
        )

    values = [_display_value(value) for value in rijstroken[selected_column]]
    color_by_value, legend_items = build_value_color_mapping(values)

    rijstroken = rijstroken.copy()
    rijstroken["__overview_value"] = values
    rijstroken["__overview_color"] = [color_by_value[value] for value in values]

    length_m, length_source, area_m2, area_source = _quantity_sources(rijstroken)
    rijstroken["__overview_length_m"] = length_m
    rijstroken["__overview_area_m2"] = area_m2

    quantity_rows = (
        rijstroken.groupby("__overview_value", dropna=False)
        .agg(
            object_count=("__overview_value", "size"),
            length_m=("__overview_length_m", "sum"),
            area_m2=("__overview_area_m2", "sum"),
        )
        .to_dict("index")
    )

    trajectory_by_value: dict[str, _TrajectoryQuantity] = {}
    for value, group in rijstroken.groupby("__overview_value", dropna=False):
        trajectory_by_value[str(value)] = trajectory_quantity_for_group(group, rijstroken)

    enriched_legend_items: list[LegendItem] = []
    trajectory_sources: list[str] = []
    trajectory_source_qualities: list[str] = []
    total_trajectory_length_m = 0.0
    total_trajectory_segment_count = 0
    total_trajectory_name_length_m = 0.0

    for item in legend_items:
        quantity = quantity_rows.get(item.label, {})
        area_value = float(quantity.get("area_m2", 0.0) or 0.0)
        trajectory = trajectory_by_value.get(item.label, CentralTrajectoryQuantity(None, 0, "niet beschikbaar"))
        if trajectory.length_m is not None:
            total_trajectory_length_m += trajectory.length_m
            total_trajectory_segment_count += trajectory.segment_count
            trajectory_sources.append(trajectory.source)
            if getattr(trajectory, "source_quality", "niet beschikbaar") != "niet beschikbaar":
                trajectory_source_qualities.append(trajectory.source_quality)
        if trajectory.name_length_m is not None:
            total_trajectory_name_length_m += trajectory.name_length_m

        enriched_legend_items.append(
            LegendItem(
                label=item.label,
                color=item.color,
                object_count=int(quantity.get("object_count", 0) or 0),
                length_m=float(quantity.get("length_m", 0.0) or 0.0),
                trajectory_length_m=trajectory.length_m,
                trajectory_segment_count=trajectory.segment_count,
                trajectory_name_length_m=trajectory.name_length_m,
                trajectory_object_metrering_length_m=trajectory.object_metrering_length_m,
                trajectory_difference_m=trajectory.difference_m,
                trajectory_warning=trajectory.warning,
                trajectory_source=trajectory.source,
                trajectory_source_quality=trajectory.source_quality,
                trajectory_object_metrering_source=trajectory.object_metrering_source,
                trajectory_object_metrering_quality=trajectory.object_metrering_quality,
                area_m2=area_value if area_value > 0 else None,
            )
        )
    legend_items = enriched_legend_items

    total_length_m = float(rijstroken["__overview_length_m"].sum())
    total_trajectory_length = total_trajectory_length_m if total_trajectory_length_m > 0 else None
    total_trajectory_name_length = total_trajectory_name_length_m if total_trajectory_name_length_m > 0 else None
    total_trajectory_difference = (
        total_trajectory_name_length - total_trajectory_length
        if total_trajectory_name_length is not None and total_trajectory_length is not None
        else None
    )
    trajectory_source = combine_trajectory_sources(trajectory_sources)
    trajectory_source_quality = combine_trajectory_sources(trajectory_source_qualities)
    total_area_value = float(rijstroken["__overview_area_m2"].sum())
    total_area_m2 = total_area_value if total_area_value > 0 else None
    if total_area_m2 is None:
        area_source = "niet beschikbaar"

    rijstroken_web = rijstroken.to_crs(epsg=4326)
    m = _base_map(rijstroken_web)

    # Popupvelden voorbereiden in eenvoudige tekstkolommen. We selecteren straks
    # alleen deze kolommen voor GeoJSON, zodat exotische pandas-types uit de
    # bronexport niet per ongeluk JSON-serialisatieproblemen geven.
    popup_fields: list[str] = []
    popup_aliases: list[str] = []

    rijstroken_web["__popup_naam"] = [
        clean_display_value(row.get("naam", "")) or clean_display_value(row.get("nummer", ""))
        for _, row in rijstroken_web.iterrows()
    ]
    if rijstroken_web["__popup_naam"].astype(str).str.len().gt(0).any():
        popup_fields.append("__popup_naam")
        popup_aliases.append("Naam")

    rijstroken_web["__popup_selected"] = rijstroken_web["__overview_value"]
    popup_fields.append("__popup_selected")
    popup_aliases.append(attribute_label)

    for label, candidates in OVERVIEW_POPUP_COLUMNS.items():
        if label == attribute_label:
            continue

        column = next((candidate for candidate in candidates if candidate in rijstroken_web.columns), None)
        if column is None:
            continue

        helper_column = "__popup_" + "".join(ch if ch.isalnum() else "_" for ch in label.lower())
        rijstroken_web[helper_column] = rijstroken_web[column].apply(_display_value)
        popup_fields.append(helper_column)
        popup_aliases.append(label)

    def style_fn(feature):
        props = feature.get("properties", {})
        color = props.get("__overview_color", "#808080")
        return {
            "color": color,
            "fillColor": color,
            "weight": 4,
            "opacity": 0.85,
            "fillOpacity": 0.65,
        }

    tooltip = folium.GeoJsonTooltip(
        fields=["__overview_value"],
        aliases=[attribute_label],
        style="font-size: 11px;",
    )

    popup = folium.GeoJsonPopup(
        fields=popup_fields,
        aliases=popup_aliases,
        localize=False,
        labels=True,
        style="font-size: 12px;",
        max_width=350,
    )

    geojson_columns = list(dict.fromkeys(["geometry", "__overview_value", "__overview_color", *popup_fields]))

    folium.GeoJson(
        rijstroken_web[geojson_columns],
        style_function=style_fn,
        tooltip=tooltip,
        popup=popup,
    ).add_to(m)

    _add_legend(m, attribute_label, legend_items)

    return OverviewMapResult(
        folium_map=m,
        row_count=len(rijstroken),
        legend_items=legend_items,
        selected_column=selected_column,
        total_length_m=total_length_m,
        total_trajectory_length_m=total_trajectory_length,
        total_trajectory_segment_count=total_trajectory_segment_count,
        total_trajectory_name_length_m=total_trajectory_name_length,
        total_trajectory_difference_m=total_trajectory_difference,
        total_area_m2=total_area_m2,
        length_source=length_source if total_length_m > 0 else "niet beschikbaar",
        trajectory_source=trajectory_source,
        trajectory_source_quality=trajectory_source_quality,
        area_source=area_source,
    )


def overview_quantity_dataframe(result: OverviewMapResult) -> pd.DataFrame:
    """
    Maak een tabel met hoeveelheden per legenda-item.

    Deze tabel is bedoeld voor Streamlit-weergave en CSV-download. De waarden
    blijven controle-informatie; ze passen geen iASSET-data aan.
    """
    rows = []
    for item in result.legend_items:
        rows.append(
            {
                "Legenda-item": item.label,
                "Aantal objecten": item.object_count,
                "Trajectlengte voorkeur (km)": (
                    round(item.trajectory_length_m / 1000, 3)
                    if item.trajectory_length_m is not None
                    else None
                ),
                "Trajectlengte naam (km)": (
                    round(item.trajectory_name_length_m / 1000, 3)
                    if item.trajectory_name_length_m is not None
                    else None
                ),
                "Trajectlengte objectmetrering (km)": (
                    round(item.trajectory_object_metrering_length_m / 1000, 3)
                    if item.trajectory_object_metrering_length_m is not None
                    else None
                ),
                "Verschil naam t.o.v. objectmetrering (m)": (
                    round(item.trajectory_difference_m, 1)
                    if item.trajectory_difference_m is not None
                    else None
                ),
                "Trajectlengte (km)": (
                    round(item.trajectory_length_m / 1000, 3)
                    if item.trajectory_length_m is not None
                    else None
                ),
                "Aantal deeltrajecten": item.trajectory_segment_count or None,
                "Trajectlengtebron voorkeur": item.trajectory_source,
                "Bronkwaliteit voorkeur": item.trajectory_source_quality,
                "Objectmetrering bron": item.trajectory_object_metrering_source,
                "Objectmetrering kwaliteit": item.trajectory_object_metrering_quality,
                "Trajectlengte waarschuwing": item.trajectory_warning or "",
                "Oppervlakte (m²)": round(item.area_m2, 1) if item.area_m2 is not None else None,
                "Objectlengte (km)": round(item.length_m / 1000, 3) if item.length_m else 0.0,
                "Kleur": item.color,
            }
        )

    return pd.DataFrame(rows)


def render_overview_map_html(
    result: OverviewMapResult,
    title: str,
    subtitle: str = "",
) -> str:
    """
    Render de actuele Overzicht-kaart als downloadbare HTML.

    Waarom HTML als eerste exportvorm?
    Een Folium-kaart is zelf al Leaflet/HTML/JavaScript. Daardoor kunnen we de
    kaart betrouwbaar exporteren zonder screenshot-tooling of browserdriver.
    De download bevat de gekozen visualisatie, legenda, popups en kaartlaag.
    """
    safe_title = html.escape(clean_display_value(title) or "iASSET Overzicht")
    safe_subtitle = html.escape(clean_display_value(subtitle))

    html_doc = result.folium_map.get_root().render()

    title_tag = f"<title>{safe_title}</title>"
    if re.search(r"<title>.*?</title>", html_doc, flags=re.IGNORECASE | re.DOTALL):
        html_doc = re.sub(
            r"<title>.*?</title>",
            title_tag,
            html_doc,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )
    else:
        html_doc = html_doc.replace("<head>", f"<head>\n    {title_tag}", 1)

    subtitle_html = f"<div style='font-size:12px;color:#444;margin-top:2px;'>{safe_subtitle}</div>" if safe_subtitle else ""
    total_parts = []
    if result.total_trajectory_length_m is not None:
        total_parts.append(f"Totaal trajectlengte: {_format_km(result.total_trajectory_length_m)} km")
    if (
        result.total_trajectory_name_length_m is not None
        and result.total_trajectory_length_m is not None
        and abs(result.total_trajectory_name_length_m - result.total_trajectory_length_m) >= 1
    ):
        total_parts.append(f"Totaal naamlengte: {_format_km(result.total_trajectory_name_length_m)} km")
    if result.total_area_m2 is not None:
        total_parts.append(f"Totaal oppervlak: {_format_m2(result.total_area_m2)} m²")
    if result.total_length_m:
        total_parts.append(f"Totaal objectlengte: {_format_km(result.total_length_m)} km")
    total_html = (
        "<div style='font-size:12px;color:#444;margin-top:4px;'>"
        + html.escape(" | ".join(total_parts))
        + "</div>"
        if total_parts
        else ""
    )
    source_html = (
        "<div style='font-size:11px;color:#666;margin-top:4px;'>"
        f"Trajectlengtebron voorkeur: {html.escape(result.trajectory_source)} "
        f"({html.escape(result.trajectory_source_quality)}). "
        f"Objectlengtebron: {html.escape(result.length_source)}. "
        f"Oppervlaktebron: {html.escape(result.area_source)}."
        "</div>"
    )

    export_panel = f"""
    <div style="
        position: fixed;
        top: 18px;
        right: 18px;
        z-index: 9999;
        background: white;
        padding: 10px 12px;
        border-radius: 5px;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
        max-width: 360px;
        font-family: sans-serif;
        font-size: 13px;
    ">
        <b>{safe_title}</b>
        {subtitle_html}
        {total_html}
        {source_html}
        <div style="font-size:11px;color:#666;margin-top:4px;">
            Alleen-lezen export uit de iASSET Advisor.
        </div>
    </div>
    """

    if "</body>" in html_doc:
        return html_doc.replace("</body>", f"{export_panel}\n</body>", 1)

    return html_doc + export_panel
