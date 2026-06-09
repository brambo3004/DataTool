"""
Centrale traject- en metreringlogica.

Deze module houdt bewust meerdere begrippen uit elkaar:

- voorkeurstrajectlengte: de beste beschikbare trajectlengte voor de UI;
- exacte/objectmetrering: begin/eind- of rangevelden, als die beschikbaar zijn;
- grove objectmetrering: losse metreringpunten uit het objectpaspoort;
- administratieve naamlengte: afgeleid uit de onderhoudsprojectnaam;
- objectlengte: geometrische/technische lengte van individuele objecten
  (die wordt in ``overview_map`` berekend).

Waarom centraal?
Dezelfde logica is nodig in Overzicht én later in Project Adviseur voor
conceptnamen van nieuwe onderhoudscomplexen. Door begin/einde, segmentering,
bronkwaliteit en afrondingsregels hier te bundelen voorkomen we dat modules
verschillende definities van "trajectlengte" gaan gebruiken.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import re
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd

from .utils import clean_display_value


# Kolomnamen die we in iASSET-exports of voorbereidingsbestanden kunnen
# tegenkomen. De lijst is bewust ruim, maar parsing blijft streng: alleen
# positieve numerieke metreringen worden gebruikt.
HM_POINT_COLUMN_CANDIDATES = (
    "Metrering",
    "metrering",
    "Hectometrering",
    "hectometrering",
    "ligging hectometrering 1",
    "Ligging hectometrering 1",
)

HM_START_COLUMN_CANDIDATES = (
    "Metrering begin",
    "metrering begin",
    "Beginmetrering",
    "beginmetrering",
    "Begin metrering",
    "begin metrering",
    "Metrering van",
    "metrering van",
    "Van metrering",
    "van metrering",
    "HM begin",
    "hm begin",
    "Hectometrering begin",
    "hectometrering begin",
)

HM_END_COLUMN_CANDIDATES = (
    "Metrering einde",
    "metrering einde",
    "Eindmetrering",
    "eindmetrering",
    "Einde metrering",
    "einde metrering",
    "Metrering tot",
    "metrering tot",
    "Tot metrering",
    "tot metrering",
    "HM einde",
    "hm einde",
    "Hectometrering einde",
    "hectometrering einde",
)

PROJECT_COLUMN_CANDIDATES = (
    "Onderhoudsproject",
    "onderhoudsproject",
    "Advies_Onderhoudsproject",
)


@dataclass(frozen=True)
class ProjectRange:
    """Begin- en eindmetrering zoals herleid uit een projectnaam."""

    start_km: float
    end_km: float

    @property
    def length_m(self) -> float:
        """Lengte in meters van de administratieve naamrange."""
        return max(0.0, (self.end_km - self.start_km) * 1000.0)


@dataclass(frozen=True)
class TrajectoryQuantity:
    """
    Trajectmetriek voor één legenda- of projectgroep.

    ``length_m`` is de voorkeurslengte voor de UI. Vanaf v0.31 is losse
    objectpaspoort-metrering niet meer automatisch de voorkeursbron, omdat die
    in de praktijk vaak op honderd meter is afgerond. Expliciete begin/eind-
    metrering blijft wel sterker dan een onderhoudsprojectnaam.
    """

    length_m: float | None
    segment_count: int
    source: str
    source_quality: str = "niet beschikbaar"

    # Exacte/objectmetrering als er echte begin/eind- of rangevelden zijn.
    precise_length_m: float | None = None
    precise_segment_count: int = 0
    precise_source: str = "niet beschikbaar"
    precise_quality: str = "niet beschikbaar"

    # Objectmetrering in brede zin: ook grove losse punten uit de paspoortdata.
    object_metrering_length_m: float | None = None
    object_metrering_segment_count: int = 0
    object_metrering_source: str = "niet beschikbaar"
    object_metrering_quality: str = "niet beschikbaar"

    # Administratieve trajectrange uit de onderhoudsprojectnaam.
    name_length_m: float | None = None
    name_segment_count: int = 0
    name_source: str = "niet beschikbaar"

    difference_m: float | None = None
    warning: str = ""


def parse_positive_number(value: Any) -> float | None:
    """
    Parseer Nederlandse en Engelse getalnotaties naar een positief getal.

    Voorbeelden:
    - ``"18,475"`` -> ``18.475``
    - ``"18.475"`` -> ``18.475``
    - ``"hm 18,475"`` -> ``18.475``
    """
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        number = float(value)
        return number if number >= 0 else None

    text = clean_display_value(value)
    if not text:
        return None

    text = text.replace("m²", "").replace("m2", "").replace("meter", "")
    text = re.sub(r"[^0-9,.\-]", "", text)
    if not text or text in {"-", ".", ","}:
        return None

    # Nederlandse notatie: 1.234,56 -> 1234.56.
    # Voor hectometrering zoals 18.475 laten we de punt juist staan.
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

    return number if number >= 0 else None


def parse_hm_value(value: Any) -> float | None:
    """Parseer metrering naar kilometers."""
    return parse_positive_number(value)


def _decimal_from_float_km(value_km: float) -> Decimal:
    """Zet een km-waarde robuust om naar Decimal voor afrondingsregels."""
    return Decimal(str(float(value_km)))


def round_km_to_nearest_5m(value_km: float) -> float:
    """
    Rond een metrering af op het beheer-vijfmeterpunt binnen hetzelfde hectometerblok.

    De handmatige beheerregel is net iets anders dan standaard afronden:
    - exact op het honderdmeterpunt blijft exact staan;
    - direct na het honderdmeterpunt gaat de waarde naar het volgende 5-meterpunt;
    - direct voor het volgende honderdmeterpunt blijft de waarde binnen hetzelfde
      blok, dus bijvoorbeeld 14.398 wordt 14.395 en niet 14.400.

    Voorbeelden:
    - 14.000 -> 14.000
    - 14.001 t/m 14.007 -> 14.005
    - 14.008 t/m 14.012 -> 14.010
    - 14.093 t/m 14.099 -> 14.095
    """
    value_m = _decimal_from_float_km(value_km) * Decimal("1000")
    base_m = (value_m / Decimal("100")).to_integral_value(rounding=ROUND_FLOOR) * Decimal("100")
    offset_m = value_m - base_m

    if offset_m == 0:
        rounded_m = base_m
    else:
        rounded_offset = (offset_m / Decimal("5")).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * Decimal("5")
        if rounded_offset <= 0:
            rounded_offset = Decimal("5")
        elif rounded_offset >= 100:
            rounded_offset = Decimal("95")
        rounded_m = base_m + rounded_offset

    return float((rounded_m / Decimal("1000")).quantize(Decimal("0.001")))


def round_km_to_name_tenth_up(value_km: float) -> float:
    """
    Rond een metrering naar boven af op één decimaal voor de projectnaam.

    Voorbeelden:
    - 3.800 -> 3.8
    - 3.805 -> 3.9
    - 3.900 -> 3.9

    Dit is bewust geen normale ``round()``. De naamgevingsregel is: zodra het
    knippunt voorbij een tiende kilometer ligt, gaat de projectnaam naar de
    volgende tiende.
    """
    value = _decimal_from_float_km(value_km)
    rounded = (value * Decimal("10")).to_integral_value(rounding=ROUND_CEILING) / Decimal("10")
    return float(rounded.quantize(Decimal("0.1")))


def format_name_hm(value_km: float) -> str:
    """
    Format een projectnaam-metrering als ``xx.x``.

    Kilometers onder 10 krijgen een voorloopnul: ``4.2`` wordt ``04.2``.
    """
    rounded = round_km_to_name_tenth_up(value_km)
    return f"{rounded:04.1f}" if rounded < 10 else f"{rounded:.1f}"


def parse_hm_values_from_text(value: Any) -> list[float]:
    """
    Haal alle metreringwaarden uit een tekstveld.

    Dit is bedoeld voor cellen die een range bevatten, zoals
    ``"12.300 - 14.405"`` of ``"van 12,300 tot 14,405"``.
    """
    text = clean_display_value(value)
    if not text:
        return []

    matches = re.findall(r"\d+(?:[,.]\d+)?", text)
    values: list[float] = []
    for match in matches:
        number = parse_hm_value(match)
        if number is not None:
            values.append(number)

    return values


def first_existing_column(gdf: gpd.GeoDataFrame | pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Zoek de eerste kandidaatkolom die in de data bestaat."""
    for column in candidates:
        if column in gdf.columns:
            return column
    return None


def parse_project_range(value: Any) -> ProjectRange | None:
    """
    Haal begin- en eindmetrering uit een onderhoudsprojectnaam.

    Voorbeelden:
    - ``N354-HRB-18.5-18.7``
    - ``N398 HRB 00,1 00,4``

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

    start = parse_hm_value(match.group(1))
    end = parse_hm_value(match.group(2))
    if start is None or end is None or start == end:
        return None

    left, right = sorted((start, end))
    return ProjectRange(start_km=left, end_km=right)


def _merge_intervals(intervals: list[tuple[float, float]], max_gap_km: float = 0.025) -> list[tuple[float, float]]:
    """
    Voeg overlappende of bijna-aansluitende intervallen samen.

    ``max_gap_km`` staat standaard op 25 meter. Dat vangt kleine afrondings- en
    knipverschillen af zonder losse wegdelen zomaar samen te trekken.
    """
    clean_intervals = sorted(
        (min(start, end), max(start, end))
        for start, end in intervals
        if start is not None and end is not None and abs(end - start) > 0
    )
    if not clean_intervals:
        return []

    merged: list[tuple[float, float]] = [clean_intervals[0]]
    for start, end in clean_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap_km:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def intervals_length_m(intervals: list[tuple[float, float]], max_gap_km: float = 0.025) -> tuple[float | None, int]:
    """Bereken totale trajectlengte uit begin/eind-intervallen."""
    merged = _merge_intervals(intervals, max_gap_km=max_gap_km)
    if not merged:
        return None, 0

    total_km = sum(end - start for start, end in merged)
    if total_km <= 0:
        return None, 0

    return total_km * 1000.0, len(merged)


def hm_points_length_m(hm_values: list[float], max_gap_km: float = 0.25) -> tuple[float | None, int]:
    """
    Benader trajectlengte uit losse metreringwaarden.

    Belangrijk:
    - We nemen niet simpelweg min/max over de hele groep, omdat hetzelfde
      legenda-item op twee losse wegdelen kan voorkomen.
    - Waarden worden in segmenten geknipt wanneer er een duidelijke sprong zit.
    - Een segment met maar één hm-punt krijgt geen lengte.

    Deze bron is vanaf v0.31 expliciet "grof": in iASSET-paspoortexports is
    ``Metrering`` vaak afgerond op honderd meter en daarmee niet geschikt als
    precieze trajectbron.
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


def object_metrering_quantity_for_group(group: gpd.GeoDataFrame | pd.DataFrame) -> TrajectoryQuantity:
    """
    Bereken trajectlengte uit objectmetrering.

    Voorkeur binnen objectmetrering:
    1. expliciete begin/eind-kolommen: beste interne bron;
    2. range in één metreringkolom: bruikbaar als beheer-/rangebron;
    3. losse metreringpunten: alleen grove benadering.
    """
    start_column = first_existing_column(group, HM_START_COLUMN_CANDIDATES)
    end_column = first_existing_column(group, HM_END_COLUMN_CANDIDATES)

    if start_column and end_column:
        intervals: list[tuple[float, float]] = []
        for _, row in group.iterrows():
            start = parse_hm_value(row.get(start_column))
            end = parse_hm_value(row.get(end_column))
            if start is not None and end is not None and start != end:
                intervals.append((start, end))

        length_m, segment_count = intervals_length_m(intervals)
        if length_m is not None:
            source = f"objectmetrering '{start_column}'-'{end_column}'"
            return TrajectoryQuantity(
                length_m=length_m,
                segment_count=segment_count,
                source=source,
                source_quality="precies",
                precise_length_m=length_m,
                precise_segment_count=segment_count,
                precise_source=source,
                precise_quality="precies",
                object_metrering_length_m=length_m,
                object_metrering_segment_count=segment_count,
                object_metrering_source=source,
                object_metrering_quality="precies",
            )

    point_column = first_existing_column(group, HM_POINT_COLUMN_CANDIDATES)
    if point_column is None:
        return TrajectoryQuantity(length_m=None, segment_count=0, source="niet beschikbaar")

    intervals = []
    points = []
    for value in group[point_column]:
        values = parse_hm_values_from_text(value)
        if len(values) >= 2:
            intervals.append((values[0], values[-1]))
        elif len(values) == 1:
            points.append(values[0])

    length_m, segment_count = intervals_length_m(intervals)
    if length_m is not None:
        source = f"objectmetrering range uit '{point_column}'"
        return TrajectoryQuantity(
            length_m=length_m,
            segment_count=segment_count,
            source=source,
            source_quality="beheer-range",
            precise_length_m=length_m,
            precise_segment_count=segment_count,
            precise_source=source,
            precise_quality="beheer-range",
            object_metrering_length_m=length_m,
            object_metrering_segment_count=segment_count,
            object_metrering_source=source,
            object_metrering_quality="beheer-range",
        )

    length_m, segment_count = hm_points_length_m(points)
    if length_m is not None:
        source = f"objectmetrering punten uit '{point_column}'"
        return TrajectoryQuantity(
            length_m=length_m,
            segment_count=segment_count,
            source=f"{source} (grof)",
            source_quality="grof",
            # Niet vullen als precise_length_m: dit is géén precieze bron.
            object_metrering_length_m=length_m,
            object_metrering_segment_count=segment_count,
            object_metrering_source=source,
            object_metrering_quality="grof",
        )

    return TrajectoryQuantity(length_m=None, segment_count=0, source="niet beschikbaar")


def project_name_quantity_for_group(
    group: gpd.GeoDataFrame | pd.DataFrame,
    all_objects: gpd.GeoDataFrame | pd.DataFrame,
    value_column: str = "__overview_value",
) -> TrajectoryQuantity:
    """
    Bereken administratieve naamlengte uit onderhoudsprojectnamen.

    We gebruiken een projectrange alleen voor een legenda-item als hetzelfde
    project binnen de volledige rijstrookset niet over meerdere legenda-items is
    verdeeld. Anders zou dezelfde projectrange dubbel meetellen.
    """
    project_column = first_existing_column(group, PROJECT_COLUMN_CANDIDATES)
    if project_column is None or value_column not in all_objects.columns:
        return TrajectoryQuantity(length_m=None, segment_count=0, source="niet beschikbaar")

    length_m = 0.0
    segment_count = 0

    for project_name in group[project_column].dropna().unique():
        project_range = parse_project_range(project_name)
        if project_range is None:
            continue

        same_project = all_objects[all_objects[project_column].astype(str) == str(project_name)]
        unique_values = {
            clean_display_value(value)
            for value in same_project.get(value_column, pd.Series(dtype=object))
            if clean_display_value(value)
        }
        if len(unique_values) > 1:
            continue

        length_m += project_range.length_m
        segment_count += 1

    if length_m <= 0:
        return TrajectoryQuantity(length_m=None, segment_count=0, source="niet beschikbaar")

    return TrajectoryQuantity(
        length_m=length_m,
        segment_count=segment_count,
        source="onderhoudsprojectnaam",
        source_quality="administratief",
        name_length_m=length_m,
        name_segment_count=segment_count,
        name_source="onderhoudsprojectnaam",
    )


def combine_trajectory_sources(sources: list[str]) -> str:
    """Vat gebruikte bronnen samen zonder herhaling."""
    real_sources = [source for source in sources if source and source != "niet beschikbaar"]
    if not real_sources:
        return "niet beschikbaar"

    return " + ".join(dict.fromkeys(real_sources))


def _source_is_precise_enough(quantity: TrajectoryQuantity) -> bool:
    """Bepaal of objectmetrering sterk genoeg is om voorkeursbron te zijn."""
    return quantity.source_quality in {"precies", "beheer-range"} and quantity.length_m is not None


def trajectory_quantity_for_group(
    group: gpd.GeoDataFrame | pd.DataFrame,
    all_objects: gpd.GeoDataFrame | pd.DataFrame,
    value_column: str = "__overview_value",
    warning_threshold_m: float = 50.0,
    warning_threshold_ratio: float = 0.10,
) -> TrajectoryQuantity:
    """
    Bereken voorkeurs-trajectlengte en vergelijkingsbronnen voor één groep.

    v0.31-bronhiërarchie:
    1. expliciete begin/eind- of range-metrering;
    2. onderhoudsprojectnaam als administratieve bron;
    3. losse objectmetreringpunten alleen als grove fallback.

    Hiermee voorkomen we dat de afgeronde iASSET-kolom ``Metrering`` ten
    onrechte een preciezer ogende trajectlengte geeft dan de projectnaam.
    """
    object_metrering = object_metrering_quantity_for_group(group)
    name = project_name_quantity_for_group(group, all_objects, value_column=value_column)

    if _source_is_precise_enough(object_metrering):
        preferred_length = object_metrering.length_m
        preferred_segments = object_metrering.segment_count
        preferred_source = object_metrering.source
        preferred_quality = object_metrering.source_quality
    elif name.length_m is not None:
        preferred_length = name.length_m
        preferred_segments = name.segment_count
        preferred_source = name.source
        preferred_quality = name.source_quality
    elif object_metrering.length_m is not None:
        preferred_length = object_metrering.length_m
        preferred_segments = object_metrering.segment_count
        preferred_source = object_metrering.source
        preferred_quality = object_metrering.source_quality
    else:
        preferred_length = None
        preferred_segments = 0
        preferred_source = "niet beschikbaar"
        preferred_quality = "niet beschikbaar"

    difference_m: float | None = None
    warning_parts: list[str] = []

    object_length = object_metrering.object_metrering_length_m
    if object_length is not None and name.length_m is not None:
        difference_m = name.length_m - object_length
        denominator = max(object_length, 1.0)
        if abs(difference_m) >= warning_threshold_m or abs(difference_m) / denominator >= warning_threshold_ratio:
            if object_metrering.object_metrering_quality == "grof":
                warning_parts.append(
                    "Objectmetrering lijkt grof afgerond; onderhoudsprojectnaam is gebruikt als administratieve voorkeurslengte."
                )
            else:
                warning_parts.append(
                    "Naamlengte wijkt duidelijk af van objectmetrering; controleer begin/eindmetrering."
                )
    elif object_metrering.object_metrering_quality == "grof" and name.length_m is None:
        warning_parts.append(
            "Alleen grove objectmetrering beschikbaar; trajectlengte is een benadering."
        )

    return TrajectoryQuantity(
        length_m=preferred_length,
        segment_count=preferred_segments,
        source=preferred_source,
        source_quality=preferred_quality,
        precise_length_m=object_metrering.precise_length_m,
        precise_segment_count=object_metrering.precise_segment_count,
        precise_source=object_metrering.precise_source,
        precise_quality=object_metrering.precise_quality,
        object_metrering_length_m=object_metrering.object_metrering_length_m,
        object_metrering_segment_count=object_metrering.object_metrering_segment_count,
        object_metrering_source=object_metrering.object_metrering_source,
        object_metrering_quality=object_metrering.object_metrering_quality,
        name_length_m=name.name_length_m,
        name_segment_count=name.name_segment_count,
        name_source=name.name_source,
        difference_m=difference_m,
        warning=" ".join(warning_parts),
    )
