"""
Projectgrenzen op een geijkte iASSET-referentieas (v0.34.3).

Deze module draait de eerdere referentieasproef bewust om:

- de iASSET-wegas is de as waarop we werken;
- NWB-hectopunten zijn alleen ijkpunten om routepositie op de iASSET-as
  naar hectometrering te vertalen;
- onderhoudsprojectnamen en objectgeometrie worden alleen diagnostisch
  gecontroleerd;
- er worden géén beheerknippen of iASSET-mutaties gemaakt.

Waarom een aparte module?
De Streamlit-app moet vooral UI blijven. De GIS-logica staat hier, zodat we
haar gericht kunnen testen met kleine synthetische wegen en later kunnen
uitbreiden naar andere wegen of domeinen.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import math
import re
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import linemerge

from .nwb import filter_iasset_wegassen_for_road
from .sorting_diagnostics import project_geometry_range_on_axis
from .trajectory import format_name_hm, parse_project_range
from .utils import clean_display_value, normalize_text


PROJECT_AXIS_SCHEMA_VERSION = "projectaxis-v0.34.3"

HECTOMETER_COLUMN_CANDIDATES = (
    "hectomtrng",
    "hm_val",
    "hectometrering",
    "hectometer",
    "hm",
)

PRIMARY_SUBTHEMES = {
    "rijstrook",
    "parallelweg",
    "fietspad",
    "busbaan",
    "landbouwpad",
}

PROJECT_TYPE_FAMILIES = ("LBP", "HRB", "PW", "FP", "BB")
PROJECT_TYPES_WITH_REQUIRED_SITUERING = {"PW", "FP", "BB", "LBP"}
ALLOWED_REQUIRED_SITUERING_CODES = {"L", "R", "LR"}
STATUS_RANK = {"ok": 0, "overzicht": 0, "projectie": 0, "aandacht": 1, "controleer": 2}
DEFAULT_BOUNDARY_SNAP_TOLERANCE_M = 2.5


@dataclass(frozen=True)
class ProjectAxisDiagnosticsResult:
    """
    Resultaat van de v0.34.3-projectgrensdiagnose.

    Alle tabellen zijn gewone DataFrames. Dat houdt de Streamlit-laag simpel en
    maakt export naar CSV zonder extra conversie mogelijk.
    """

    calibration_anchors: pd.DataFrame
    project_boundaries: pd.DataFrame
    project_coverage: pd.DataFrame
    object_ranges: pd.DataFrame
    warning: str = ""


def _empty_anchor_frame() -> pd.DataFrame:
    """Maak een lege ijkpunttabel met vaste kolommen."""
    return pd.DataFrame(
        columns=[
            "axis_id",
            "axis_naam",
            "Wegnummer",
            "hm_km",
            "route_m",
            "x_rd",
            "y_rd",
            "afstand_tot_iasset_wegas_m",
            "point_count",
            "hm_monotoon",
            "status",
            "waarschuwing",
            "bronkwaliteit",
        ]
    )


def _empty_project_boundary_frame() -> pd.DataFrame:
    """Maak een lege projectgrenstabel met vaste kolommen."""
    return pd.DataFrame(
        columns=[
            "Onderhoudsproject",
            "naam_wegnummer",
            "project_type",
            "project_family",
            "situering",
            "naam_begin_label",
            "naam_eind_label",
            "naam_validatie_status",
            "naam_validatie_melding",
            "axis_id",
            "axis_naam",
            "objecten_in_project",
            "objecten_op_axis",
            "project_begin_km",
            "project_eind_km",
            "project_lengte_m",
            "as_begin_m",
            "as_eind_m",
            "as_lengte_m",
            "lengteverschil_naam_vs_as_m",
            "begin_binnen_ijking",
            "eind_binnen_ijking",
            "begin_buiten_ijkbereik",
            "eind_buiten_ijkbereik",
            "begin_zone_kleur",
            "begin_zone_id",
            "eind_zone_kleur",
            "eind_zone_id",
            "fysiek_object_begin_km",
            "fysiek_object_eind_km",
            "fysiek_object_lengte_m",
            "snap_tolerantie_m",
            "object_begin_dichtstbijzijnde_hm",
            "object_begin_snap_afstand_m",
            "object_begin_gesnapt_naar_hm",
            "object_begin_naamregel",
            "object_eind_dichtstbijzijnde_hm",
            "object_eind_snap_afstand_m",
            "object_eind_gesnapt_naar_hm",
            "object_eind_naamregel",
            "verschil_projectnaam_vs_objectligging_m",
            "objectligging_status",
            "objectligging_melding",
            "status_projectnaam",
            "status_projectgrens",
            "status",
            "advies",
            "waarschuwing",
            "bronkwaliteit",
        ]
    )


def _empty_coverage_frame() -> pd.DataFrame:
    """Maak een lege projectdekkingstabel met vaste kolommen."""
    return pd.DataFrame(
        columns=[
            "axis_id",
            "axis_naam",
            "project_type",
            "project_family",
            "situering",
            "controle_type",
            "van_m",
            "tot_m",
            "lengte_m",
            "hard_gat_van_m",
            "hard_gat_tot_m",
            "hard_gat_lengte_m",
            "naamzone_marge_links_m",
            "naamzone_marge_rechts_m",
            "project_links",
            "project_rechts",
            "dekking_uniek_m",
            "projectbereik_m",
            "ijking_span_m",
            "dekking_pct",
            "status",
            "advies",
            "bronkwaliteit",
        ]
    )


def _empty_object_range_frame() -> pd.DataFrame:
    """Maak een lege objectprojectietabel met vaste kolommen."""
    return pd.DataFrame(
        columns=[
            "sys_id",
            "nummer",
            "subthema",
            "Onderhoudsproject",
            "project_type",
            "project_family",
            "situering",
            "axis_id",
            "axis_naam",
            "route_begin_m",
            "route_eind_m",
            "referentie_begin_km",
            "referentie_eind_km",
            "afstand_tot_as_m",
            "primair_object",
            "status",
            "waarschuwing",
            "bronkwaliteit",
        ]
    )


def _safe_to_rd(gdf: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    """
    Zet een GeoDataFrame veilig om naar RD New.

    iASSET-exports en testdata kunnen CRS-informatie missen. Als coördinaten op
    RD lijken, nemen we EPSG:28992 aan. Bij lon/lat converteren we naar RD. Bij
    fouten geven we een leeg frame terug in plaats van te crashen.
    """
    if gdf is None or getattr(gdf, "empty", True):
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:28992")

    if "geometry" not in gdf.columns:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:28992")

    result = gdf.copy()
    try:
        if result.crs is None:
            minx, miny, maxx, maxy = result.total_bounds
            if -180 <= minx <= 180 and -90 <= miny <= 90 and -180 <= maxx <= 180 and -90 <= maxy <= 90:
                result = result.set_crs(epsg=4326, allow_override=True)
            else:
                result = result.set_crs(epsg=28992, allow_override=True)

        if result.crs is not None and result.crs.to_epsg() != 28992:
            result = result.to_crs(epsg=28992)
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:28992")

    return result


def _is_valid_geometry(geometry: Any) -> bool:
    """Controleer of een geometrie bruikbaar is voor projectie."""
    try:
        return geometry is not None and not geometry.is_empty
    except Exception:
        return False


def _line_like_geometry(geometry: Any) -> Any | None:
    """
    Maak een lijnachtige geometrie bruikbaar voor projectie.

    Shapely kan ook op MultiLineString projecteren, maar linemerge geeft voor
    eenvoudige meerdelige assen vaak een stabielere routevolgorde.
    """
    if not _is_valid_geometry(geometry):
        return None

    try:
        geom_type = geometry.geom_type
    except Exception:
        return None

    if geom_type == "LineString":
        return geometry

    if geom_type == "MultiLineString":
        try:
            merged = linemerge(geometry)
            return merged if _is_valid_geometry(merged) else geometry
        except Exception:
            return geometry

    return None


def _axis_identifier(row: pd.Series, fallback_index: Any) -> str:
    """Bepaal een stabiele id voor een iASSET-wegas."""
    for column in ["nummer", "Object nummer", "naam", "Object naam"]:
        if column in row.index:
            value = clean_display_value(row.get(column))
            if value:
                return value
    return f"wegas-{fallback_index}"


def _axis_name(row: pd.Series, fallback: str) -> str:
    """Bepaal een leesbare asnaam."""
    for column in ["naam", "Object naam", "nummer"]:
        if column in row.index:
            value = clean_display_value(row.get(column))
            if value:
                return value
    return fallback


def _prepare_axes(wegassen_gdf: gpd.GeoDataFrame | None, selected_road: str) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Filter en prepareer iASSET-wegassen voor één geselecteerde weg.

    Corrupte of lege geometrieën worden overgeslagen en als waarschuwing
    teruggegeven. De rest van de diagnose kan daardoor gewoon doorgaan.
    """
    warnings: list[str] = []
    if wegassen_gdf is None or getattr(wegassen_gdf, "empty", True):
        return [], ["Geen iASSET-wegassenbestand beschikbaar voor projectgrensdiagnose."]

    try:
        filtered = filter_iasset_wegassen_for_road(wegassen_gdf, selected_road)
    except Exception:
        filtered = gpd.GeoDataFrame(geometry=[], crs="EPSG:28992")

    filtered = _safe_to_rd(filtered)
    if filtered.empty:
        return [], [f"Geen iASSET-wegas gevonden voor {clean_display_value(selected_road)}."]

    axes: list[dict[str, Any]] = []
    skipped = 0
    for index, row in filtered.iterrows():
        geometry = _line_like_geometry(row.geometry)
        if geometry is None:
            skipped += 1
            continue

        try:
            length_m = float(geometry.length)
        except Exception:
            skipped += 1
            continue

        if not math.isfinite(length_m) or length_m <= 0:
            skipped += 1
            continue

        axis_id = _axis_identifier(row, index)
        axes.append(
            {
                "axis_id": axis_id,
                "axis_naam": _axis_name(row, axis_id),
                "Wegnummer": clean_display_value(row.get("Wegnummer", selected_road)) or clean_display_value(selected_road),
                "geometry": geometry,
                "axis_length_m": length_m,
            }
        )

    if skipped:
        warnings.append(f"{skipped} iASSET-wegasrij(en) overgeslagen door lege of ongeldige geometrie.")

    if not axes:
        warnings.append("Geen bruikbare iASSET-wegasgeometrie beschikbaar.")

    return axes, warnings


def _first_existing_column(df: pd.DataFrame | gpd.GeoDataFrame | None, candidates: Iterable[str]) -> str | None:
    """Geef de eerste bestaande kolomnaam uit een kandidaatlijst terug."""
    if df is None:
        return None
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _parse_nwb_hectometer_to_km(value: Any) -> float | None:
    """
    Parseer een NWB-hectometerwaarde naar kilometers.

    NWB levert hectometrering vaak als 115 voor hm 11,5. Waarden met decimalen
    zoals 11.5 laten we staan. Ongeldige, negatieve of lege waarden leveren
    ``None`` op.
    """
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    text = clean_display_value(value).replace(",", ".")
    if not text:
        return None

    number = pd.to_numeric(text, errors="coerce")
    if pd.isna(number):
        return None

    try:
        numeric = float(number)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(numeric) or numeric < 0:
        return None

    if numeric >= 100 or float(numeric).is_integer():
        return numeric / 10.0

    return numeric


def _nearest_axis_projection(point: Point, axes: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float | None, float | None]:
    """Projecteer een punt op de dichtstbijzijnde iASSET-wegas."""
    best_axis: dict[str, Any] | None = None
    best_route: float | None = None
    best_distance: float | None = None

    for axis in axes:
        geometry = axis["geometry"]
        try:
            route_m = float(geometry.project(point))
            projected = geometry.interpolate(route_m)
            distance_m = float(point.distance(projected))
        except Exception:
            continue

        if best_distance is None or distance_m < best_distance:
            best_axis = axis
            best_route = route_m
            best_distance = distance_m

    return best_axis, best_route, best_distance


def _is_monotonic(values: list[float]) -> bool:
    """Controleer of een reeks oplopend of aflopend is."""
    if len(values) < 2:
        return False
    increasing = all(right >= left for left, right in zip(values, values[1:], strict=False))
    decreasing = all(right <= left for left, right in zip(values, values[1:], strict=False))
    return increasing or decreasing


def _aggregate_anchor_points(raw_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Vat geprojecteerde hectopunten samen tot één ijkpunt per hm per as.

    Meerdere NWB-punten met dezelfde hectometrering kunnen links/rechts of op
    aansluitende wegvakken liggen. Door de routeposities te middelen/medianen
    dempen we kleine bronverschillen zonder de iASSET-as aan te passen.
    """
    if not raw_rows:
        return _empty_anchor_frame()

    raw = pd.DataFrame(raw_rows)
    raw = raw[raw["status"] == "ijkpunt"].copy()
    if raw.empty:
        return _empty_anchor_frame()

    raw["_hm_bucket"] = pd.to_numeric(raw["hm_km"], errors="coerce").round(3)
    raw["route_m"] = pd.to_numeric(raw["route_m"], errors="coerce")
    raw["afstand_tot_iasset_wegas_m"] = pd.to_numeric(raw["afstand_tot_iasset_wegas_m"], errors="coerce")
    raw = raw.dropna(subset=["_hm_bucket", "route_m"])
    if raw.empty:
        return _empty_anchor_frame()

    grouped_rows: list[dict[str, Any]] = []
    for (axis_id, hm_km), group in raw.groupby(["axis_id", "_hm_bucket"], dropna=False, sort=True):
        route_m = float(group["route_m"].median())
        x_rd = float(pd.to_numeric(group["x_rd"], errors="coerce").mean())
        y_rd = float(pd.to_numeric(group["y_rd"], errors="coerce").mean())
        distance_m = float(group["afstand_tot_iasset_wegas_m"].max())
        first = group.iloc[0]
        grouped_rows.append(
            {
                "axis_id": clean_display_value(axis_id),
                "axis_naam": clean_display_value(first.get("axis_naam", "")),
                "Wegnummer": clean_display_value(first.get("Wegnummer", "")),
                "hm_km": round(float(hm_km), 3),
                "route_m": round(route_m, 2),
                "x_rd": round(x_rd, 3),
                "y_rd": round(y_rd, 3),
                "afstand_tot_iasset_wegas_m": round(distance_m, 2),
                "point_count": int(len(group)),
                "hm_monotoon": True,
                "status": "ijkpunt",
                "waarschuwing": "",
                "bronkwaliteit": "experimenteel",
            }
        )

    anchors = pd.DataFrame(grouped_rows, columns=_empty_anchor_frame().columns)
    if anchors.empty:
        return _empty_anchor_frame()

    # Per as controleren of hectometrering langs de route monotoon loopt.
    checked_rows: list[pd.DataFrame] = []
    for axis_id, group in anchors.groupby("axis_id", dropna=False, sort=False):
        group = group.sort_values("route_m").reset_index(drop=True)
        hm_values = pd.to_numeric(group["hm_km"], errors="coerce").dropna().astype(float).tolist()
        monotonic = _is_monotonic(hm_values)
        group["hm_monotoon"] = bool(monotonic)
        if not monotonic:
            group["status"] = "controleer"
            group["waarschuwing"] = (
                "Hectometrering loopt niet monotoon langs deze iASSET-wegas; "
                "projectgrenzen op hm kunnen hier niet betrouwbaar worden geïnterpoleerd."
            )
        checked_rows.append(group)

    result = pd.concat(checked_rows, ignore_index=True) if checked_rows else _empty_anchor_frame()
    return result.sort_values(["axis_id", "route_m", "hm_km"]).reset_index(drop=True)


def _build_axis_anchors(
    axes: list[dict[str, Any]],
    hectopoints_gdf: gpd.GeoDataFrame | None,
    *,
    max_anchor_distance_m: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Projecteer NWB-hectopunten op de iASSET-wegassen en maak ijkpunten."""
    warnings: list[str] = []
    if not axes:
        return _empty_anchor_frame(), warnings

    hectopoints = _safe_to_rd(hectopoints_gdf)
    if hectopoints.empty:
        return _empty_anchor_frame(), ["Geen NWB-hectopunten met geometrie beschikbaar voor ijking."]

    hm_column = _first_existing_column(hectopoints, HECTOMETER_COLUMN_CANDIDATES)
    if not hm_column:
        return _empty_anchor_frame(), ["Geen herkenbare hectometerkolom gevonden in de NWB-hectopunten."]

    raw_rows: list[dict[str, Any]] = []
    rejected_distance = 0
    rejected_hm = 0
    rejected_geometry = 0

    for _, row in hectopoints.iterrows():
        geometry = row.geometry
        if not _is_valid_geometry(geometry):
            rejected_geometry += 1
            continue

        hm_km = _parse_nwb_hectometer_to_km(row.get(hm_column))
        if hm_km is None:
            rejected_hm += 1
            continue

        try:
            point = geometry.centroid
        except Exception:
            rejected_geometry += 1
            continue

        axis, route_m, distance_m = _nearest_axis_projection(point, axes)
        if axis is None or route_m is None or distance_m is None:
            rejected_geometry += 1
            continue

        if distance_m > float(max_anchor_distance_m):
            rejected_distance += 1
            continue

        raw_rows.append(
            {
                "axis_id": axis["axis_id"],
                "axis_naam": axis["axis_naam"],
                "Wegnummer": axis["Wegnummer"],
                "hm_km": float(hm_km),
                "route_m": float(route_m),
                "x_rd": float(point.x),
                "y_rd": float(point.y),
                "afstand_tot_iasset_wegas_m": float(distance_m),
                "status": "ijkpunt",
            }
        )

    if rejected_geometry:
        warnings.append(f"{rejected_geometry} NWB-hectopunt(en) overgeslagen door lege of ongeldige geometrie.")
    if rejected_hm:
        warnings.append(f"{rejected_hm} NWB-hectopunt(en) overgeslagen door ongeldige hectometrering.")
    if rejected_distance:
        warnings.append(
            f"{rejected_distance} NWB-hectopunt(en) lagen verder dan {max_anchor_distance_m:g} m van de iASSET-wegas."
        )

    anchors = _aggregate_anchor_points(raw_rows)
    if anchors.empty:
        warnings.append("Geen bruikbare ijkpunten over na projectie op de iASSET-wegas.")

    return anchors, warnings


def _anchors_for_axis(anchors: pd.DataFrame, axis_id: str) -> pd.DataFrame:
    """Geef de ijkpunten voor één as, gesorteerd op routepositie."""
    if anchors is None or anchors.empty:
        return _empty_anchor_frame()
    work = anchors[anchors["axis_id"].map(clean_display_value) == clean_display_value(axis_id)].copy()
    if work.empty:
        return _empty_anchor_frame()
    work["route_m"] = pd.to_numeric(work["route_m"], errors="coerce")
    work["hm_km"] = pd.to_numeric(work["hm_km"], errors="coerce")
    return work.dropna(subset=["route_m", "hm_km"]).sort_values("route_m").reset_index(drop=True)


def _axis_is_calibrated(axis_anchors: pd.DataFrame) -> bool:
    """Controleer of een as minimaal twee monotone ijkpunten heeft."""
    if axis_anchors is None or len(axis_anchors) < 2:
        return False
    if "hm_monotoon" in axis_anchors.columns and not bool(axis_anchors["hm_monotoon"].fillna(False).all()):
        return False
    return True


def _route_to_km(route_m: Any, axis_anchors: pd.DataFrame) -> tuple[float | None, bool]:
    """
    Vertaal routepositie op iASSET-as naar geijkte hectometrering.

    Buiten de ijkpuntrange klemmen we naar het dichtstbijzijnde ijkpunt en
    markeren we ``in_range=False``. Zo blijft de diagnose zichtbaar, maar krijgt
    de gebruiker een waarschuwing.
    """
    if axis_anchors is None or len(axis_anchors) < 2:
        return None, False

    try:
        route_value = float(route_m)
    except (TypeError, ValueError, OverflowError):
        return None, False

    if not math.isfinite(route_value):
        return None, False

    route_values = pd.to_numeric(axis_anchors["route_m"], errors="coerce").astype(float).tolist()
    hm_values = pd.to_numeric(axis_anchors["hm_km"], errors="coerce").astype(float).tolist()

    if route_value <= route_values[0]:
        return hm_values[0], route_value == route_values[0]
    if route_value >= route_values[-1]:
        return hm_values[-1], route_value == route_values[-1]

    for index in range(1, len(route_values)):
        left_route = route_values[index - 1]
        right_route = route_values[index]
        if route_value > right_route:
            continue

        span = right_route - left_route
        if span <= 0:
            return hm_values[index - 1], False

        fraction = (route_value - left_route) / span
        value = hm_values[index - 1] + fraction * (hm_values[index] - hm_values[index - 1])
        return value, True

    return hm_values[-1], False


def _km_to_route(hm_km: Any, axis_anchors: pd.DataFrame) -> tuple[float | None, bool]:
    """
    Vertaal een project-hectometrering naar routepositie op de iASSET-as.

    Dit kan alleen wanneer de hm-waarden op de as monotoon zijn. Bij niet-
    monotone ijking geven we geen route terug, omdat een grens dan op meerdere
    plekken kan liggen.
    """
    if not _axis_is_calibrated(axis_anchors):
        return None, False

    try:
        hm_value = float(hm_km)
    except (TypeError, ValueError, OverflowError):
        return None, False

    if not math.isfinite(hm_value):
        return None, False

    pairs = [
        (float(row["hm_km"]), float(row["route_m"]))
        for _, row in axis_anchors.iterrows()
        if pd.notna(row.get("hm_km")) and pd.notna(row.get("route_m"))
    ]
    if len(pairs) < 2:
        return None, False

    pairs = sorted(pairs, key=lambda item: item[0])
    hm_values = [pair[0] for pair in pairs]
    route_values = [pair[1] for pair in pairs]

    if hm_value <= hm_values[0]:
        return route_values[0], hm_value == hm_values[0]
    if hm_value >= hm_values[-1]:
        return route_values[-1], hm_value == hm_values[-1]

    for index in range(1, len(hm_values)):
        left_hm = hm_values[index - 1]
        right_hm = hm_values[index]
        if hm_value > right_hm:
            continue

        span = right_hm - left_hm
        if span <= 0:
            return route_values[index - 1], False

        fraction = (hm_value - left_hm) / span
        value = route_values[index - 1] + fraction * (route_values[index] - route_values[index - 1])
        return value, True

    return route_values[-1], False


def _round_or_none(value: Any, digits: int = 3) -> float | None:
    """Rond een getal veilig af of geef ``None`` terug."""
    try:
        if value is None or pd.isna(value):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        return round(number, digits)
    except (TypeError, ValueError, OverflowError):
        return None


def _project_type(project_name: Any) -> str:
    """Haal de projecttypecode uit een onderhoudsprojectnaam, bijvoorbeeld HRB."""
    text = clean_display_value(project_name).upper().replace(",", ".")
    match = re.search(r"\bN\s*\d{3,4}\s*[-_/ ]+\s*([A-Z0-9]+)\s*[-_/ ]+", text)
    return match.group(1) if match else ""


def _project_road(project_name: Any) -> str:
    """Haal het N-wegnummer uit een onderhoudsprojectnaam."""
    text = clean_display_value(project_name).upper()
    match = re.search(r"\bN\s*(\d{3,4})\b", text)
    return f"N{match.group(1)}" if match else ""


def _subtheme_project_family(subthema: Any) -> str:
    """
    Vertaal een primair iASSET-subthema naar de projectfamilie.

    Deze mapping gebruiken we alleen diagnostisch, bijvoorbeeld om te bepalen
    of een gat tussen twee projectnamen ook echt samenvalt met fysieke primaire
    objecten van dat spoor.
    """
    value = normalize_text(subthema)
    mapping = {
        "rijstrook": "HRB",
        "parallelweg": "PW",
        "fietspad": "FP",
        "busbaan": "BB",
        "landbouwpad": "LBP",
    }
    return mapping.get(value, "")


def _situering_code(value: Any) -> str:
    """
    Maak van iASSET-situering een compacte L/R/LR-code.

    iASSET-velden kunnen waarden bevatten zoals 'rechts', 'links buiten' of
    combinaties. We gebruiken dit alleen als hulpmiddel voor diagnose; onbekende
    waarden blijven leeg in plaats van dat de tool crasht.
    """
    text = normalize_text(value)
    if not text:
        return ""
    has_left = "links" in text or text in {"l", "li"}
    has_right = "rechts" in text or text in {"r", "re"}
    if has_left and has_right:
        return "LR"
    if has_left:
        return "L"
    if has_right:
        return "R"
    upper = clean_display_value(value).upper().replace(" ", "")
    if upper in {"L", "R", "LR"}:
        return upper
    return ""


def _object_project_type_from_row(row: pd.Series, project_name: str = "") -> tuple[str, str, str]:
    """
    Bepaal projecttype/familie/situering voor een objectrij.

    Als een object een onderhoudsprojectnaam heeft, gebruiken we die. Voor
    primaire objecten zonder onderhoudsproject vallen we terug op subthema +
    Situering, zodat de gatcontrole kan zien of er fysiek een HRB/PWR/FPR-spoor
    in het gat aanwezig is.
    """
    project_type = _project_type(project_name)
    family, situering = _split_project_type(project_type)
    if family:
        return project_type, family, situering

    family = _subtheme_project_family(row.get("subthema", ""))
    situering = _situering_code(row.get("Situering", ""))
    if not family:
        return "", "", ""

    if family == "HRB":
        project_type = f"{family}{situering}" if situering in {"L", "R"} else family
    elif situering in {"L", "R", "LR"}:
        project_type = f"{family}{situering}"
    else:
        project_type = family

    return project_type, family, situering


def _worst_status(*statuses: str) -> str:
    """Geef de zwaarste status terug volgens de projectas-diagnose."""
    clean = [clean_display_value(status).lower() for status in statuses if clean_display_value(status)]
    if not clean:
        return "ok"
    return sorted(clean, key=lambda status: STATUS_RANK.get(status, 0), reverse=True)[0]


def _split_project_type(project_type: Any) -> tuple[str, str]:
    """
    Splits een projecttype in familie en situering.

    Voorbeelden:
    - HRB -> (HRB, "")
    - HRBR -> (HRB, R)
    - PWL -> (PW, L)
    - LBPL -> (LBP, L)

    We doen dit centraal, zodat dekking en overlap niet per ongeluk HRB, PWR
    en FPR als één spoor behandelen.
    """
    code = clean_display_value(project_type).upper()
    for family in PROJECT_TYPE_FAMILIES:
        if code == family:
            return family, ""
        if code.startswith(family):
            return family, code[len(family):]
    return code, ""


def _parse_decimal_label(label: str) -> float | None:
    """Parseer een hm-label uit de projectnaam, zonder begin/einde te sorteren."""
    try:
        value = float(clean_display_value(label).replace(",", "."))
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _validate_project_name(project_name: Any, selected_road: str) -> dict[str, Any]:
    """
    Valideer de onderhoudsprojectnaam volgens de beheerregels.

    De parser is bewust strenger dan ``parse_project_range``. Die bestaande
    functie is tolerant voor oude varianten, terwijl deze diagnose juist moet
    laten zien of de naamvorm betrouwbaar genoeg is voor projectgrenscontrole.
    """
    raw = clean_display_value(project_name)
    text = raw.upper().replace(",", ".")
    result: dict[str, Any] = {
        "naam_wegnummer": "",
        "project_type": "",
        "project_family": "",
        "situering": "",
        "naam_begin_label": "",
        "naam_eind_label": "",
        "begin_km": None,
        "end_km": None,
        "status": "ok",
        "melding": "",
    }

    match = re.match(
        r"^\s*N\s*(?P<road>\d{3,4})\s*[-_/ ]+\s*(?P<type>[A-Z0-9]+)\s*[-_/ ]+"
        r"(?P<start>\d+(?:\.\d+)?)\s*[-_/ ]+(?P<end>\d+(?:\.\d+)?)\s*$",
        text,
    )
    if not match:
        result["status"] = "controleer"
        result["melding"] = "projectnaam heeft geen herkenbaar patroon Nxxx-type-begin-einde"
        return result

    road = f"N{match.group('road')}"
    project_type = match.group("type")
    family, situering = _split_project_type(project_type)
    start_label = match.group("start")
    end_label = match.group("end")
    start_km = _parse_decimal_label(start_label)
    end_km = _parse_decimal_label(end_label)

    result.update(
        {
            "naam_wegnummer": road,
            "project_type": project_type,
            "project_family": family,
            "situering": situering,
            "naam_begin_label": start_label,
            "naam_eind_label": end_label,
            "begin_km": start_km,
            "end_km": end_km,
        }
    )

    warnings: list[tuple[str, str]] = []
    selected = clean_display_value(selected_road).upper().replace(" ", "")
    if selected and road != selected:
        warnings.append(("controleer", f"wegnummer in projectnaam ({road}) wijkt af van selectie ({selected})"))

    strict = re.match(
        r"^N\d{3,4}-[A-Z0-9]+-\d{2,3}\.\d-\d{2,3}\.\d$",
        text,
    )
    if not strict:
        warnings.append(
            (
                "controleer",
                "naamvorm moet Nxxx-type-begin-einde zijn met koppeltekens en exact één cijfer na de punt",
            )
        )

    for label_name, label, value in (("begin", start_label, start_km), ("einde", end_label, end_km)):
        if value is None:
            warnings.append(("controleer", f"{label_name}metrering is niet numeriek"))
            continue
        if value < 10 and not re.match(r"^\d{2}\.\d$", label):
            warnings.append(("controleer", f"{label_name}metrering onder 10 moet een voorloopnul hebben"))
        if not re.match(r"^\d{2,3}\.\d$", label):
            warnings.append(("controleer", f"{label_name}metrering moet exact één cijfer na de punt hebben"))

    if start_km is not None and end_km is not None and start_km >= end_km:
        warnings.append(("controleer", "beginmetrering is groter dan of gelijk aan eindmetrering"))

    if family not in PROJECT_TYPE_FAMILIES:
        warnings.append(("controleer", f"projecttype {project_type} is niet herkend als primaire projectfamilie"))
    elif family in PROJECT_TYPES_WITH_REQUIRED_SITUERING and situering not in ALLOWED_REQUIRED_SITUERING_CODES:
        warnings.append(("aandacht", f"projecttype {family} heeft normaal een situering L, R of LR nodig"))
    elif family == "HRB" and situering not in {"", "L", "R"}:
        warnings.append(("aandacht", f"situering {situering} bij HRB is niet standaard"))

    if warnings:
        result["status"] = _worst_status(*(status for status, _ in warnings))
        result["melding"] = "; ".join(message for _, message in warnings)

    return result


def _format_hm_label(value_km: float | Decimal) -> str:
    """
    Format een hectometerwaarde als beheerlabel ``xx.x``.

    Deze helper gebruikt geen ``round()`` om binaire float-ruis te vermijden.
    """
    value = Decimal(str(float(value_km))).quantize(Decimal("0.1"))
    value_float = float(value)
    return f"{value_float:04.1f}" if value_float < 10 else f"{value_float:.1f}"


def _snap_name_rule_details(value_km: Any, snap_tolerance_m: float = DEFAULT_BOUNDARY_SNAP_TOLERANCE_M) -> dict[str, Any]:
    """
    Bepaal het projectnaamlabel voor een fysieke grens met snap-tolerantie.

    Beheerregel v0.34.3:
    - ligt een fysieke grens binnen ``snap_tolerance_m`` van een hectometerpunt,
      dan gebruiken we dat hectometerpunt;
    - ligt de grens daarbuiten, dan geldt de bestaande naar-boven-regel.

    Waarom?
    Objectgeometrie zal vrijwel nooit exact op hetzelfde coördinaat liggen als
    een hectometerpunt. Zonder snap-tolerantie zou bijvoorbeeld 12.301 altijd
    12.4 worden, terwijl dit in de praktijk gewoon de grens op hm 12.3 kan zijn.
    """
    empty = {
        "label": "",
        "nearest_hm_km": None,
        "snap_distance_m": None,
        "snapped": False,
        "effective_km": None,
    }
    try:
        if value_km is None or pd.isna(value_km):
            return empty
        value = Decimal(str(float(value_km)))
        tolerance = max(Decimal("0"), Decimal(str(float(snap_tolerance_m)))) / Decimal("1000")
    except (TypeError, ValueError, OverflowError):
        return empty

    if not math.isfinite(float(value)) or value < 0:
        return empty

    nearest_hm_index = (value * Decimal("10")).to_integral_value(rounding=ROUND_HALF_UP)
    nearest_hm = nearest_hm_index / Decimal("10")
    snap_distance_m = abs(value - nearest_hm) * Decimal("1000")
    snapped = snap_distance_m <= tolerance * Decimal("1000")

    if snapped:
        effective_km = nearest_hm
        label = _format_hm_label(effective_km)
    else:
        effective_km = (value * Decimal("10")).to_integral_value(rounding=ROUND_CEILING) / Decimal("10")
        label = _format_hm_label(effective_km)

    return {
        "label": label,
        "nearest_hm_km": float(nearest_hm),
        "snap_distance_m": float(snap_distance_m),
        "snapped": bool(snapped),
        "effective_km": float(effective_km),
    }


def _format_name_rule_or_empty(
    value_km: Any,
    snap_tolerance_m: float = DEFAULT_BOUNDARY_SNAP_TOLERANCE_M,
) -> str:
    """
    Format een fysieke km-waarde volgens snap-tolerantie + onderhoudsnaamregel.

    Eerst snappen we naar een nabij hectometerpunt. Alleen wanneer de grens
    verder weg ligt dan de snap-tolerantie, wordt naar boven afgerond.
    """
    return clean_display_value(_snap_name_rule_details(value_km, snap_tolerance_m).get("label"))


def _namezone_km_range(
    label_km: Any,
    snap_tolerance_m: float = DEFAULT_BOUNDARY_SNAP_TOLERANCE_M,
) -> tuple[float | None, float | None]:
    """
    Geef de fysieke km-zone terug die bij één geschreven hm-label past.

    Voor label 34.8 en snap-tolerantie 2,5 m hoort grofweg:
    - vanaf net na de snapzone rond 34.7: 34.7025 km;
    - tot en met de snapzone rond 34.8: 34.8025 km.

    Dit gebruiken we niet om iASSET te muteren, maar om gatmeldingen niet te
    baseren op een te exacte interpretatie van projectnamen.
    """
    try:
        value = Decimal(str(float(label_km)))
        tolerance_km = max(Decimal("0"), Decimal(str(float(snap_tolerance_m)))) / Decimal("1000")
    except (TypeError, ValueError, OverflowError):
        return None, None

    if not math.isfinite(float(value)) or value < 0:
        return None, None

    # Projectnamen hebben één cijfer achter de punt; normaliseer daarom eerst
    # naar het bijbehorende hectometerpunt.
    label_hm = (value * Decimal("10")).to_integral_value(rounding=ROUND_HALF_UP) / Decimal("10")
    lower = label_hm - Decimal("0.1") + tolerance_km
    upper = label_hm + tolerance_km
    if lower < 0:
        lower = Decimal("0")
    return float(lower), float(upper)


def _namezone_route_range(
    label_km: Any,
    axis_anchors: pd.DataFrame,
    snap_tolerance_m: float = DEFAULT_BOUNDARY_SNAP_TOLERANCE_M,
) -> tuple[float | None, float | None]:
    """Vertaal de naamzone rond een geschreven hm-label naar route-meters."""
    lower_km, upper_km = _namezone_km_range(label_km, snap_tolerance_m)
    lower_route = upper_route = None
    if lower_km is not None:
        lower_route, _ = _km_to_route(lower_km, axis_anchors)
    if upper_km is not None:
        upper_route, _ = _km_to_route(upper_km, axis_anchors)
    if lower_route is None or upper_route is None:
        return None, None
    return min(float(lower_route), float(upper_route)), max(float(lower_route), float(upper_route))



def _selected_project_names(road_gdf: gpd.GeoDataFrame, selected_road: str) -> list[str]:
    """Verzamel unieke onderhoudsprojectnamen voor de geselecteerde weg."""
    if road_gdf is None or road_gdf.empty or "Onderhoudsproject" not in road_gdf.columns:
        return []

    selected = clean_display_value(selected_road).upper().replace(" ", "")
    names: set[str] = set()
    for value in road_gdf["Onderhoudsproject"].tolist():
        name = clean_display_value(value)
        if not name:
            continue

        project_road = _project_road(name)
        if project_road and project_road != selected:
            continue

        if parse_project_range(name) is None:
            # Niet-parseerbare namen blijven uit de berekening, maar worden
            # in de objectdiagnose nog zichtbaar als ze objecten bevatten.
            continue

        names.add(name)

    return sorted(names)


def _zone_severity(color: str) -> int:
    """Zet zonekleur om naar een rangorde."""
    color_norm = normalize_text(color)
    if color_norm == "rood":
        return 2
    if color_norm == "oranje":
        return 1
    return 0


def _find_boundary_zone(
    zones_df: pd.DataFrame | None,
    *,
    axis_id: str,
    route_m: float | None,
    boundary_zone_buffer_m: float,
) -> tuple[str, str]:
    """
    Controleer of een projectgrens in of vlak bij een afwijkingszone ligt.

    Veel zones zijn sample-gebaseerd en kunnen 0 meter lengte hebben. Daarom
    gebruiken we een kleine buffer, zodat een grens op 1-2 meter van zo'n
    samplepunt niet ten onrechte groen lijkt.
    """
    if zones_df is None or zones_df.empty or route_m is None:
        return "", ""

    required = {"afstand_van_m", "afstand_tot_m", "kleurklasse"}
    if not required.issubset(set(zones_df.columns)):
        return "", ""

    work = zones_df.copy()
    if "nummer" in work.columns:
        axis_norm = clean_display_value(axis_id)
        exact = work[work["nummer"].map(clean_display_value) == axis_norm]
        if not exact.empty:
            work = exact

    matches: list[tuple[int, str, str]] = []
    for _, row in work.iterrows():
        try:
            start_m = float(row.get("afstand_van_m"))
            end_m = float(row.get("afstand_tot_m"))
        except (TypeError, ValueError, OverflowError):
            continue

        left = min(start_m, end_m) - float(boundary_zone_buffer_m)
        right = max(start_m, end_m) + float(boundary_zone_buffer_m)
        if left <= float(route_m) <= right:
            color = clean_display_value(row.get("kleurklasse", ""))
            zone_id = clean_display_value(row.get("zone_id", ""))
            matches.append((_zone_severity(color), color, zone_id))

    if not matches:
        return "", ""

    matches = sorted(matches, key=lambda item: item[0], reverse=True)
    worst_color = matches[0][1]
    zone_ids = ", ".join(sorted({item[2] for item in matches if item[2]}))
    return worst_color, zone_ids


def _build_object_ranges(
    road_gdf: gpd.GeoDataFrame,
    axes: list[dict[str, Any]],
    anchors: pd.DataFrame,
    *,
    max_object_offset_m: float,
) -> pd.DataFrame:
    """Projecteer objectgeometrieën met onderhoudsproject op de dichtstbijzijnde iASSET-wegas."""
    if road_gdf is None or road_gdf.empty or not axes or "Onderhoudsproject" not in road_gdf.columns:
        return _empty_object_range_frame()

    working = _safe_to_rd(road_gdf)
    if working.empty:
        return _empty_object_range_frame()

    rows: list[dict[str, Any]] = []
    for object_index, row in working.iterrows():
        project_name = clean_display_value(row.get("Onderhoudsproject", ""))
        subthema_norm = normalize_text(row.get("subthema", ""))
        is_primary = subthema_norm in PRIMARY_SUBTHEMES

        # v0.34.3: primaire objecten zonder onderhoudsproject blijven zichtbaar
        # voor de gatcontrole. Secundaire objecten zonder projectnaam voegen we
        # niet toe, anders wordt de export te druk en minder bruikbaar.
        if not project_name and not is_primary:
            continue

        object_project_type, object_family, object_situering = _object_project_type_from_row(row, project_name)

        geometry = row.geometry if "geometry" in row.index else None
        if not _is_valid_geometry(geometry):
            rows.append(
                {
                    "sys_id": row.get("sys_id", object_index),
                    "nummer": clean_display_value(row.get("nummer", "")),
                    "subthema": clean_display_value(row.get("subthema", "")),
                    "Onderhoudsproject": project_name,
                    "project_type": object_project_type,
                    "project_family": object_family,
                    "situering": object_situering,
                    "axis_id": "",
                    "axis_naam": "",
                    "route_begin_m": None,
                    "route_eind_m": None,
                    "referentie_begin_km": None,
                    "referentie_eind_km": None,
                    "afstand_tot_as_m": None,
                    "primair_object": is_primary,
                    "status": "controleer",
                    "waarschuwing": "object heeft geen bruikbare geometrie",
                    "bronkwaliteit": "experimenteel",
                }
            )
            continue

        best: dict[str, Any] | None = None
        best_projection: tuple[float | None, float | None, float | None, float | None] | None = None

        for axis in axes:
            projection = project_geometry_range_on_axis(geometry, axis["geometry"])
            _, _, _, offset_m = projection
            if offset_m is None:
                continue
            if best is None or float(offset_m) < float(best_projection[3]):  # type: ignore[index]
                best = axis
                best_projection = projection

        if best is None or best_projection is None:
            continue

        route_start, _, route_end, offset_m = best_projection
        axis_anchors = _anchors_for_axis(anchors, best["axis_id"])
        begin_km, begin_in_range = _route_to_km(route_start, axis_anchors)
        end_km, end_in_range = _route_to_km(route_end, axis_anchors)

        warning_parts: list[str] = []
        if offset_m is not None and offset_m > max_object_offset_m:
            warning_parts.append(f"afstand tot iASSET-wegas > {max_object_offset_m:g} m")
        if not (begin_in_range and end_in_range):
            warning_parts.append("object valt deels buiten ijkbereik")

        status = "projectie" if not warning_parts else "controleer"

        rows.append(
            {
                "sys_id": row.get("sys_id", object_index),
                "nummer": clean_display_value(row.get("nummer", "")),
                "subthema": clean_display_value(row.get("subthema", "")),
                "Onderhoudsproject": project_name,
                "project_type": object_project_type,
                "project_family": object_family,
                "situering": object_situering,
                "axis_id": best["axis_id"],
                "axis_naam": best["axis_naam"],
                "route_begin_m": _round_or_none(route_start, 2),
                "route_eind_m": _round_or_none(route_end, 2),
                "referentie_begin_km": _round_or_none(min(begin_km, end_km) if begin_km is not None and end_km is not None else None, 3),
                "referentie_eind_km": _round_or_none(max(begin_km, end_km) if begin_km is not None and end_km is not None else None, 3),
                "afstand_tot_as_m": _round_or_none(offset_m, 2),
                "primair_object": is_primary,
                "status": status,
                "waarschuwing": "; ".join(warning_parts),
                "bronkwaliteit": "experimenteel",
            }
        )

    if not rows:
        return _empty_object_range_frame()

    return pd.DataFrame(rows, columns=_empty_object_range_frame().columns)


def _object_summary_for_project(object_ranges: pd.DataFrame, project_name: str, axis_id: str) -> dict[str, Any]:
    """Vat fysieke objectprojecties voor één project en as samen."""
    if object_ranges is None or object_ranges.empty:
        return {
            "objecten_in_project": 0,
            "objecten_op_axis": 0,
            "route_begin_m": None,
            "route_eind_m": None,
            "begin_km": None,
            "end_km": None,
            "length_m": None,
        }

    project_rows = object_ranges[object_ranges["Onderhoudsproject"].map(clean_display_value) == clean_display_value(project_name)].copy()
    axis_rows = project_rows[project_rows["axis_id"].map(clean_display_value) == clean_display_value(axis_id)].copy()

    usable = axis_rows[
        (axis_rows["status"] == "projectie")
        & (axis_rows.get("primair_object", pd.Series(True, index=axis_rows.index)).astype(bool))
    ].copy()

    route_starts = pd.to_numeric(usable.get("route_begin_m", pd.Series(dtype=float)), errors="coerce").dropna()
    route_ends = pd.to_numeric(usable.get("route_eind_m", pd.Series(dtype=float)), errors="coerce").dropna()
    begin_values = pd.to_numeric(usable.get("referentie_begin_km", pd.Series(dtype=float)), errors="coerce").dropna()
    end_values = pd.to_numeric(usable.get("referentie_eind_km", pd.Series(dtype=float)), errors="coerce").dropna()

    if route_starts.empty or route_ends.empty:
        return {
            "objecten_in_project": int(len(project_rows)),
            "objecten_op_axis": int(len(axis_rows)),
            "route_begin_m": None,
            "route_eind_m": None,
            "begin_km": None,
            "end_km": None,
            "length_m": None,
        }

    route_begin = float(min(route_starts.min(), route_ends.min()))
    route_end = float(max(route_starts.max(), route_ends.max()))

    begin_km = float(begin_values.min()) if not begin_values.empty else None
    end_km = float(end_values.max()) if not end_values.empty else None
    length_m = abs(route_end - route_begin)

    return {
        "objecten_in_project": int(len(project_rows)),
        "objecten_op_axis": int(len(axis_rows)),
        "route_begin_m": route_begin,
        "route_eind_m": route_end,
        "begin_km": begin_km,
        "end_km": end_km,
        "length_m": length_m,
    }


def _choose_project_axis(
    project_name: str,
    project_start_km: float,
    project_end_km: float,
    project_length_m: float,
    axes: list[dict[str, Any]],
    anchors: pd.DataFrame,
    object_ranges: pd.DataFrame,
) -> dict[str, Any] | None:
    """Kies de meest waarschijnlijke iASSET-as voor een projectrange."""
    candidates: list[dict[str, Any]] = []

    for axis in axes:
        axis_anchors = _anchors_for_axis(anchors, axis["axis_id"])
        if len(axis_anchors) < 2:
            continue

        route_start, start_in_range = _km_to_route(project_start_km, axis_anchors)
        route_end, end_in_range = _km_to_route(project_end_km, axis_anchors)
        if route_start is None or route_end is None:
            continue

        as_length_m = abs(float(route_end) - float(route_start))
        object_summary = _object_summary_for_project(object_ranges, project_name, axis["axis_id"])
        object_count = int(object_summary["objecten_op_axis"])
        length_delta = abs(as_length_m - float(project_length_m))

        score = length_delta
        if not (start_in_range and end_in_range):
            score += 100000.0
        # Een project met objecten op deze as is waarschijnlijker dan een
        # parallel of alternatief wegasdeel zonder objecten.
        score -= min(object_count, 25) * 1000.0

        candidates.append(
            {
                "axis": axis,
                "axis_anchors": axis_anchors,
                "route_start": route_start,
                "route_end": route_end,
                "start_in_range": start_in_range,
                "end_in_range": end_in_range,
                "as_length_m": as_length_m,
                "object_summary": object_summary,
                "score": score,
            }
        )

    if not candidates:
        return None

    return sorted(candidates, key=lambda item: item["score"])[0]


def _build_project_boundaries(
    road_gdf: gpd.GeoDataFrame,
    axes: list[dict[str, Any]],
    anchors: pd.DataFrame,
    object_ranges: pd.DataFrame,
    deviation_zones: pd.DataFrame | None,
    selected_road: str,
    *,
    boundary_zone_buffer_m: float,
    length_tolerance_m: float,
    boundary_snap_tolerance_m: float,
) -> pd.DataFrame:
    """Bouw de diagnose per onderhoudsprojectgrens."""
    project_names = _selected_project_names(road_gdf, selected_road)
    if not project_names:
        return _empty_project_boundary_frame()

    rows: list[dict[str, Any]] = []
    for project_name in project_names:
        name_check = _validate_project_name(project_name, selected_road)
        project_range = parse_project_range(project_name)
        if project_range is None:
            continue

        project_start_km = float(project_range.start_km)
        project_end_km = float(project_range.end_km)
        project_length_m = float(project_range.length_m)
        project_type = clean_display_value(name_check.get("project_type")) or _project_type(project_name)
        project_family = clean_display_value(name_check.get("project_family"))
        situering = clean_display_value(name_check.get("situering"))

        base_row = {
            "Onderhoudsproject": project_name,
            "naam_wegnummer": clean_display_value(name_check.get("naam_wegnummer")),
            "project_type": project_type,
            "project_family": project_family,
            "situering": situering,
            "naam_begin_label": clean_display_value(name_check.get("naam_begin_label")),
            "naam_eind_label": clean_display_value(name_check.get("naam_eind_label")),
            "naam_validatie_status": clean_display_value(name_check.get("status")) or "ok",
            "naam_validatie_melding": clean_display_value(name_check.get("melding")),
            "project_begin_km": _round_or_none(project_start_km, 3),
            "project_eind_km": _round_or_none(project_end_km, 3),
            "project_lengte_m": _round_or_none(project_length_m, 1),
            "status_projectnaam": clean_display_value(name_check.get("status")) or "ok",
        }

        chosen = _choose_project_axis(
            project_name,
            project_start_km,
            project_end_km,
            project_length_m,
            axes,
            anchors,
            object_ranges,
        )

        if chosen is None:
            final_status = _worst_status(str(base_row["status_projectnaam"]), "controleer")
            rows.append(
                {
                    **base_row,
                    "axis_id": "",
                    "axis_naam": "",
                    "objecten_in_project": int((road_gdf["Onderhoudsproject"].map(clean_display_value) == project_name).sum()) if "Onderhoudsproject" in road_gdf.columns else 0,
                    "objecten_op_axis": 0,
                    "as_begin_m": None,
                    "as_eind_m": None,
                    "as_lengte_m": None,
                    "lengteverschil_naam_vs_as_m": None,
                    "begin_binnen_ijking": False,
                    "eind_binnen_ijking": False,
                    "begin_buiten_ijkbereik": True,
                    "eind_buiten_ijkbereik": True,
                    "begin_zone_kleur": "",
                    "begin_zone_id": "",
                    "eind_zone_kleur": "",
                    "eind_zone_id": "",
                    "fysiek_object_begin_km": None,
                    "fysiek_object_eind_km": None,
                    "fysiek_object_lengte_m": None,
                    "snap_tolerantie_m": _round_or_none(boundary_snap_tolerance_m, 2),
                    "object_begin_dichtstbijzijnde_hm": None,
                    "object_begin_snap_afstand_m": None,
                    "object_begin_gesnapt_naar_hm": False,
                    "object_begin_naamregel": "",
                    "object_eind_dichtstbijzijnde_hm": None,
                    "object_eind_snap_afstand_m": None,
                    "object_eind_gesnapt_naar_hm": False,
                    "object_eind_naamregel": "",
                    "verschil_projectnaam_vs_objectligging_m": None,
                    "objectligging_status": "niet_beschikbaar",
                    "objectligging_melding": "geen bruikbare objectprojectie",
                    "status_projectgrens": "controleer",
                    "status": final_status,
                    "advies": "Geen bruikbare ijking; controleer iASSET-wegas en NWB-hectopunten.",
                    "waarschuwing": "geen bruikbare ijking op iASSET-wegas",
                    "bronkwaliteit": "experimenteel",
                }
            )
            continue

        axis = chosen["axis"]
        route_start = float(chosen["route_start"])
        route_end = float(chosen["route_end"])
        as_length_m = float(chosen["as_length_m"])
        length_delta_m = as_length_m - project_length_m
        object_summary = chosen["object_summary"]

        begin_zone_color, begin_zone_id = _find_boundary_zone(
            deviation_zones,
            axis_id=axis["axis_id"],
            route_m=route_start,
            boundary_zone_buffer_m=boundary_zone_buffer_m,
        )
        end_zone_color, end_zone_id = _find_boundary_zone(
            deviation_zones,
            axis_id=axis["axis_id"],
            route_m=route_end,
            boundary_zone_buffer_m=boundary_zone_buffer_m,
        )

        physical_delta_m = None
        if object_summary["begin_km"] is not None and object_summary["end_km"] is not None:
            physical_delta_m = max(
                abs(float(object_summary["begin_km"]) - project_start_km),
                abs(float(object_summary["end_km"]) - project_end_km),
            ) * 1000.0

        object_begin_rule = _snap_name_rule_details(
            object_summary.get("begin_km"),
            snap_tolerance_m=float(boundary_snap_tolerance_m),
        )
        object_end_rule = _snap_name_rule_details(
            object_summary.get("end_km"),
            snap_tolerance_m=float(boundary_snap_tolerance_m),
        )
        object_begin_name_rule = clean_display_value(object_begin_rule.get("label"))
        object_end_name_rule = clean_display_value(object_end_rule.get("label"))

        boundary_warnings: list[str] = []
        if not chosen["start_in_range"]:
            boundary_warnings.append("begin buiten ijkbereik")
        if not chosen["end_in_range"]:
            boundary_warnings.append("eind buiten ijkbereik")
        if abs(length_delta_m) > float(length_tolerance_m):
            boundary_warnings.append(f"lengteverschil naam versus geijkte as > {length_tolerance_m:g} m")
        if begin_zone_color:
            boundary_warnings.append(f"begingrens in {begin_zone_color} afwijkingszone")
        if end_zone_color:
            boundary_warnings.append(f"eindgrens in {end_zone_color} afwijkingszone")

        object_warnings: list[str] = []
        object_status = "ok"
        if object_summary["objecten_in_project"] and not object_summary["objecten_op_axis"]:
            object_warnings.append("projectobjecten liggen niet op gekozen as")
            object_status = "controleer"
        if object_summary["objecten_in_project"] and object_summary["length_m"] is None:
            object_warnings.append("geen bruikbare primaire objectprojectie")
            object_status = _worst_status(object_status, "controleer")
        if physical_delta_m is not None and physical_delta_m > float(length_tolerance_m):
            object_warnings.append(f"objectligging wijkt > {length_tolerance_m:g} m af van projectnaam")
            object_status = _worst_status(object_status, "aandacht")

        if object_begin_name_rule and object_end_name_rule:
            name_start_label = clean_display_value(name_check.get("naam_begin_label"))
            name_end_label = clean_display_value(name_check.get("naam_eind_label"))
            if name_start_label and name_end_label and (
                object_begin_name_rule != name_start_label or object_end_name_rule != name_end_label
            ):
                object_warnings.append(
                    "fysieke objectligging geeft volgens naamregel "
                    f"{object_begin_name_rule}-{object_end_name_rule}"
                )
                object_status = _worst_status(object_status, "aandacht")

        worst_zone = max(_zone_severity(begin_zone_color), _zone_severity(end_zone_color))
        if worst_zone == 2 or any("buiten ijkbereik" in item for item in boundary_warnings):
            boundary_status = "controleer"
        elif boundary_warnings:
            boundary_status = "aandacht"
        else:
            boundary_status = "ok"

        # Objectligging blijft bewust buiten de eindstatus. Een fietspad of
        # parallelweg kan logisch ver van de hoofdas liggen. We tonen het daarom
        # apart, maar laten de projectgrensstatus niet overschreeuwen.
        final_status = _worst_status(str(base_row["status_projectnaam"]), boundary_status)

        if final_status == "ok" and object_status != "ok":
            advice = "Projectgrens en naamvorm zijn ok; controleer objectligging apart als context."
        elif final_status == "ok":
            advice = "Geen aandachtspunt gevonden. Gebruik dit alleen als diagnose, niet als automatische mutatie."
        elif worst_zone:
            advice = "Controleer projectgrens visueel op de NWB-afwijkingszone voordat je deze grens gebruikt."
        elif base_row["status_projectnaam"] != "ok":
            advice = "Controleer eerst de onderhoudsprojectnaam; pas niets automatisch aan."
        else:
            advice = "Controleer projectgrens en ijkpunten; pas niets automatisch aan."

        rows.append(
            {
                **base_row,
                "axis_id": axis["axis_id"],
                "axis_naam": axis["axis_naam"],
                "objecten_in_project": int(object_summary["objecten_in_project"]),
                "objecten_op_axis": int(object_summary["objecten_op_axis"]),
                "as_begin_m": _round_or_none(route_start, 2),
                "as_eind_m": _round_or_none(route_end, 2),
                "as_lengte_m": _round_or_none(as_length_m, 1),
                "lengteverschil_naam_vs_as_m": _round_or_none(length_delta_m, 1),
                "begin_binnen_ijking": bool(chosen["start_in_range"]),
                "eind_binnen_ijking": bool(chosen["end_in_range"]),
                "begin_buiten_ijkbereik": not bool(chosen["start_in_range"]),
                "eind_buiten_ijkbereik": not bool(chosen["end_in_range"]),
                "begin_zone_kleur": begin_zone_color,
                "begin_zone_id": begin_zone_id,
                "eind_zone_kleur": end_zone_color,
                "eind_zone_id": end_zone_id,
                "fysiek_object_begin_km": _round_or_none(object_summary["begin_km"], 3),
                "fysiek_object_eind_km": _round_or_none(object_summary["end_km"], 3),
                "fysiek_object_lengte_m": _round_or_none(object_summary["length_m"], 1),
                "snap_tolerantie_m": _round_or_none(boundary_snap_tolerance_m, 2),
                "object_begin_dichtstbijzijnde_hm": _round_or_none(object_begin_rule.get("nearest_hm_km"), 3),
                "object_begin_snap_afstand_m": _round_or_none(object_begin_rule.get("snap_distance_m"), 2),
                "object_begin_gesnapt_naar_hm": bool(object_begin_rule.get("snapped", False)),
                "object_begin_naamregel": object_begin_name_rule,
                "object_eind_dichtstbijzijnde_hm": _round_or_none(object_end_rule.get("nearest_hm_km"), 3),
                "object_eind_snap_afstand_m": _round_or_none(object_end_rule.get("snap_distance_m"), 2),
                "object_eind_gesnapt_naar_hm": bool(object_end_rule.get("snapped", False)),
                "object_eind_naamregel": object_end_name_rule,
                "verschil_projectnaam_vs_objectligging_m": _round_or_none(physical_delta_m, 1),
                "objectligging_status": object_status,
                "objectligging_melding": "; ".join(object_warnings),
                "status_projectgrens": boundary_status,
                "status": final_status,
                "advies": advice,
                "waarschuwing": "; ".join(
                    item for item in [base_row["naam_validatie_melding"], *boundary_warnings] if item
                ),
                "bronkwaliteit": "experimenteel",
            }
        )

    if not rows:
        return _empty_project_boundary_frame()

    return pd.DataFrame(rows, columns=_empty_project_boundary_frame().columns).sort_values(
        ["axis_id", "project_type", "as_begin_m", "Onderhoudsproject"],
        na_position="last",
    ).reset_index(drop=True)


def _project_type_matches_gap(object_type: str, gap_project_type: str) -> bool:
    """
    Bepaal of een objectspoor past bij het projecttype van een gat.

    Exacte matches zijn leidend. Een gecombineerde LR-situering mag ook matchen
    met L of R, omdat bijvoorbeeld BBLR voorlopig als toegestane gecombineerde
    beheernaam wordt beschouwd.
    """
    object_type = clean_display_value(object_type).upper()
    gap_project_type = clean_display_value(gap_project_type).upper()
    if not object_type or not gap_project_type:
        return False
    if object_type == gap_project_type:
        return True

    gap_family, gap_situering = _split_project_type(gap_project_type)
    object_family, object_situering = _split_project_type(object_type)
    if gap_family != object_family:
        return False

    if object_situering == "LR" and gap_situering in {"L", "R"}:
        return True
    if gap_situering == "LR" and object_situering in {"L", "R"}:
        return True

    # Bij HRB zonder L/R is situering vaak niet relevant.
    if gap_family == "HRB" and not gap_situering and object_family == "HRB":
        return True

    return False


def _primary_object_presence_in_gap(
    object_ranges: pd.DataFrame,
    *,
    axis_id: str,
    project_type: str,
    start_m: float,
    end_m: float,
) -> dict[str, Any]:
    """
    Controleer of er fysieke primaire objecten liggen binnen een projectgat.

    v0.34.1 meldde elk gat tussen twee projectnamen van hetzelfde type. In de
    praktijk is dat voor parallelwegen en fietspaden te streng: soms bestaat het
    spoor daar fysiek niet. Daarom markeren we een gat pas als controlepunt als
    er ook primaire objectprojecties van hetzelfde spoor in dat interval liggen.
    """
    if object_ranges is None or object_ranges.empty:
        return {"count": 0, "length_m": 0.0, "objects": ""}

    left = min(float(start_m), float(end_m))
    right = max(float(start_m), float(end_m))
    if not math.isfinite(left) or not math.isfinite(right) or left >= right:
        return {"count": 0, "length_m": 0.0, "objects": ""}

    work = object_ranges.copy()
    if "axis_id" not in work.columns:
        return {"count": 0, "length_m": 0.0, "objects": ""}

    work = work[work["axis_id"].map(clean_display_value) == clean_display_value(axis_id)].copy()
    if work.empty:
        return {"count": 0, "length_m": 0.0, "objects": ""}

    primary = work.get("primair_object", pd.Series(False, index=work.index)).astype(bool)
    work = work[primary].copy()
    if work.empty:
        return {"count": 0, "length_m": 0.0, "objects": ""}

    work = work[
        work.get("project_type", pd.Series("", index=work.index))
        .map(lambda value: _project_type_matches_gap(clean_display_value(value), project_type))
    ].copy()
    if work.empty:
        return {"count": 0, "length_m": 0.0, "objects": ""}

    starts = pd.to_numeric(work.get("route_begin_m", pd.Series(dtype=float)), errors="coerce")
    ends = pd.to_numeric(work.get("route_eind_m", pd.Series(dtype=float)), errors="coerce")
    work = work.assign(
        _start_m=pd.concat([starts, ends], axis=1).min(axis=1),
        _end_m=pd.concat([starts, ends], axis=1).max(axis=1),
    ).dropna(subset=["_start_m", "_end_m"])
    if work.empty:
        return {"count": 0, "length_m": 0.0, "objects": ""}

    work = work[(work["_start_m"] < right) & (work["_end_m"] > left)].copy()
    if work.empty:
        return {"count": 0, "length_m": 0.0, "objects": ""}

    overlap_lengths = (work["_end_m"].clip(upper=right) - work["_start_m"].clip(lower=left)).clip(lower=0)
    object_labels = []
    for _, row in work.head(10).iterrows():
        label = clean_display_value(row.get("nummer", "")) or clean_display_value(row.get("Onderhoudsproject", ""))
        if label:
            object_labels.append(label)

    return {
        "count": int(len(work)),
        "length_m": float(overlap_lengths.sum()),
        "objects": ", ".join(dict.fromkeys(object_labels)),
    }


def _merge_intervals_m(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Voeg overlappende route-intervallen samen."""
    clean = sorted((min(a, b), max(a, b)) for a, b in intervals if a is not None and b is not None and a != b)
    if not clean:
        return []

    merged = [clean[0]]
    for start, end in clean[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _build_project_coverage(
    project_boundaries: pd.DataFrame,
    anchors: pd.DataFrame,
    axes: list[dict[str, Any]],
    object_ranges: pd.DataFrame,
    *,
    gap_tolerance_m: float,
    boundary_snap_tolerance_m: float,
) -> pd.DataFrame:
    """
    Signaleer projectdekking, gaten en overlap per projecttype.

    v0.34.0 vergeleek alle projecttypen op één as met elkaar. Daardoor leek een
    HRB-project te overlappen met een parallelweg of fietspad. In v0.34.3 wordt
    per spoor gecontroleerd: HRB met HRB, PWR met PWR, FPR met FPR, enzovoort.
    We melden alleen interne gaten tussen opeenvolgende projecten van hetzelfde
    type; het ontbreken van een parallelweg aan het begin/einde van een N-weg is
    namelijk geen fout.
    """
    if project_boundaries is None or project_boundaries.empty:
        return _empty_coverage_frame()

    rows: list[dict[str, Any]] = []
    axis_names = {axis["axis_id"]: axis["axis_naam"] for axis in axes}

    group_columns = ["axis_id", "project_type"]
    for (axis_id_raw, project_type_raw), axis_projects in project_boundaries.groupby(group_columns, dropna=False, sort=True):
        axis_id = clean_display_value(axis_id_raw)
        project_type = clean_display_value(project_type_raw)
        if not axis_id or not project_type:
            continue

        first = axis_projects.iloc[0]
        project_family = clean_display_value(first.get("project_family", ""))
        situering = clean_display_value(first.get("situering", ""))

        axis_anchors = _anchors_for_axis(anchors, axis_id)
        if len(axis_anchors) >= 2:
            span_start = float(pd.to_numeric(axis_anchors["route_m"], errors="coerce").min())
            span_end = float(pd.to_numeric(axis_anchors["route_m"], errors="coerce").max())
            calibration_span = max(0.0, span_end - span_start)
        else:
            calibration_span = None

        intervals: list[dict[str, Any]] = []
        for _, row in axis_projects.iterrows():
            try:
                start = float(row.get("as_begin_m"))
                end = float(row.get("as_eind_m"))
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(start) or not math.isfinite(end) or start == end:
                continue
            intervals.append(
                {
                    "start_m": min(start, end),
                    "end_m": max(start, end),
                    "name": clean_display_value(row.get("Onderhoudsproject", "")),
                    "project_begin_km": row.get("project_begin_km"),
                    "project_eind_km": row.get("project_eind_km"),
                }
            )

        if not intervals:
            continue

        intervals = sorted(intervals, key=lambda item: (item["start_m"], item["end_m"], item["name"]))
        merged = _merge_intervals_m([(item["start_m"], item["end_m"]) for item in intervals])
        unique_length = sum(end - start for start, end in merged)
        project_span = max(item["end_m"] for item in intervals) - min(item["start_m"] for item in intervals)
        coverage_pct = unique_length / project_span * 100.0 if project_span > 0 else None

        common = {
            "axis_id": axis_id,
            "axis_naam": axis_names.get(axis_id, axis_id),
            "project_type": project_type,
            "project_family": project_family,
            "situering": situering,
            "bronkwaliteit": "experimenteel",
        }

        rows.append(
            {
                **common,
                "controle_type": "dekking",
                "van_m": _round_or_none(min(item["start_m"] for item in intervals), 2),
                "tot_m": _round_or_none(max(item["end_m"] for item in intervals), 2),
                "lengte_m": _round_or_none(unique_length, 1),
                "hard_gat_van_m": None,
                "hard_gat_tot_m": None,
                "hard_gat_lengte_m": None,
                "naamzone_marge_links_m": None,
                "naamzone_marge_rechts_m": None,
                "project_links": "",
                "project_rechts": "",
                "dekking_uniek_m": _round_or_none(unique_length, 1),
                "projectbereik_m": _round_or_none(project_span, 1),
                "ijking_span_m": _round_or_none(calibration_span, 1),
                "dekking_pct": _round_or_none(coverage_pct, 1),
                "status": "overzicht",
                "advies": (
                    "Totale projectdekking binnen dit projecttype. "
                    "Gaten en overlap hieronder worden alleen met hetzelfde projecttype vergeleken."
                ),
            }
        )

        previous = intervals[0]
        for current in intervals[1:]:
            previous_start = float(previous["start_m"])
            previous_end = float(previous["end_m"])
            previous_name = clean_display_value(previous["name"])
            start = float(current["start_m"])
            end = float(current["end_m"])
            name = clean_display_value(current["name"])

            if start - previous_end > gap_tolerance_m:
                prev_zone_start, prev_zone_end = _namezone_route_range(
                    previous.get("project_eind_km"),
                    axis_anchors,
                    snap_tolerance_m=float(boundary_snap_tolerance_m),
                )
                current_zone_start, current_zone_end = _namezone_route_range(
                    current.get("project_begin_km"),
                    axis_anchors,
                    snap_tolerance_m=float(boundary_snap_tolerance_m),
                )

                hard_gap_start = max(previous_end, float(prev_zone_end)) if prev_zone_end is not None else previous_end
                hard_gap_end = min(start, float(current_zone_start)) if current_zone_start is not None else start
                hard_gap_length = max(0.0, hard_gap_end - hard_gap_start)
                naamzone_marge_links = max(0.0, hard_gap_start - previous_end)
                naamzone_marge_rechts = max(0.0, start - hard_gap_end)

                if hard_gap_length > gap_tolerance_m:
                    gap_presence = _primary_object_presence_in_gap(
                        object_ranges,
                        axis_id=axis_id,
                        project_type=project_type,
                        start_m=hard_gap_start,
                        end_m=hard_gap_end,
                    )

                    # Alleen echte gatkandidaten melden: een onderbreking zonder
                    # primaire objecten is bij parallelwegen/fietspaden meestal
                    # geen fout maar gewoon een ontbrekend fysiek spoor.
                    if int(gap_presence["count"]) > 0:
                        rows.append(
                            {
                                **common,
                                "controle_type": "gat",
                                "van_m": _round_or_none(previous_end, 2),
                                "tot_m": _round_or_none(start, 2),
                                "lengte_m": _round_or_none(start - previous_end, 1),
                                "hard_gat_van_m": _round_or_none(hard_gap_start, 2),
                                "hard_gat_tot_m": _round_or_none(hard_gap_end, 2),
                                "hard_gat_lengte_m": _round_or_none(hard_gap_length, 1),
                                "naamzone_marge_links_m": _round_or_none(naamzone_marge_links, 1),
                                "naamzone_marge_rechts_m": _round_or_none(naamzone_marge_rechts, 1),
                                "project_links": previous_name,
                                "project_rechts": name,
                                "dekking_uniek_m": None,
                                "projectbereik_m": _round_or_none(project_span, 1),
                                "ijking_span_m": _round_or_none(calibration_span, 1),
                                "dekking_pct": None,
                                "status": "controleer",
                                "advies": (
                                    "Controleer dit harde gat: na aftrek van de toegestane projectnaamzone "
                                    "liggen er nog primaire objecten van hetzelfde spoor in dit interval"
                                    + (
                                        f" ({gap_presence['objects']})."
                                        if clean_display_value(gap_presence.get("objects", ""))
                                        else "."
                                    )
                                ),
                            }
                        )
            elif previous_end - start > gap_tolerance_m:
                rows.append(
                    {
                        **common,
                        "controle_type": "overlap",
                        "van_m": _round_or_none(start, 2),
                        "tot_m": _round_or_none(previous_end, 2),
                        "lengte_m": _round_or_none(previous_end - start, 1),
                        "hard_gat_van_m": None,
                        "hard_gat_tot_m": None,
                        "hard_gat_lengte_m": None,
                        "naamzone_marge_links_m": None,
                        "naamzone_marge_rechts_m": None,
                        "project_links": previous_name,
                        "project_rechts": name,
                        "dekking_uniek_m": None,
                        "projectbereik_m": _round_or_none(project_span, 1),
                        "ijking_span_m": _round_or_none(calibration_span, 1),
                        "dekking_pct": None,
                        "status": "controleer",
                        "advies": "Controleer dubbele dekking of projectnaamgrenzen binnen hetzelfde projecttype.",
                    }
                )

            if end > float(previous["end_m"]):
                previous = current

    if not rows:
        return _empty_coverage_frame()

    return pd.DataFrame(rows, columns=_empty_coverage_frame().columns)


def build_project_axis_diagnostics(
    road_gdf: gpd.GeoDataFrame,
    wegassen_gdf: gpd.GeoDataFrame | None,
    hectopoints_gdf: gpd.GeoDataFrame | None,
    deviation_zones: pd.DataFrame | None,
    selected_road: str,
    *,
    max_anchor_distance_m: float = 40.0,
    max_object_offset_m: float = 40.0,
    boundary_zone_buffer_m: float = 25.0,
    length_tolerance_m: float = 25.0,
    gap_tolerance_m: float = 5.0,
    boundary_snap_tolerance_m: float = DEFAULT_BOUNDARY_SNAP_TOLERANCE_M,
) -> ProjectAxisDiagnosticsResult:
    """
    Bouw v0.34-diagnose: projectgrenzen op geijkte iASSET-wegas.

    Parameters blijven bewust conservatief:
    - ``max_anchor_distance_m`` voorkomt dat hectopunten van parallelle of
      kruisende wegen de ijking vervuilen;
    - ``boundary_zone_buffer_m`` vangt sample-gebaseerde afwijkingszones op;
    - ``boundary_snap_tolerance_m`` laat fysieke grenzen nabij een hectometerpunt
      naar dat hectometerpunt snappen voordat de naamregel wordt toegepast;
    - ``length_tolerance_m`` en ``gap_tolerance_m`` zijn diagnosegrenzen, geen
      automatische beslisregels.
    """
    warnings: list[str] = []

    axes, axis_warnings = _prepare_axes(wegassen_gdf, selected_road)
    warnings.extend(axis_warnings)

    anchors, anchor_warnings = _build_axis_anchors(
        axes,
        hectopoints_gdf,
        max_anchor_distance_m=float(max_anchor_distance_m),
    )
    warnings.extend(anchor_warnings)

    object_ranges = _build_object_ranges(
        road_gdf,
        axes,
        anchors,
        max_object_offset_m=float(max_object_offset_m),
    )

    project_boundaries = _build_project_boundaries(
        road_gdf,
        axes,
        anchors,
        object_ranges,
        deviation_zones,
        selected_road,
        boundary_zone_buffer_m=float(boundary_zone_buffer_m),
        length_tolerance_m=float(length_tolerance_m),
        boundary_snap_tolerance_m=float(boundary_snap_tolerance_m),
    )

    project_coverage = _build_project_coverage(
        project_boundaries,
        anchors,
        axes,
        object_ranges,
        gap_tolerance_m=float(gap_tolerance_m),
        boundary_snap_tolerance_m=float(boundary_snap_tolerance_m),
    )

    if project_boundaries.empty:
        warnings.append("Geen onderhoudsprojectnamen met herkenbare hm-range gevonden voor projectgrensdiagnose.")

    return ProjectAxisDiagnosticsResult(
        calibration_anchors=anchors,
        project_boundaries=project_boundaries,
        project_coverage=project_coverage,
        object_ranges=object_ranges,
        warning=" ".join(dict.fromkeys(part for part in warnings if part)),
    )
