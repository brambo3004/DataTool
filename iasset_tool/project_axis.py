"""
Projectgrenzen en groenveld-projectvoorstellen op een geijkte iASSET-referentieas (v0.35.4).

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


PROJECT_AXIS_SCHEMA_VERSION = "projectaxis-v0.35.4"

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

# v0.35.4: de groenveld-kniplogica werkt niet meer object-voor-object,
# maar eerst met technische reeksen. Besteknummer is ondersteunend; het lege
# veld "verhardingssoort" wordt bewust genegeerd.
GREENFIELD_TECHNICAL_PROFILE_FIELDS = (
    "Soort verharding_N",
    "Soort deklaag specifiek",
    "Jaar aanleg",
    "Jaar deklaag",
    "Jaar conservering",
    "Jaar herstrating",
)

GREENFIELD_SUPPORTING_PROFILE_FIELDS = (
    "Besteknummer",
)

GREENFIELD_IGNORED_FIELDS = (
    "verhardingssoort",
)

GREENFIELD_SPLIT_COLUMNS = GREENFIELD_TECHNICAL_PROFILE_FIELDS + GREENFIELD_SUPPORTING_PROFILE_FIELDS

# Lokale afwijking volgens beheerafspraak:
# maximaal 2 objecten én korter dan 100 m én links/rechts hetzelfde technische
# profiel. Zo wordt een slordige paspoortwaarde geen zelfstandig project.
GREENFIELD_LOCAL_DEVIATION_MAX_OBJECTS = 2
GREENFIELD_LOCAL_DEVIATION_MAX_LENGTH_M = 100.0

# v0.35.4: diagnose van hectometerintervallen. Een interval kan fysiek korter
# of langer zijn dan de administratieve 100 m. Dat is geen automatische fout,
# maar wel belangrijke uitleg bij grenzen zoals einde N398.
HECTOMETER_INTERVAL_ATTENTION_M = 10.0
GREENFIELD_NEAR_ZERO_LENGTH_M = 0.5

# Oude constants blijven beschikbaar voor backwards-compatible tests/imports,
# maar worden sinds v0.35.4 niet meer als beslislaag gebruikt.
GREENFIELD_STRONG_CHANGE_FIELDS = set(GREENFIELD_TECHNICAL_PROFILE_FIELDS)
GREENFIELD_SOFT_ONLY_FIELDS = set(GREENFIELD_SUPPORTING_PROFILE_FIELDS)
GREENFIELD_MIN_HARD_SPLIT_SPAN_M = 250.0
GREENFIELD_STRONG_SINGLE_FIELD_SPAN_M = 750.0


@dataclass(frozen=True)
class ProjectAxisDiagnosticsResult:
    """
    Resultaat van de referentieasdiagnose.

    Alle tabellen zijn gewone DataFrames. Dat houdt de Streamlit-laag simpel en
    maakt export naar CSV zonder extra conversie mogelijk.

    v0.35.1 verfijnt naast de bestaande projectgrenscontrole ook groenveld-
    projectvoorstellen toe. Die voorstellen gebruiken de bestaande
    onderhoudsprojectnaam uit iASSET niet als uitgangspunt, maar vergelijken daar
    achteraf wel mee.
    """

    calibration_anchors: pd.DataFrame
    project_boundaries: pd.DataFrame
    project_coverage: pd.DataFrame
    object_ranges: pd.DataFrame
    project_proposals: pd.DataFrame
    proposal_object_assignments: pd.DataFrame
    proposal_iasset_comparison: pd.DataFrame
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
            "naam",
            "Besteknummer",
            "verhardingssoort",
            "Soort verharding_N",
            "Soort deklaag specifiek",
            "Jaar aanleg",
            "Jaar deklaag",
            "Jaar conservering",
            "Jaar herstrating",
            "status",
            "waarschuwing",
            "bronkwaliteit",
        ]
    )


def _empty_project_proposal_frame() -> pd.DataFrame:
    """Maak een lege tabel voor groenveld-onderhoudsprojectvoorstellen."""
    return pd.DataFrame(
        columns=[
            "voorstel_id",
            "wegnummer",
            "axis_id",
            "axis_naam",
            "project_type",
            "project_family",
            "situering",
            "fysiek_begin_m",
            "fysiek_eind_m",
            "fysiek_lengte_m",
            "fysiek_begin_km",
            "fysiek_eind_km",
            "snap_tolerantie_m",
            "begin_dichtstbijzijnde_hm",
            "begin_snap_afstand_m",
            "begin_gesnapt_naar_hm",
            "naam_begin",
            "eind_dichtstbijzijnde_hm",
            "eind_snap_afstand_m",
            "eind_gesnapt_naar_hm",
            "naam_eind",
            "begin_hm_interval",
            "begin_hm_interval_lengte_m",
            "begin_hm_interval_verwacht_m",
            "begin_hm_interval_afwijking_m",
            "begin_grenspositie_in_interval_m",
            "begin_grenspositie_in_interval_pct",
            "begin_grensdiagnose",
            "eind_hm_interval",
            "eind_hm_interval_lengte_m",
            "eind_hm_interval_verwacht_m",
            "eind_hm_interval_afwijking_m",
            "eind_grenspositie_in_interval_m",
            "eind_grenspositie_in_interval_pct",
            "eind_grensdiagnose",
            "grensdiagnose",
            "onderhoudsproject_voorgesteld",
            "knipreden_begin",
            "knipreden_eind",
            "knipprofiel",
            "technisch_profiel",
            "bestek_signalen",
            "datakwaliteit_signalen",
            "lokale_afwijkingen",
            "ingesloten_objecten",
            "harde_knipsignalen",
            "zachte_signalen",
            "aantal_primaire_objecten",
            "bestaande_onderhoudsprojecten",
            "vergelijking_iasset_status",
            "status_voorstel",
            "hoofdmelding",
            "contextmelding",
            "bronkwaliteit",
        ]
    )


def _empty_proposal_object_assignment_frame() -> pd.DataFrame:
    """Maak een lege object-toewijzingstabel voor projectvoorstellen."""
    return pd.DataFrame(
        columns=[
            "voorstel_id",
            "onderhoudsproject_voorgesteld",
            "sys_id",
            "nummer",
            "naam",
            "subthema",
            "project_type",
            "project_family",
            "situering",
            "bestaand_onderhoudsproject",
            "axis_id",
            "fysiek_begin_m",
            "fysiek_eind_m",
            "fysiek_begin_km",
            "fysiek_eind_km",
            "Besteknummer",
            "verhardingssoort",
            "Soort verharding_N",
            "Soort deklaag specifiek",
            "Jaar aanleg",
            "Jaar deklaag",
            "Jaar conservering",
            "Jaar herstrating",
            "technisch_profiel",
            "besteknummer_norm",
            "lokale_afwijking_type",
            "ingesloten_in_voorstel",
            "object_kniprol",
            "toewijzing_status",
            "toewijzing_melding",
            "bronkwaliteit",
        ]
    )


def _empty_proposal_iasset_comparison_frame() -> pd.DataFrame:
    """Maak een lege vergelijkingstabel tussen groenveldvoorstellen en iASSET."""
    return pd.DataFrame(
        columns=[
            "vergelijking_niveau",
            "bestaand_onderhoudsproject",
            "voorstel_id",
            "onderhoudsproject_voorgesteld",
            "verschil_type",
            "aantal_objecten",
            "status",
            "hoofdmelding",
            "contextmelding",
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
        "nearest_hm_label": _format_hm_label(nearest_hm),
        "snap_distance_m": float(snap_distance_m),
        "snapped": bool(snapped),
        "snapped_to_hm": bool(snapped),
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
    return str(_snap_name_rule_details(value_km, snap_tolerance_m).get("label") or "")


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
                    "naam": clean_display_value(row.get("naam", "")),
                    "Besteknummer": clean_display_value(row.get("Besteknummer", "")),
                    "verhardingssoort": clean_display_value(row.get("verhardingssoort", "")),
                    "Soort verharding_N": clean_display_value(row.get("Soort verharding_N", "")),
                    "Soort deklaag specifiek": clean_display_value(row.get("Soort deklaag specifiek", "")),
                    "Jaar aanleg": clean_display_value(row.get("Jaar aanleg", "")),
                    "Jaar deklaag": clean_display_value(row.get("Jaar deklaag", "")),
                    "Jaar conservering": clean_display_value(row.get("Jaar conservering", "")),
                    "Jaar herstrating": clean_display_value(row.get("Jaar herstrating", "")),
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
                "naam": clean_display_value(row.get("naam", "")),
                "Besteknummer": clean_display_value(row.get("Besteknummer", "")),
                "verhardingssoort": clean_display_value(row.get("verhardingssoort", "")),
                "Soort verharding_N": clean_display_value(row.get("Soort verharding_N", "")),
                "Soort deklaag specifiek": clean_display_value(row.get("Soort deklaag specifiek", "")),
                "Jaar aanleg": clean_display_value(row.get("Jaar aanleg", "")),
                "Jaar deklaag": clean_display_value(row.get("Jaar deklaag", "")),
                "Jaar conservering": clean_display_value(row.get("Jaar conservering", "")),
                "Jaar herstrating": clean_display_value(row.get("Jaar herstrating", "")),
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





def _normalise_greenfield_value(value: Any) -> str:
    """Normaliseer een kenmerkwaarde voor de groenveld-kniplogica.

    iASSET-exports bevatten geregeld lege cellen, NaN-achtige teksten of
    jaartallen als ``2020.0``. Voor de kniplogica willen we die waarden
    consequent behandelen. Een lege waarde blijft leeg: dat is geen crashreden,
    maar later wél input voor een datakwaliteitsmelding.
    """
    display = clean_display_value(value)
    if not display or display.lower() in {"nan", "none", "nat", "null", "-", "n.v.t.", "nvt"}:
        return ""
    # Jaartallen kunnen soms als 2020.0 binnenkomen; toon ze als 2020.
    try:
        numeric = float(display.replace(",", "."))
        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError, AttributeError):
        pass
    return display.strip()


def _greenfield_technical_profile(row: pd.Series) -> dict[str, str]:
    """Maak het leidende technische profiel voor projectvoorstellen.

    Besteknummer zit hier bewust niet in. Dat veld is nuttig als ondersteunend
    signaal, maar in de praktijk ook vaak ontbrekend of achterhaald. Een
    onderhoudsprojectknip moet daarom primair uit de technische paspoortvelden
    volgen.
    """
    return {
        column: _normalise_greenfield_value(row.get(column, ""))
        for column in GREENFIELD_TECHNICAL_PROFILE_FIELDS
    }


def _greenfield_supporting_profile(row: pd.Series) -> dict[str, str]:
    """Maak het ondersteunende profiel voor signalen zoals besteknummer."""
    return {
        column: _normalise_greenfield_value(row.get(column, ""))
        for column in GREENFIELD_SUPPORTING_PROFILE_FIELDS
    }


def _greenfield_bestek_value(row: pd.Series) -> str:
    """Lees het besteknummer als ondersteunend signaal."""
    return _normalise_greenfield_value(row.get("Besteknummer", ""))


def _greenfield_split_key(row: pd.Series) -> dict[str, str]:
    """Maak een complete uitlegsleutel voor exports en debugging.

    De technische beslislaag gebruikt ``_greenfield_technical_profile``. Deze
    sleutel bevat daarnaast het ondersteunende besteknummer, zodat de CSV blijft
    uitleggen welke beheerwaarden binnen een voorstel zijn aangetroffen.
    """
    key: dict[str, str] = {}
    key.update(_greenfield_technical_profile(row))
    key.update(_greenfield_supporting_profile(row))
    return key


def _greenfield_changed_fields(
    previous_key: dict[str, str],
    current_key: dict[str, str],
    fields: Iterable[str] | None = None,
) -> list[str]:
    """Geef terug welke beheerkenmerken tussen twee profielen verschillen."""
    field_order = tuple(fields) if fields is not None else GREENFIELD_SPLIT_COLUMNS
    changed: list[str] = []
    for column in field_order:
        if previous_key.get(column, "") != current_key.get(column, ""):
            changed.append(column)
    return changed


def _greenfield_profiles_equal(left: dict[str, str], right: dict[str, str]) -> bool:
    """Vergelijk twee technische profielen in vaste veldvolgorde."""
    return not _greenfield_changed_fields(left, right, GREENFIELD_TECHNICAL_PROFILE_FIELDS)


def _format_changed_fields(fields: Iterable[str]) -> str:
    """Formatteer kenmerkvelden stabiel en leesbaar voor knipmeldingen."""
    field_set = set(fields)
    return ", ".join(field for field in GREENFIELD_SPLIT_COLUMNS if field in field_set)


def _format_technical_profile(profile: dict[str, str]) -> str:
    """Formatteer het technische profiel compact voor CSV en kaartinspectie."""
    return ", ".join(
        f"{column}={value or '<leeg>'}"
        for column, value in profile.items()
    )


def _profile_has_missing_values(profile: dict[str, str]) -> bool:
    """Bepaal of een technisch profiel één of meer ontbrekende waarden bevat."""
    return any(not clean_display_value(value) for value in profile.values())


def _candidate_is_missing_only_deviation(
    base_profile: dict[str, str],
    candidate_profile: dict[str, str],
) -> bool:
    """Bepaal of een lokale afwijking alleen uit ontbrekende waarden bestaat.

    Voorbeeld:
    profiel A -> korte reeks met leeg ``Jaar deklaag`` -> profiel A.
    Dat is eerder datakwaliteit dan een onderhoudsprojectgrens.
    """
    changed_fields = _greenfield_changed_fields(
        base_profile,
        candidate_profile,
        GREENFIELD_TECHNICAL_PROFILE_FIELDS,
    )
    if not changed_fields:
        return False
    for field in changed_fields:
        candidate_value = clean_display_value(candidate_profile.get(field, ""))
        if candidate_value:
            return False
    return True


def _greenfield_change_decision(changed_fields: list[str], current_span_m: float) -> tuple[bool, str, str]:
    """Backwards-compatible beslisfunctie uit v0.35.1.

    Sinds v0.35.4 gebruikt de echte projectvoorstellenlogica reeksherkenning.
    Deze functie blijft bestaan omdat tests of externe notebooks haar kunnen
    importeren, maar nieuwe code hoort via technische profielen te lopen.
    """
    fields = [field for field in changed_fields if field in GREENFIELD_SPLIT_COLUMNS]
    if not fields:
        return False, "", ""

    formatted = _format_changed_fields(fields)
    technical_fields = set(fields).intersection(GREENFIELD_TECHNICAL_PROFILE_FIELDS)
    bestek_only = set(fields) == set(GREENFIELD_SUPPORTING_PROFILE_FIELDS)

    if bestek_only:
        return False, "ondersteunend besteksignaal", formatted

    if technical_fields and float(current_span_m or 0.0) >= GREENFIELD_MIN_HARD_SPLIT_SPAN_M:
        return True, "harde technische profielknip", formatted

    return False, "zacht kenmerksignaal", formatted


def _format_project_type_from_row(row: pd.Series) -> tuple[str, str, str]:
    """Lees projecttype/family/situering veilig uit een objectprojectieregel."""
    project_type = clean_display_value(row.get("project_type", ""))
    family = clean_display_value(row.get("project_family", ""))
    situering = clean_display_value(row.get("situering", ""))
    if not project_type:
        project_type, family, situering = _object_project_type_from_row(row, "")
    return project_type, family, situering


def _name_rule_from_route(
    route_m: Any,
    axis_anchors: pd.DataFrame,
    *,
    snap_tolerance_m: float,
) -> dict[str, Any]:
    """
    Bepaal de naamregel-details voor een fysieke routepositie op de geijkte as.

    Dit gebruikt dezelfde 2-stapsregel als v0.34.3:
    1. binnen snap-tolerantie naar hectometerpunt;
    2. anders naar boven op het eerstvolgende hectometerlabel.
    """
    km_value, in_range = _route_to_km(route_m, axis_anchors)
    details = _snap_name_rule_details(km_value, snap_tolerance_m=float(snap_tolerance_m))
    details["km"] = km_value
    details["in_range"] = bool(in_range)
    return details



def _hectometer_interval_diagnostics(route_m: Any, axis_anchors: pd.DataFrame) -> dict[str, Any]:
    """
    Beschrijf in welk geijkt hectometerinterval een routepositie valt.

    Waarom deze diagnose?
    Bij de N398 zagen we dat het interval 6.2-6.3 fysiek ongeveer 73 m is,
    terwijl de administratieve hectometrering 100 m opschuift. De lineaire
    ijking is dan rekenkundig logisch, maar zonder uitleg lijkt de grenswaarde
    onverklaarbaar. Deze helper maakt dat expliciet in de exports.
    """
    empty = {
        "hm_interval": "",
        "hm_interval_lengte_m": None,
        "hm_interval_verwacht_m": None,
        "hm_interval_afwijking_m": None,
        "grenspositie_in_interval_m": None,
        "grenspositie_in_interval_pct": None,
        "grensdiagnose": "",
    }
    if axis_anchors is None or len(axis_anchors) < 2:
        return empty

    try:
        route_value = float(route_m)
    except (TypeError, ValueError, OverflowError):
        return empty

    if not math.isfinite(route_value):
        return empty

    work = axis_anchors.copy()
    work["route_m"] = pd.to_numeric(work.get("route_m"), errors="coerce")
    work["hm_km"] = pd.to_numeric(work.get("hm_km"), errors="coerce")
    work = work.dropna(subset=["route_m", "hm_km"]).sort_values("route_m").reset_index(drop=True)
    if len(work) < 2:
        return empty

    routes = work["route_m"].astype(float).tolist()
    hms = work["hm_km"].astype(float).tolist()

    interval_index: int | None = None
    buiten_ijkbereik = False
    if route_value < routes[0]:
        interval_index = 1
        buiten_ijkbereik = True
    elif route_value > routes[-1]:
        interval_index = len(routes) - 1
        buiten_ijkbereik = True
    elif route_value == routes[-1]:
        # Geef bij de laatste anchor het voorgaande interval terug; dat verklaart
        # juist eindpuntcasussen zoals het korte interval 6.2-6.3.
        interval_index = len(routes) - 1
    else:
        for index in range(1, len(routes)):
            if route_value <= routes[index]:
                interval_index = index
                break

    if interval_index is None or interval_index <= 0 or interval_index >= len(routes):
        return empty

    left_route = float(routes[interval_index - 1])
    right_route = float(routes[interval_index])
    left_hm = float(hms[interval_index - 1])
    right_hm = float(hms[interval_index])

    actual_length_m = right_route - left_route
    expected_length_m = abs(right_hm - left_hm) * 1000.0
    if actual_length_m <= 0 or expected_length_m <= 0:
        return empty

    position_m = route_value - left_route
    position_pct = (position_m / actual_length_m) * 100.0
    deviation_m = actual_length_m - expected_length_m
    interval_label = f"{_format_hm_label(left_hm)}-{_format_hm_label(right_hm)}"

    diagnosis_parts: list[str] = []
    if buiten_ijkbereik:
        diagnosis_parts.append("grens buiten ijkbereik; dichtstbijzijnde interval gebruikt voor diagnose")
    if abs(deviation_m) >= HECTOMETER_INTERVAL_ATTENTION_M:
        diagnosis_parts.append(
            "hectometerinterval "
            f"{interval_label} is fysiek {actual_length_m:.1f} m in plaats van "
            f"{expected_length_m:.1f} m"
        )

    return {
        "hm_interval": interval_label,
        "hm_interval_lengte_m": float(actual_length_m),
        "hm_interval_verwacht_m": float(expected_length_m),
        "hm_interval_afwijking_m": float(deviation_m),
        "grenspositie_in_interval_m": float(position_m),
        "grenspositie_in_interval_pct": float(position_pct),
        "grensdiagnose": "; ".join(diagnosis_parts),
    }


def _copy_boundary_diagnostics(prefix: str, diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Maak exportkolommen voor begin/eindgrensdiagnose."""
    return {
        f"{prefix}_hm_interval": clean_display_value(diagnostics.get("hm_interval", "")),
        f"{prefix}_hm_interval_lengte_m": _round_or_none(diagnostics.get("hm_interval_lengte_m"), 2),
        f"{prefix}_hm_interval_verwacht_m": _round_or_none(diagnostics.get("hm_interval_verwacht_m"), 2),
        f"{prefix}_hm_interval_afwijking_m": _round_or_none(diagnostics.get("hm_interval_afwijking_m"), 2),
        f"{prefix}_grenspositie_in_interval_m": _round_or_none(
            diagnostics.get("grenspositie_in_interval_m"),
            2,
        ),
        f"{prefix}_grenspositie_in_interval_pct": _round_or_none(
            diagnostics.get("grenspositie_in_interval_pct"),
            1,
        ),
        f"{prefix}_grensdiagnose": clean_display_value(diagnostics.get("grensdiagnose", "")),
    }


def _unique_clean_values(values: Iterable[Any]) -> list[str]:
    """Unieke, niet-lege waarden in stabiele volgorde."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        display = clean_display_value(value)
        if not display or display in seen:
            continue
        seen.add(display)
        result.append(display)
    return result


def _choose_segment_status(object_rows: pd.DataFrame, begin_rule: dict[str, Any], end_rule: dict[str, Any], segment_length_m: float | None = None) -> tuple[str, str]:
    """Bepaal een voorzichtige status en hoofdmelding voor één groenveldvoorstel."""
    warnings: list[str] = []
    status = "ok"

    if not begin_rule.get("label") or not end_rule.get("label"):
        warnings.append("voorstelgrens kan niet naar onderhoudsprojectnaam worden vertaald")
        status = _worst_status(status, "controleer")

    try:
        length_value = float(segment_length_m) if segment_length_m is not None else None
    except (TypeError, ValueError, OverflowError):
        length_value = None
    if length_value is not None and math.isfinite(length_value) and length_value <= GREENFIELD_NEAR_ZERO_LENGTH_M:
        warnings.append("voorstel heeft vrijwel geen fysieke lengte")
        status = _worst_status(status, "controleer")

    if not begin_rule.get("in_range", False):
        warnings.append("begin voorstel buiten ijkbereik")
        status = _worst_status(status, "controleer")
    if not end_rule.get("in_range", False):
        warnings.append("eind voorstel buiten ijkbereik")
        status = _worst_status(status, "controleer")

    object_statuses = object_rows.get("status", pd.Series(dtype="object")).fillna("").astype(str).str.lower()
    if (object_statuses == "controleer").any():
        warnings.append("één of meer primaire objecten hebben projectiewaarschuwingen")
        status = _worst_status(status, "aandacht")

    if not warnings:
        return status, "Voorstel opgebouwd uit primaire objecten vanaf nul."
    return status, "; ".join(dict.fromkeys(warnings))


def _build_project_proposals(
    object_ranges: pd.DataFrame,
    anchors: pd.DataFrame,
    selected_road: str,
    *,
    gap_tolerance_m: float,
    boundary_snap_tolerance_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Bouw v0.35-groenveldvoorstellen voor onderhoudsprojecten.

    v0.35.4 gebruikt reeksherkenning:
    - eerst technische profielen bepalen;
    - dan aaneengesloten reeksen vormen;
    - lokale afwijkingen insluiten als datakwaliteit/controle;
    - pas daarna onderhoudsprojectgrenzen voorstellen.

    Bestaande iASSET-onderhoudsprojecten blijven alleen vergelijkingsmateriaal.
    """
    if object_ranges is None or object_ranges.empty:
        return (
            _empty_project_proposal_frame(),
            _empty_proposal_object_assignment_frame(),
            _empty_proposal_iasset_comparison_frame(),
        )

    working = object_ranges.copy()
    if "primair_object" not in working.columns:
        return (
            _empty_project_proposal_frame(),
            _empty_proposal_object_assignment_frame(),
            _empty_proposal_iasset_comparison_frame(),
        )

    working = working[working["primair_object"].fillna(False).astype(bool)].copy()
    if working.empty:
        return (
            _empty_project_proposal_frame(),
            _empty_proposal_object_assignment_frame(),
            _empty_proposal_iasset_comparison_frame(),
        )

    for column in ["route_begin_m", "route_eind_m"]:
        working[column] = pd.to_numeric(working.get(column), errors="coerce")
    working = working.dropna(subset=["route_begin_m", "route_eind_m", "axis_id", "project_type"])
    if working.empty:
        return (
            _empty_project_proposal_frame(),
            _empty_proposal_object_assignment_frame(),
            _empty_proposal_iasset_comparison_frame(),
        )

    working["route_start_norm_m"] = working[["route_begin_m", "route_eind_m"]].min(axis=1)
    working["route_end_norm_m"] = working[["route_begin_m", "route_eind_m"]].max(axis=1)

    proposal_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    proposal_counter = 0

    def _object_label(row: pd.Series) -> str:
        """Maak een korte objectidentificatie voor signalen in exports."""
        for column in ("nummer", "naam", "sys_id"):
            value = clean_display_value(row.get(column, ""))
            if value:
                return value
        return "object"

    def _run_extent_m(group_df: pd.DataFrame, indices: list[int]) -> tuple[float, float, float]:
        """Bepaal begin, eind en lengte van een technische reeks."""
        if not indices:
            return 0.0, 0.0, 0.0
        segment = group_df.loc[indices]
        start_m = float(segment["route_start_norm_m"].min())
        end_m = float(segment["route_end_norm_m"].max())
        return start_m, end_m, max(end_m - start_m, 0.0)

    def _bestek_values_for_indices(group_df: pd.DataFrame, indices: list[int]) -> list[str]:
        """Lees unieke, gevulde besteknummers binnen een reeks."""
        if not indices:
            return []
        values = [
            _greenfield_bestek_value(row)
            for _, row in group_df.loc[indices].iterrows()
        ]
        return list(dict.fromkeys(value for value in values if value))

    def _make_run(
        group_df: pd.DataFrame,
        indices: list[int],
        technical_profile: dict[str, str],
        *,
        gap_before_reason: str = "",
    ) -> dict[str, Any]:
        """Maak een technische reeks met metadata voor latere voorstelvorming."""
        start_m, end_m, length_m = _run_extent_m(group_df, indices)
        return {
            "indices": list(indices),
            "technical_profile": dict(technical_profile),
            "start_m": start_m,
            "end_m": end_m,
            "length_m": length_m,
            "gap_before_reason": gap_before_reason,
            "data_quality_signals": [],
            "local_deviation_signals": [],
            "bestek_signals": [],
            "hard_signals": [],
            "included_object_labels": [],
        }

    def _build_initial_runs(group_df: pd.DataFrame) -> list[dict[str, Any]]:
        """Vorm aaneengesloten technische reeksen binnen één fysiek spoor.

        Besteknummer wordt hierbij bewust genegeerd. Een wijziging of ontbrekende
        waarde in besteknummer kan later een signaal worden, maar mag niet alleen
        een onderhoudsprojectknip afdwingen.
        """
        runs: list[dict[str, Any]] = []
        current_indices: list[int] = []
        current_profile: dict[str, str] | None = None
        current_gap_before = ""
        previous_end_m: float | None = None

        def flush_current() -> None:
            nonlocal current_indices, current_profile, current_gap_before
            if current_indices and current_profile is not None:
                runs.append(
                    _make_run(
                        group_df,
                        current_indices,
                        current_profile,
                        gap_before_reason=current_gap_before,
                    )
                )
            current_indices = []
            current_profile = None
            current_gap_before = ""

        for idx, object_row in group_df.iterrows():
            start_m = float(object_row["route_start_norm_m"])
            end_m = float(object_row["route_end_norm_m"])
            profile = _greenfield_technical_profile(object_row)
            gap_before_reason = ""
            if previous_end_m is not None:
                gap_m = start_m - float(previous_end_m)
                if gap_m > float(gap_tolerance_m):
                    gap_before_reason = f"gat > {gap_tolerance_m:g} m"

            if not current_indices:
                current_indices = [idx]
                current_profile = profile
                current_gap_before = gap_before_reason
                previous_end_m = end_m
                continue

            if gap_before_reason:
                flush_current()
                current_indices = [idx]
                current_profile = profile
                current_gap_before = gap_before_reason
            elif current_profile is not None and _greenfield_profiles_equal(current_profile, profile):
                current_indices.append(idx)
            else:
                flush_current()
                current_indices = [idx]
                current_profile = profile
                current_gap_before = ""

            previous_end_m = max(float(previous_end_m or end_m), end_m)

        flush_current()
        return runs

    def _classify_local_run(
        base_profile: dict[str, str],
        candidate_profile: dict[str, str],
    ) -> tuple[str, str]:
        """Classificeer een lokale tussenreeks als datakwaliteit of controle."""
        changed_fields = _greenfield_changed_fields(
            base_profile,
            candidate_profile,
            GREENFIELD_TECHNICAL_PROFILE_FIELDS,
        )
        formatted_fields = _format_changed_fields(changed_fields)
        if _candidate_is_missing_only_deviation(base_profile, candidate_profile):
            return "lokale_datakwaliteit", formatted_fields
        return "lokale_technische_afwijking", formatted_fields

    def _merge_local_deviation_runs(
        group_df: pd.DataFrame,
        runs: list[dict[str, Any]],
        object_roles: dict[int, str],
    ) -> list[dict[str, Any]]:
        """Sluit lokale A-B-A-afwijkingen in de omliggende stabiele reeks in.

        Beheerregel:
        maximaal 2 objecten én korter dan 100 meter én links/rechts hetzelfde
        technische profiel. Een missende waarde wordt datakwaliteit; een echte
        korte technische afwijking wordt controle, maar geen projectknip.
        """
        changed = True
        while changed and len(runs) >= 3:
            changed = False
            i = 1
            while i < len(runs) - 1:
                left = runs[i - 1]
                candidate = runs[i]
                right = runs[i + 1]

                if not _greenfield_profiles_equal(left["technical_profile"], right["technical_profile"]):
                    i += 1
                    continue

                # Niet over fysieke gaten heen samenvoegen.
                if clean_display_value(candidate.get("gap_before_reason", "")) or clean_display_value(right.get("gap_before_reason", "")):
                    i += 1
                    continue

                object_count = len(candidate["indices"])
                _, _, candidate_length_m = _run_extent_m(group_df, candidate["indices"])
                is_local = (
                    object_count <= GREENFIELD_LOCAL_DEVIATION_MAX_OBJECTS
                    and candidate_length_m < GREENFIELD_LOCAL_DEVIATION_MAX_LENGTH_M
                )
                if not is_local:
                    i += 1
                    continue

                deviation_type, changed_fields_text = _classify_local_run(
                    left["technical_profile"],
                    candidate["technical_profile"],
                )
                labels = [_object_label(row) for _, row in group_df.loc[candidate["indices"]].iterrows()]
                label_text = ", ".join(labels)
                length_text = f"{candidate_length_m:.0f} m"
                if deviation_type == "lokale_datakwaliteit":
                    signal = (
                        "lokale datakwaliteit: "
                        f"{changed_fields_text or 'technisch profiel'} ontbreekt bij "
                        f"{object_count} object(en), {length_text}; links/rechts hetzelfde profiel"
                    )
                    signal_bucket = "data_quality_signals"
                else:
                    signal = (
                        "lokale technische afwijking: "
                        f"{changed_fields_text or 'technisch profiel'} wijkt af bij "
                        f"{object_count} object(en), {length_text}; geen projectknip"
                    )
                    signal_bucket = "local_deviation_signals"

                merged_indices = left["indices"] + candidate["indices"] + right["indices"]
                merged_run = _make_run(
                    group_df,
                    merged_indices,
                    left["technical_profile"],
                    gap_before_reason=left.get("gap_before_reason", ""),
                )

                for bucket in ("data_quality_signals", "local_deviation_signals", "bestek_signals", "hard_signals", "included_object_labels"):
                    values: list[str] = []
                    for run in (left, candidate, right):
                        for value in run.get(bucket, []):
                            value_text = clean_display_value(value)
                            if value_text and value_text not in values:
                                values.append(value_text)
                    merged_run[bucket] = values

                merged_run[signal_bucket].append(signal)
                for label in labels:
                    if label and label not in merged_run["included_object_labels"]:
                        merged_run["included_object_labels"].append(label)

                for object_idx in candidate["indices"]:
                    object_roles[object_idx] = deviation_type

                runs = runs[: i - 1] + [merged_run] + runs[i + 2 :]
                changed = True
                i = max(1, i - 1)

        return runs

    def _add_bestek_signals(group_df: pd.DataFrame, run: dict[str, Any], object_roles: dict[int, str]) -> None:
        """Voeg ondersteunende besteksignalen toe zonder op bestek te knippen."""
        indices = run["indices"]
        if not indices:
            return

        bestek_values: list[str] = []
        missing_indices: list[int] = []
        for idx, object_row in group_df.loc[indices].iterrows():
            bestek = _greenfield_bestek_value(object_row)
            if bestek:
                if bestek not in bestek_values:
                    bestek_values.append(bestek)
            else:
                missing_indices.append(idx)

        if len(bestek_values) > 1:
            run["bestek_signals"].append(
                "ondersteunend besteksignaal: meerdere besteknummers binnen hetzelfde technische profiel: "
                + ", ".join(bestek_values)
            )

        if missing_indices:
            missing_segment = group_df.loc[missing_indices]
            missing_length_m = float(
                (missing_segment["route_end_norm_m"] - missing_segment["route_start_norm_m"]).clip(lower=0).sum()
            )
            if (
                len(missing_indices) <= GREENFIELD_LOCAL_DEVIATION_MAX_OBJECTS
                and missing_length_m < GREENFIELD_LOCAL_DEVIATION_MAX_LENGTH_M
                and len(bestek_values) == 1
            ):
                run["data_quality_signals"].append(
                    "lokale datakwaliteit: enkele objecten missen Besteknummer binnen verder stabiel technisch profiel"
                )
                for idx in missing_indices:
                    if not object_roles.get(idx):
                        object_roles[idx] = "lokale_datakwaliteit"
            elif len(bestek_values) == 0:
                run["bestek_signals"].append("ondersteunend besteksignaal: Besteknummer ontbreekt voor alle objecten")
            else:
                run["bestek_signals"].append(
                    f"ondersteunend besteksignaal: Besteknummer ontbreekt bij {len(missing_indices)} object(en)"
                )

    def _transition_reason(
        group_df: pd.DataFrame,
        current_run: dict[str, Any],
        next_run: dict[str, Any] | None,
    ) -> str:
        """Beschrijf waarom een projectvoorstel eindigt richting de volgende reeks."""
        if next_run is None:
            return "einde spoor"

        gap_reason = clean_display_value(next_run.get("gap_before_reason", ""))
        if gap_reason:
            return gap_reason

        changed_fields = _greenfield_changed_fields(
            current_run["technical_profile"],
            next_run["technical_profile"],
            GREENFIELD_TECHNICAL_PROFILE_FIELDS,
        )
        if not changed_fields:
            return "geen technische knip"

        changed_text = _format_changed_fields(changed_fields)
        current_bestek = set(_bestek_values_for_indices(group_df, current_run["indices"]))
        next_bestek = set(_bestek_values_for_indices(group_df, next_run["indices"]))
        bestek_changed = bool(current_bestek and next_bestek and current_bestek.isdisjoint(next_bestek))

        changed_has_missing = any(
            not clean_display_value(current_run["technical_profile"].get(field, ""))
            or not clean_display_value(next_run["technical_profile"].get(field, ""))
            for field in changed_fields
        )
        if changed_has_missing:
            base = f"controleer structurele technische profielknip: {changed_text}"
        else:
            base = f"harde technische profielknip: {changed_text}"

        if bestek_changed:
            base += "; besteknummer wijzigt mee"
        return base

    def _run_hard_signals(begin_reason: str, end_reason: str) -> list[str]:
        """Selecteer harde signalen uit begin/eindredenen."""
        signals: list[str] = []
        for reason in (begin_reason, end_reason):
            reason_text = clean_display_value(reason)
            if not reason_text:
                continue
            lower = reason_text.lower()
            if "harde" in lower or "gat >" in lower or "structurele technische" in lower:
                signals.append(reason_text)
        return list(dict.fromkeys(signals))

    def _proposal_status_with_run_signals(
        status: str,
        hoofd: str,
        run: dict[str, Any],
    ) -> tuple[str, str]:
        """Verwerk datakwaliteit en lokale afwijkingen in de voorstelstatus."""
        if run.get("local_deviation_signals"):
            status = _worst_status(status, "controleer")
            if hoofd == "Voorstel opgebouwd uit primaire objecten vanaf nul.":
                hoofd = "Voorstel bevat lokale technische afwijking binnen verder stabiele reeks."
        elif run.get("data_quality_signals") or run.get("bestek_signals"):
            status = _worst_status(status, "aandacht")
            if hoofd == "Voorstel opgebouwd uit primaire objecten vanaf nul.":
                hoofd = "Voorstel bevat datakwaliteits- of besteksignaal binnen stabiele reeks."
        return status, hoofd

    group_columns = ["axis_id", "project_type"]
    for (axis_id_raw, project_type_raw), group in working.groupby(group_columns, dropna=False, sort=True):
        axis_id = clean_display_value(axis_id_raw)
        project_type = clean_display_value(project_type_raw)
        if not axis_id or not project_type:
            continue

        axis_anchors = _anchors_for_axis(anchors, axis_id)
        if not _axis_is_calibrated(axis_anchors):
            continue

        group = group.sort_values(
            by=["route_start_norm_m", "route_end_norm_m", "nummer"],
            ascending=[True, True, True],
            kind="stable",
        ).reset_index(drop=True)

        object_roles: dict[int, str] = {}
        runs = _build_initial_runs(group)
        runs = _merge_local_deviation_runs(group, runs, object_roles)
        for run in runs:
            _add_bestek_signals(group, run, object_roles)

        previous_end_reason = "start spoor"
        for run_index, run in enumerate(runs):
            next_run = runs[run_index + 1] if run_index + 1 < len(runs) else None
            begin_reason = previous_end_reason or "start spoor"
            if clean_display_value(run.get("gap_before_reason", "")):
                begin_reason = clean_display_value(run.get("gap_before_reason", ""))
            end_reason = _transition_reason(group, run, next_run)

            segment = group.loc[run["indices"]].copy()
            if segment.empty:
                previous_end_reason = end_reason
                continue

            proposal_counter += 1
            first = segment.iloc[0]
            proposal_id = f"{selected_road}-{project_type}-{proposal_counter:04d}"

            start_m = float(segment["route_start_norm_m"].min())
            end_m = float(segment["route_end_norm_m"].max())
            segment_length_m = max(end_m - start_m, 0.0)
            begin_rule = _name_rule_from_route(start_m, axis_anchors, snap_tolerance_m=float(boundary_snap_tolerance_m))
            end_rule = _name_rule_from_route(end_m, axis_anchors, snap_tolerance_m=float(boundary_snap_tolerance_m))
            begin_interval = _hectometer_interval_diagnostics(start_m, axis_anchors)
            end_interval = _hectometer_interval_diagnostics(end_m, axis_anchors)
            begin_label = str(begin_rule.get("label") or "")
            end_label = str(end_rule.get("label") or "")

            road_label = clean_display_value(selected_road)
            proposed_name = f"{road_label}-{project_type}-{begin_label}-{end_label}" if begin_label and end_label else ""
            status, hoofd = _choose_segment_status(segment, begin_rule, end_rule, segment_length_m)
            status, hoofd = _proposal_status_with_run_signals(status, hoofd, run)

            existing_projects = _unique_clean_values(segment.get("Onderhoudsproject", []))
            existing_joined = " | ".join(existing_projects)
            if len(existing_projects) > 1:
                comparison_status = "controleer"
                context_status = "objecten komen uit meerdere bestaande onderhoudsprojecten"
            elif len(existing_projects) == 1 and proposed_name and existing_projects[0] != proposed_name:
                comparison_status = "aandacht"
                context_status = "voorgestelde naam wijkt af van bestaande iASSET-naam"
            elif len(existing_projects) == 1 and proposed_name == existing_projects[0]:
                comparison_status = "ok"
                context_status = "bestaande iASSET-naam komt overeen"
            else:
                comparison_status = "aandacht"
                context_status = "geen bestaand onderhoudsproject op primaire objecten"

            status = _worst_status(status, comparison_status if comparison_status != "ok" else "ok")

            technical_profile_text = _format_technical_profile(run["technical_profile"])
            bestek_signal_text = " | ".join(dict.fromkeys(run.get("bestek_signals", [])))
            data_quality_signal_text = " | ".join(dict.fromkeys(run.get("data_quality_signals", [])))
            local_deviation_text = " | ".join(dict.fromkeys(run.get("local_deviation_signals", [])))
            included_objects_text = " | ".join(dict.fromkeys(run.get("included_object_labels", [])))
            hard_signal_text = " | ".join(dict.fromkeys(run.get("hard_signals", []) + _run_hard_signals(begin_reason, end_reason)))
            soft_signal_text = " | ".join(
                dict.fromkeys(
                    [
                        value
                        for value in (
                            run.get("bestek_signals", [])
                            + run.get("data_quality_signals", [])
                            + run.get("local_deviation_signals", [])
                        )
                        if clean_display_value(value)
                    ]
                )
            )

            context_parts = [
                context_status,
                f"knipreden begin: {begin_reason}",
                f"knipreden eind: {end_reason or 'einde spoor'}",
                f"harde knipsignalen: {hard_signal_text}" if hard_signal_text else "",
                f"zachte signalen binnen voorstel: {soft_signal_text}" if soft_signal_text else "",
                (
                    "grensdiagnose: "
                    + " | ".join(
                        part for part in [
                            clean_display_value(begin_interval.get("grensdiagnose", "")),
                            clean_display_value(end_interval.get("grensdiagnose", "")),
                        ]
                        if part
                    )
                    if clean_display_value(begin_interval.get("grensdiagnose", ""))
                    or clean_display_value(end_interval.get("grensdiagnose", ""))
                    else ""
                ),
                f"technisch profiel: {technical_profile_text}" if technical_profile_text else "",
            ]
            context = "; ".join(part for part in context_parts if part)

            begin_diagnostic_columns = _copy_boundary_diagnostics("begin", begin_interval)
            end_diagnostic_columns = _copy_boundary_diagnostics("eind", end_interval)
            combined_grensdiagnose = " | ".join(
                part for part in [
                    begin_diagnostic_columns.get("begin_grensdiagnose", ""),
                    end_diagnostic_columns.get("eind_grensdiagnose", ""),
                ]
                if clean_display_value(part)
            )

            family = clean_display_value(first.get("project_family", ""))
            situering = clean_display_value(first.get("situering", ""))

            proposal_rows.append(
                {
                    "voorstel_id": proposal_id,
                    "wegnummer": road_label,
                    "axis_id": axis_id,
                    "axis_naam": clean_display_value(first.get("axis_naam", axis_id)),
                    "project_type": project_type,
                    "project_family": family,
                    "situering": situering,
                    "fysiek_begin_m": _round_or_none(start_m, 0),
                    "fysiek_eind_m": _round_or_none(end_m, 0),
                    "fysiek_lengte_m": _round_or_none(segment_length_m, 0),
                    "fysiek_begin_km": _round_or_none(begin_rule.get("km"), 3),
                    "fysiek_eind_km": _round_or_none(end_rule.get("km"), 3),
                    "snap_tolerantie_m": float(boundary_snap_tolerance_m),
                    "begin_dichtstbijzijnde_hm": clean_display_value(begin_rule.get("nearest_hm_label")),
                    "begin_snap_afstand_m": _round_or_none(begin_rule.get("snap_distance_m"), 2),
                    "begin_gesnapt_naar_hm": bool(begin_rule.get("snapped_to_hm", False)),
                    "naam_begin": begin_label,
                    "eind_dichtstbijzijnde_hm": clean_display_value(end_rule.get("nearest_hm_label")),
                    "eind_snap_afstand_m": _round_or_none(end_rule.get("snap_distance_m"), 2),
                    "eind_gesnapt_naar_hm": bool(end_rule.get("snapped_to_hm", False)),
                    "naam_eind": end_label,
                    **begin_diagnostic_columns,
                    **end_diagnostic_columns,
                    "grensdiagnose": combined_grensdiagnose,
                    "onderhoudsproject_voorgesteld": proposed_name,
                    "knipreden_begin": begin_reason,
                    "knipreden_eind": end_reason or "einde spoor",
                    "knipprofiel": "v0.35.4 reeksherkenning/lokale afwijking",
                    "technisch_profiel": technical_profile_text,
                    "bestek_signalen": bestek_signal_text,
                    "datakwaliteit_signalen": data_quality_signal_text,
                    "lokale_afwijkingen": local_deviation_text,
                    "ingesloten_objecten": included_objects_text,
                    "harde_knipsignalen": hard_signal_text,
                    "zachte_signalen": soft_signal_text,
                    "aantal_primaire_objecten": int(len(segment)),
                    "bestaande_onderhoudsprojecten": existing_joined,
                    "vergelijking_iasset_status": comparison_status,
                    "status_voorstel": status,
                    "hoofdmelding": hoofd,
                    "contextmelding": context,
                    "bronkwaliteit": "experimenteel-groenveld",
                }
            )

            for object_idx, object_row in segment.iterrows():
                object_profile = _greenfield_technical_profile(object_row)
                object_role = clean_display_value(object_roles.get(int(object_idx), "normaal"))
                assignment_rows.append(
                    {
                        "voorstel_id": proposal_id,
                        "onderhoudsproject_voorgesteld": proposed_name,
                        "sys_id": clean_display_value(object_row.get("sys_id", "")),
                        "nummer": clean_display_value(object_row.get("nummer", "")),
                        "naam": clean_display_value(object_row.get("naam", "")),
                        "subthema": clean_display_value(object_row.get("subthema", "")),
                        "project_type": project_type,
                        "project_family": family,
                        "situering": situering,
                        "bestaand_onderhoudsproject": clean_display_value(object_row.get("Onderhoudsproject", "")),
                        "axis_id": axis_id,
                        "fysiek_begin_m": _round_or_none(object_row.get("route_start_norm_m"), 0),
                        "fysiek_eind_m": _round_or_none(object_row.get("route_end_norm_m"), 0),
                        "fysiek_begin_km": _round_or_none(object_row.get("referentie_begin_km"), 3),
                        "fysiek_eind_km": _round_or_none(object_row.get("referentie_eind_km"), 3),
                        "Besteknummer": clean_display_value(object_row.get("Besteknummer", "")),
                        "verhardingssoort": clean_display_value(object_row.get("verhardingssoort", "")),
                        "Soort verharding_N": clean_display_value(object_row.get("Soort verharding_N", "")),
                        "Soort deklaag specifiek": clean_display_value(object_row.get("Soort deklaag specifiek", "")),
                        "Jaar aanleg": clean_display_value(object_row.get("Jaar aanleg", "")),
                        "Jaar deklaag": clean_display_value(object_row.get("Jaar deklaag", "")),
                        "Jaar conservering": clean_display_value(object_row.get("Jaar conservering", "")),
                        "Jaar herstrating": clean_display_value(object_row.get("Jaar herstrating", "")),
                        "technisch_profiel": _format_technical_profile(object_profile),
                        "besteknummer_norm": _greenfield_bestek_value(object_row),
                        "lokale_afwijking_type": object_role if object_role != "normaal" else "",
                        "ingesloten_in_voorstel": bool(object_role != "normaal"),
                        "object_kniprol": object_role,
                        "toewijzing_status": clean_display_value(object_row.get("status", "")),
                        "toewijzing_melding": clean_display_value(object_row.get("waarschuwing", "")),
                        "bronkwaliteit": "experimenteel-groenveld",
                    }
                )

            previous_end_reason = end_reason

    proposals = (
        pd.DataFrame(proposal_rows, columns=_empty_project_proposal_frame().columns)
        if proposal_rows
        else _empty_project_proposal_frame()
    )
    assignments = (
        pd.DataFrame(assignment_rows, columns=_empty_proposal_object_assignment_frame().columns)
        if assignment_rows
        else _empty_proposal_object_assignment_frame()
    )
    comparison = _build_proposal_iasset_comparison(proposals, assignments)
    return proposals, assignments, comparison


def _build_proposal_iasset_comparison(proposals: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Vergelijk groenveldvoorstellen achteraf met bestaande iASSET-onderhoudsprojecten."""
    if proposals is None or proposals.empty or assignments is None or assignments.empty:
        return _empty_proposal_iasset_comparison_frame()

    rows: list[dict[str, Any]] = []

    # Niveau 1: per bestaand onderhoudsproject: valt dit volgens de tool in één of meerdere voorstellen?
    assigned_existing = assignments[
        assignments["bestaand_onderhoudsproject"].fillna("").astype(str).str.strip() != ""
    ].copy()
    if not assigned_existing.empty:
        for existing_name, group in assigned_existing.groupby("bestaand_onderhoudsproject", sort=True):
            proposal_ids = _unique_clean_values(group["voorstel_id"])
            proposed_names = _unique_clean_values(group["onderhoudsproject_voorgesteld"])
            object_count = int(len(group))

            if len(proposal_ids) > 1:
                status = "controleer"
                verschil_type = "bestaand project splitst over meerdere voorstellen"
                hoofd = "Bestaand iASSET-project valt volgens groenveldlogica uiteen."
            elif len(proposed_names) == 1 and clean_display_value(existing_name) == proposed_names[0]:
                status = "ok"
                verschil_type = "komt overeen"
                hoofd = "Bestaand iASSET-project komt overeen met groenveldvoorstel."
            else:
                status = "aandacht"
                verschil_type = "naam wijkt af van groenveldvoorstel"
                hoofd = "Bestaande iASSET-naam wijkt af van de groenveldnaamregel."

            rows.append(
                {
                    "vergelijking_niveau": "bestaand_onderhoudsproject",
                    "bestaand_onderhoudsproject": clean_display_value(existing_name),
                    "voorstel_id": " | ".join(proposal_ids),
                    "onderhoudsproject_voorgesteld": " | ".join(proposed_names),
                    "verschil_type": verschil_type,
                    "aantal_objecten": object_count,
                    "status": status,
                    "hoofdmelding": hoofd,
                    "contextmelding": f"{object_count} primaire object(en)",
                    "bronkwaliteit": "experimenteel-groenveld",
                }
            )

    # Niveau 2: per voorstel: bundelt dit voorstel geen, één of meerdere bestaande projecten?
    for _, proposal in proposals.iterrows():
        proposal_id = clean_display_value(proposal.get("voorstel_id", ""))
        if not proposal_id:
            continue
        proposal_objects = assignments[assignments["voorstel_id"].astype(str) == proposal_id]
        existing_projects = _unique_clean_values(proposal_objects.get("bestaand_onderhoudsproject", []))
        proposed_name = clean_display_value(proposal.get("onderhoudsproject_voorgesteld", ""))
        object_count = int(len(proposal_objects))

        if not existing_projects:
            status = "aandacht"
            verschil_type = "nieuw voorstel zonder bestaande iASSET-naam"
            hoofd = "Groenveldvoorstel bevat primaire objecten zonder bestaand onderhoudsproject."
        elif len(existing_projects) > 1:
            status = "controleer"
            verschil_type = "voorstel bundelt meerdere bestaande projecten"
            hoofd = "Groenveldvoorstel bundelt objecten uit meerdere bestaande iASSET-projecten."
        elif existing_projects[0] == proposed_name:
            status = "ok"
            verschil_type = "komt overeen"
            hoofd = "Groenveldvoorstel komt overeen met bestaande iASSET-naam."
        else:
            status = "aandacht"
            verschil_type = "voorgestelde naam wijkt af"
            hoofd = "Groenveldvoorstel heeft een andere naam dan de bestaande iASSET-naam."

        rows.append(
            {
                "vergelijking_niveau": "projectvoorstel",
                "bestaand_onderhoudsproject": " | ".join(existing_projects),
                "voorstel_id": proposal_id,
                "onderhoudsproject_voorgesteld": proposed_name,
                "verschil_type": verschil_type,
                "aantal_objecten": object_count,
                "status": status,
                "hoofdmelding": hoofd,
                "contextmelding": clean_display_value(proposal.get("contextmelding", "")),
                "bronkwaliteit": "experimenteel-groenveld",
            }
        )

    if not rows:
        return _empty_proposal_iasset_comparison_frame()
    return pd.DataFrame(rows, columns=_empty_proposal_iasset_comparison_frame().columns)

def _get_str(row: pd.Series, column: str, default: str = "") -> str:
    """Lees een cel als nette tekst, ook bij ontbrekende kolommen of NaN."""
    if column not in row.index:
        return default
    value = row.get(column, default)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _get_float(row: pd.Series, column: str) -> float | None:
    """Lees een cel als float voor compacte exports."""
    if column not in row.index:
        return None
    value = row.get(column)
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None



def _join_unique_message_parts(parts: Iterable[Any]) -> str:
    """
    Voeg meldingsdelen samen zonder dubbele of lege teksten.

    De compacte controlelijst is bedoeld als werklijst voor databeheerders.
    Daarom houden we de hoofdreden kort en plaatsen we detailcontext apart.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in parts:
        value = clean_display_value(part)
        if not value:
            continue
        for subpart in [item.strip() for item in value.split(";") if item.strip()]:
            key = subpart.lower()
            if key not in seen:
                cleaned.append(subpart)
                seen.add(key)
    return "; ".join(cleaned)


def _boundary_control_messages(boundary: pd.Series) -> tuple[str, str]:
    """
    Splits een projectgrensmelding in hoofdreden en context.

    v0.34.4 zette objectligging soms in dezelfde ``melding`` als de echte
    projectgrensreden. Dat was technisch niet fout, maar onrustig in de compacte
    werklijst. Vanaf v0.34.5 staat objectligging daarom in ``contextmelding``.
    """
    status_projectnaam = _get_str(boundary, "status_projectnaam", "ok").lower() or "ok"
    status_projectgrens = _get_str(boundary, "status_projectgrens", "ok").lower() or "ok"
    naam_melding = _get_str(boundary, "naam_validatie_melding")
    waarschuwing = _get_str(boundary, "waarschuwing")
    object_melding = _get_str(boundary, "objectligging_melding")

    warning_parts = [item.strip() for item in waarschuwing.split(";") if item.strip()]
    boundary_parts = [
        part for part in warning_parts
        if part.lower() != naam_melding.lower() and part
    ]

    # Maak projectgrensredenen expliciet op basis van vaste kolommen. Zo blijft
    # de compacte export leesbaar, ook als de tekst in ``waarschuwing`` later
    # iets wijzigt.
    explicit_boundary_parts: list[str] = []
    if bool(boundary.get("begin_buiten_ijkbereik", False)):
        explicit_boundary_parts.append("begingrens buiten ijkbereik")
    if bool(boundary.get("eind_buiten_ijkbereik", False)):
        explicit_boundary_parts.append("eindgrens buiten ijkbereik")

    begin_zone = _get_str(boundary, "begin_zone_kleur")
    eind_zone = _get_str(boundary, "eind_zone_kleur")
    if begin_zone:
        explicit_boundary_parts.append(f"begingrens in {begin_zone} afwijkingszone")
    if eind_zone:
        explicit_boundary_parts.append(f"eindgrens in {eind_zone} afwijkingszone")

    length_diff = _get_float(boundary, "lengteverschil_naam_vs_as_m")
    if length_diff is not None and math.isfinite(float(length_diff)) and status_projectgrens in {"aandacht", "controleer"}:
        if any("lengteverschil" in part.lower() for part in boundary_parts):
            explicit_boundary_parts.append(
                next(part for part in boundary_parts if "lengteverschil" in part.lower())
            )

    filtered_boundary_parts: list[str] = []
    for part in boundary_parts:
        lower = part.lower()
        if "buiten ijkbereik" in lower and any("buiten ijkbereik" in item.lower() for item in explicit_boundary_parts):
            continue
        if "afwijkingszone" in lower and any("afwijkingszone" in item.lower() for item in explicit_boundary_parts):
            continue
        if "lengteverschil" in lower and any("lengteverschil" in item.lower() for item in explicit_boundary_parts):
            continue
        filtered_boundary_parts.append(part)

    if status_projectnaam in {"aandacht", "controleer"}:
        hoofd = _join_unique_message_parts([
            naam_melding,
            "onderhoudsprojectnaam vraagt controle",
        ])
        context = _join_unique_message_parts([*explicit_boundary_parts, *filtered_boundary_parts, object_melding])
    else:
        hoofd = _join_unique_message_parts([*explicit_boundary_parts, *filtered_boundary_parts])
        context = _join_unique_message_parts([object_melding])

    if not hoofd:
        hoofd = "Controleer deze projectregel in de volledige projectgrensdiagnose."

    return hoofd, context


def build_project_axis_control_export(
    project_boundaries: pd.DataFrame | None,
    project_coverage: pd.DataFrame | None,
    object_ranges: pd.DataFrame | None = None,
    *,
    selected_road: str | None = None,
) -> pd.DataFrame:
    """
    Maak een compacte controlelijst voor databeheerders.

    Waarom?
    De volledige diagnosebestanden blijven belangrijk voor analyse, maar zijn
    breed en technisch. Deze lijst bevat alleen de regels die actie of handmatige
    controle vragen: projectgrenzen met ``aandacht``/``controleer`` en dekking-
    regels zoals overlap of harde gaten.

    Objectligging nemen we bewust niet als zelfstandige hoofdregel op. Die blijft
    detailcontext in ``Projectobjecten_Referentieas_*`` en in de volledige
    projectgrenzen-export, zodat parallelwegen en fietspaden niet onnodig het
    hoofdbeeld domineren.

    v0.34.5 splitst de compacte melding in:
    - ``hoofdmelding``: de reden waarom de regel in de werklijst staat;
    - ``contextmelding``: nuttige detailinformatie, zoals objectligging.
    """
    rows: list[dict[str, Any]] = []
    road_label = selected_road or ""

    if isinstance(project_boundaries, pd.DataFrame) and not project_boundaries.empty:
        for _, boundary in project_boundaries.iterrows():
            status = _get_str(boundary, "status", "ok").lower() or "ok"
            status_projectnaam = _get_str(boundary, "status_projectnaam", "ok").lower() or "ok"
            status_projectgrens = _get_str(boundary, "status_projectgrens", "ok").lower() or "ok"

            if status not in {"aandacht", "controleer"} and status_projectnaam == "ok" and status_projectgrens == "ok":
                continue

            if status_projectnaam in {"aandacht", "controleer"}:
                categorie = "Projectnaam"
            else:
                categorie = "Projectgrens"

            hoofdmelding, contextmelding = _boundary_control_messages(boundary)

            rows.append(
                {
                    "prioriteit": 2 if status == "controleer" else 1,
                    "status": status,
                    "controle_categorie": categorie,
                    "wegnummer": road_label or _get_str(boundary, "naam_wegnummer"),
                    "project_type": _get_str(boundary, "project_type"),
                    "project_family": _get_str(boundary, "project_family"),
                    "situering": _get_str(boundary, "situering"),
                    "Onderhoudsproject": _get_str(boundary, "Onderhoudsproject"),
                    "controlepunt": _get_str(boundary, "Onderhoudsproject"),
                    "hoofdmelding": hoofdmelding,
                    "contextmelding": contextmelding,
                    # Kolom blijft bestaan voor terugwaartse herkenbaarheid, maar is vanaf v0.34.5 schoon:
                    # hij bevat alleen de hoofdreden, niet meer de objectligging-context.
                    "melding": hoofdmelding,
                    "advies": _get_str(boundary, "advies"),
                    "axis_id": _get_str(boundary, "axis_id"),
                    "as_van_m": _get_float(boundary, "as_begin_m"),
                    "as_tot_m": _get_float(boundary, "as_eind_m"),
                    "lengte_m": _get_float(boundary, "as_lengte_m"),
                    "verschil_m": _get_float(boundary, "lengteverschil_naam_vs_as_m"),
                    "bronbestand": "Projectgrenzen_Referentieas",
                }
            )

    if isinstance(project_coverage, pd.DataFrame) and not project_coverage.empty:
        for _, coverage in project_coverage.iterrows():
            status = _get_str(coverage, "status", "ok").lower() or "ok"
            controle_type = _get_str(coverage, "controle_type", "dekking").lower()
            if status not in {"aandacht", "controleer"} and controle_type not in {"gat", "overlap"}:
                continue

            project_links = _get_str(coverage, "project_links")
            project_rechts = _get_str(coverage, "project_rechts")
            if project_links and project_rechts:
                controlepunt = f"{project_links} → {project_rechts}"
            else:
                controlepunt = controle_type

            advies = _get_str(coverage, "advies")
            if controle_type == "overlap":
                hoofdmelding = "Mogelijke overlap binnen hetzelfde projecttype."
            elif controle_type == "gat":
                hoofdmelding = "Mogelijk hard gat met primair areaal binnen hetzelfde projecttype."
            else:
                hoofdmelding = "Controleer dekking op de geijkte referentieas."

            contextmelding = advies

            rows.append(
                {
                    "prioriteit": 2 if status == "controleer" else 1,
                    "status": status,
                    "controle_categorie": "Projectdekking",
                    "wegnummer": road_label,
                    "project_type": _get_str(coverage, "project_type"),
                    "project_family": _get_str(coverage, "project_family"),
                    "situering": _get_str(coverage, "situering"),
                    "Onderhoudsproject": "",
                    "controlepunt": controlepunt,
                    "hoofdmelding": hoofdmelding,
                    "contextmelding": contextmelding,
                    "melding": hoofdmelding,
                    "advies": advies,
                    "axis_id": _get_str(coverage, "axis_id"),
                    "as_van_m": _get_float(coverage, "van_m"),
                    "as_tot_m": _get_float(coverage, "tot_m"),
                    "lengte_m": _get_float(coverage, "hard_gat_lengte_m") or _get_float(coverage, "lengte_m"),
                    "verschil_m": None,
                    "bronbestand": "Projectdekking_Referentieas",
                }
            )

    columns = [
        "prioriteit",
        "status",
        "controle_categorie",
        "wegnummer",
        "project_type",
        "project_family",
        "situering",
        "Onderhoudsproject",
        "controlepunt",
        "hoofdmelding",
        "contextmelding",
        "melding",
        "advies",
        "axis_id",
        "as_van_m",
        "as_tot_m",
        "lengte_m",
        "verschil_m",
        "bronbestand",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(rows)
    result = result.sort_values(
        by=["prioriteit", "controle_categorie", "project_type", "controlepunt"],
        ascending=[False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    return result[columns]


def build_project_axis_summary(
    project_boundaries: pd.DataFrame | None,
    project_coverage: pd.DataFrame | None,
    object_ranges: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Maak een kleine samenvatting voor het scherm en voor snelle acceptatietests.

    De tabel is bewust eenvoudig gehouden: aantallen per controleblok. Daarmee
    kunnen databeheerders snel zien of een weg rustig is of aandacht vraagt,
    zonder direct de brede diagnosebestanden te hoeven openen.
    """
    rows: list[dict[str, Any]] = []

    if isinstance(project_boundaries, pd.DataFrame) and not project_boundaries.empty:
        status_counts = project_boundaries.get("status", pd.Series(dtype="object")).fillna("ok").astype(str).str.lower().value_counts()
        rows.append(
            {
                "onderdeel": "Projectgrenzen",
                "totaal": int(len(project_boundaries)),
                "ok": int(status_counts.get("ok", 0)),
                "aandacht": int(status_counts.get("aandacht", 0)),
                "controleer": int(status_counts.get("controleer", 0)),
                "toelichting": "Hoofdstatus van onderhoudsprojectgrenzen op de geijkte as.",
            }
        )
    else:
        rows.append(
            {
                "onderdeel": "Projectgrenzen",
                "totaal": 0,
                "ok": 0,
                "aandacht": 0,
                "controleer": 0,
                "toelichting": "Geen projectgrenzen berekend.",
            }
        )

    if isinstance(project_coverage, pd.DataFrame) and not project_coverage.empty:
        status_counts = project_coverage.get("status", pd.Series(dtype="object")).fillna("ok").astype(str).str.lower().value_counts()
        overlap_count = int((project_coverage.get("controle_type", pd.Series(dtype="object")).astype(str).str.lower() == "overlap").sum())
        gap_count = int((project_coverage.get("controle_type", pd.Series(dtype="object")).astype(str).str.lower() == "gat").sum())
        rows.append(
            {
                "onderdeel": "Projectdekking",
                "totaal": int(len(project_coverage)),
                "ok": int(status_counts.get("ok", 0)),
                "aandacht": int(status_counts.get("aandacht", 0)),
                "controleer": int(status_counts.get("controleer", 0)),
                "toelichting": f"{gap_count} gaten en {overlap_count} overlaps in de compacte dekkingstabel.",
            }
        )
    else:
        rows.append(
            {
                "onderdeel": "Projectdekking",
                "totaal": 0,
                "ok": 0,
                "aandacht": 0,
                "controleer": 0,
                "toelichting": "Geen gaten of overlaps geëxporteerd.",
            }
        )

    if isinstance(object_ranges, pd.DataFrame) and not object_ranges.empty:
        status_counts = object_ranges.get("status", pd.Series(dtype="object")).fillna("ok").astype(str).str.lower().value_counts()
        rows.append(
            {
                "onderdeel": "Objectprojecties",
                "totaal": int(len(object_ranges)),
                "ok": int(status_counts.get("ok", 0)),
                "aandacht": int(status_counts.get("aandacht", 0)),
                "controleer": int(status_counts.get("controleer", 0)),
                "toelichting": "Detailcontext; bepaalt niet zelfstandig de hoofdstatus.",
            }
        )
    else:
        rows.append(
            {
                "onderdeel": "Objectprojecties",
                "totaal": 0,
                "ok": 0,
                "aandacht": 0,
                "controleer": 0,
                "toelichting": "Geen objectprojecties beschikbaar.",
            }
        )

    return pd.DataFrame(rows)

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

    project_proposals, proposal_object_assignments, proposal_iasset_comparison = _build_project_proposals(
        object_ranges,
        anchors,
        selected_road,
        gap_tolerance_m=float(gap_tolerance_m),
        boundary_snap_tolerance_m=float(boundary_snap_tolerance_m),
    )

    if project_boundaries.empty:
        warnings.append("Geen onderhoudsprojectnamen met herkenbare hm-range gevonden voor projectgrensdiagnose.")
    if project_proposals.empty:
        warnings.append("Geen groenveld-projectvoorstellen gemaakt; controleer primaire objecten en ijking.")

    return ProjectAxisDiagnosticsResult(
        calibration_anchors=anchors,
        project_boundaries=project_boundaries,
        project_coverage=project_coverage,
        object_ranges=object_ranges,
        project_proposals=project_proposals,
        proposal_object_assignments=proposal_object_assignments,
        proposal_iasset_comparison=proposal_iasset_comparison,
        warning=" ".join(dict.fromkeys(part for part in warnings if part)),
    )
