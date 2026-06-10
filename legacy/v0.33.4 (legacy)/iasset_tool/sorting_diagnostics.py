
"""
Sorteerdiagnose voor onderhoudsprojecten.

Deze module maakt zichtbaar waarop de Project Adviseur-volgorde gebaseerd is
en waar de data onvoldoende is voor betrouwbare volgorde binnen hetzelfde
wegvak/metrering.

Waarom apart?
De sortering van onderhoudsprojecten is domeingevoelig: kronkelende wegen,
parallelwegen, meerdere rijstroken en extra knips binnen één hectometervak
kunnen niet betrouwbaar worden opgelost met alleen X/Y-sortering. De diagnose
toont daarom expliciet de primaire ruggengraatroute, alle-object-route en de
feitelijke Project Adviseur-sleutel.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from .config import BACKBONE_TYPES, ROAD_DIRECTIONS
from .utils import clean_display_value, is_empty_value, normalize_text, parse_hm_sort


@dataclass(frozen=True)
class LocalAxisResult:
    """Resultaat van een lokaal afgeleide route-as."""

    axis: LineString | None
    anchor_count: int
    source: str
    warning: str = ""


WEGVAKNUM_ALIASES = ("Wegvaknum", "Wegvaknum V", "Wegvaknum G", "Wegvak")
METRERING_ALIASES = ("Metrering", "Metrering V", "Metrering G")
SITUERING_ALIASES = ("Situering", "Situering V")


def _first_existing_column(gdf: gpd.GeoDataFrame, aliases: Iterable[str]) -> str | None:
    """Geef de eerste beschikbare kolom uit een aliaslijst terug."""
    for column in aliases:
        if column in gdf.columns:
            return column
    return None


def _clean_bucket_value(value: Any) -> str:
    """
    Maak een waarde geschikt als groepeersleutel.

    Lege iASSET-waarden krijgen een expliciete tekst, zodat ontbrekende data
    in de diagnose zichtbaar blijft in plaats van stil weg te vallen.
    """
    text = clean_display_value(value)
    return text if text else "<leeg>"


def _parse_wegvaknum(value: Any) -> float:
    """Zet Wegvaknum veilig om naar een sorteerbaar getal."""
    if is_empty_value(value):
        return 999999.9

    text = clean_display_value(value).replace(",", ".")
    number = pd.to_numeric(text, errors="coerce")
    if pd.isna(number):
        return 999999.9

    try:
        return float(number)
    except (TypeError, ValueError, OverflowError):
        return 999999.9


def _primary_mask(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Selecteer primaire ruggengraatobjecten op basis van subthema_clean."""
    if gdf is None or gdf.empty:
        return pd.Series(dtype=bool)

    if "subthema_clean" in gdf.columns:
        values = gdf["subthema_clean"].map(normalize_text)
    elif "subthema" in gdf.columns:
        values = gdf["subthema"].map(normalize_text)
    else:
        values = pd.Series([""] * len(gdf), index=gdf.index)

    return values.isin({normalize_text(value) for value in BACKBONE_TYPES})


def _direction_axis_value(point: Point, direction_code: str) -> float:
    """
    Geef een stabiele X/Y-fallbackwaarde volgens de bekende globale wegrichting.

    Deze fallback is nadrukkelijk géén goede oplossing voor bochtige wegen; hij
    maakt alleen de diagnose reproduceerbaar wanneer metrering/wegvakdata mist.
    """
    if direction_code == "WTE":
        return float(point.x)
    if direction_code == "ETW":
        return float(-point.x)
    if direction_code == "STN":
        return float(point.y)
    if direction_code == "NTS":
        return float(-point.y)
    return float(point.x)


def _representative_point_safe(geometry) -> Point | None:
    """Geef een representatief punt terug zonder te crashen op lege geometrie."""
    if geometry is None:
        return None

    try:
        if geometry.is_empty:
            return None
        return geometry.representative_point()
    except Exception:
        return None


def _merge_geometry_safe(geometries):
    """Voeg geometrieën samen met fallback voor verschillende GeoPandas-versies."""
    try:
        return geometries.union_all()
    except AttributeError:
        return geometries.unary_union


def build_local_axis(gdf: gpd.GeoDataFrame, selected_road: str | None = None) -> LocalAxisResult:
    """
    Leid een ruwe lokale route-as af uit primaire objecten.

    De as wordt alleen gebruikt voor diagnose. We bouwen ankerpunten per
    wegvak/metrering-bucket, zodat dubbele rijstroken of tegengestelde richtingen
    binnen hetzelfde vak niet direct een zigzaglijn veroorzaken.

    Beperkingen:
    - zonder voldoende metrering of wegvakdata is de as minder betrouwbaar;
    - rotondes en gescheiden rijbanen blijven controlepunten;
    - de Project Adviseur-sortering wordt hier niet aangepast.
    """
    if gdf is None or gdf.empty:
        return LocalAxisResult(axis=None, anchor_count=0, source="geen_data", warning="Geen data beschikbaar.")

    primary = gdf[_primary_mask(gdf)].copy()
    if primary.empty:
        return LocalAxisResult(
            axis=None,
            anchor_count=0,
            source="geen_primaire_objecten",
            warning="Geen primaire ruggengraatobjecten gevonden.",
        )

    wegvak_col = _first_existing_column(primary, WEGVAKNUM_ALIASES)
    metrering_col = _first_existing_column(primary, METRERING_ALIASES)

    if wegvak_col:
        primary["_diag_wegvak_sort"] = primary[wegvak_col].map(_parse_wegvaknum)
        primary["_diag_wegvak_bucket"] = primary[wegvak_col].map(_clean_bucket_value)
    else:
        primary["_diag_wegvak_sort"] = 999999.9
        primary["_diag_wegvak_bucket"] = "<geen_kolom>"

    if "hm_sort" in primary.columns:
        primary["_diag_hm_sort"] = pd.to_numeric(primary["hm_sort"], errors="coerce").fillna(99999.9)
    elif metrering_col:
        primary["_diag_hm_sort"] = primary[metrering_col].map(parse_hm_sort)
    else:
        primary["_diag_hm_sort"] = 99999.9

    primary["_diag_hm_bucket"] = primary["_diag_hm_sort"].round(4).astype(str)

    road_label = selected_road
    if not road_label and "Wegnummer" in primary.columns and not primary.empty:
        road_label = clean_display_value(primary["Wegnummer"].iloc[0])
    direction_code = ROAD_DIRECTIONS.get(str(road_label), "UNKNOWN")

    anchors: list[tuple[float, float, float, Point]] = []
    for (_, _), bucket in primary.groupby(["_diag_wegvak_bucket", "_diag_hm_bucket"], dropna=False):
        valid_geometries = bucket.geometry.dropna()
        if valid_geometries.empty:
            continue

        try:
            merged = _merge_geometry_safe(valid_geometries)
            point = merged.centroid
        except Exception:
            points = [
                candidate
                for geometry in valid_geometries
                if (candidate := _representative_point_safe(geometry)) is not None
            ]
            if not points:
                continue
            x_mid = median([point.x for point in points])
            y_mid = median([point.y for point in points])
            point = Point(x_mid, y_mid)

        if point is None or point.is_empty:
            continue

        wegvak_sort = float(bucket["_diag_wegvak_sort"].min())
        hm_sort = float(bucket["_diag_hm_sort"].min())
        fallback_sort = _direction_axis_value(point, direction_code)
        anchors.append((wegvak_sort, hm_sort, fallback_sort, point))

    anchors.sort(key=lambda item: (item[0], item[1], item[2]))

    unique_points: list[Point] = []
    for _, _, _, point in anchors:
        if not unique_points:
            unique_points.append(point)
            continue

        # Voorkom dubbele opeenvolgende punten. Een LineString met dezelfde
        # punten achter elkaar is gevoelig voor projectiefouten.
        if point.distance(unique_points[-1]) > 0.01:
            unique_points.append(point)

    if len(unique_points) < 2:
        return LocalAxisResult(
            axis=None,
            anchor_count=len(unique_points),
            source="onvoldoende_ankerpunten",
            warning=(
                "Te weinig unieke ankerpunten om een lokale route-as te maken. "
                "Controleer of Metrering/Wegvaknum gevuld zijn."
            ),
        )

    try:
        axis = LineString([(point.x, point.y) for point in unique_points])
    except Exception:
        return LocalAxisResult(
            axis=None,
            anchor_count=len(unique_points),
            source="as_fout",
            warning="Lokale route-as kon niet worden opgebouwd uit de ankerpunten.",
        )

    if axis.length <= 0:
        return LocalAxisResult(
            axis=None,
            anchor_count=len(unique_points),
            source="as_lengte_nul",
            warning="Lokale route-as heeft lengte 0.",
        )

    return LocalAxisResult(axis=axis, anchor_count=len(unique_points), source="primaire_objecten")


def _geometry_sample_points(geometry, max_points: int = 200) -> list[Point]:
    """
    Verzamel punten uit een geometrie voor projectie op de lokale as.

    Voor polygonen gebruiken we grenscoördinaten, omdat begin/einde langs de weg
    meestal beter uit de rand dan uit alleen de centroïde blijkt. Bij complexe
    geometrieën begrenzen we het aantal punten om de diagnose snel te houden.
    """
    if geometry is None:
        return []

    try:
        if geometry.is_empty:
            return []
    except Exception:
        return []

    coords: list[tuple[float, float]] = []

    def add_coords_from_geom(geom) -> None:
        if geom is None:
            return

        geom_type = getattr(geom, "geom_type", "")

        if geom_type == "Point":
            coords.append((geom.x, geom.y))
        elif geom_type in {"LineString", "LinearRing"}:
            coords.extend(list(geom.coords))
        elif geom_type == "Polygon":
            coords.extend(list(geom.exterior.coords))
        elif hasattr(geom, "geoms"):
            for part in geom.geoms:
                add_coords_from_geom(part)

    try:
        add_coords_from_geom(geometry)
    except Exception:
        coords = []

    if not coords:
        point = _representative_point_safe(geometry)
        return [point] if point is not None else []

    if len(coords) > max_points:
        step = max(1, len(coords) // max_points)
        coords = coords[::step][:max_points]

    return [Point(x, y) for x, y in coords]


def _project_geometry_range(geometry, axis: LineString | None) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Projecteer een object op de lokale as.

    Retourneert:
    - startpositie langs de as;
    - midden/medianepositie langs de as;
    - eindpositie langs de as;
    - dwarsafstand vanaf representatief punt tot de as.
    """
    if axis is None or geometry is None:
        return None, None, None, None

    points = _geometry_sample_points(geometry)
    if not points:
        return None, None, None, None

    try:
        positions = [float(axis.project(point)) for point in points]
    except Exception:
        return None, None, None, None

    if not positions:
        return None, None, None, None

    representative = _representative_point_safe(geometry)
    if representative is not None:
        try:
            lateral_offset = float(representative.distance(axis))
        except Exception:
            lateral_offset = None
    else:
        lateral_offset = None

    return min(positions), float(median(positions)), max(positions), lateral_offset


def project_geometry_range_on_axis(
    geometry,
    axis: LineString | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Publieke wrapper voor projectie op de lokale route-as.

    De Project Adviseur gebruikt deze functie vanaf v0.12 als gecontroleerde
    tie-breaker binnen dezelfde hectometrering. De eigenlijke projectiefunctie
    blijft hier centraal staan, zodat diagnose en sortering dezelfde berekening
    gebruiken.
    """
    return _project_geometry_range(geometry, axis)


def _bucket_columns(gdf: gpd.GeoDataFrame) -> tuple[str | None, str | None, str | None]:
    """Geef de kolommen terug die voor binnenvak-diagnose worden gebruikt."""
    return (
        _first_existing_column(gdf, WEGVAKNUM_ALIASES),
        _first_existing_column(gdf, METRERING_ALIASES),
        _first_existing_column(gdf, SITUERING_ALIASES),
    )


def _attention_severity(messages: list[str]) -> str:
    """
    Vertaal diagnoseteksten naar een aparte ernstwaarde.

    Waarom apart?
    De kolom ``sort_warning`` blijft een leesbare toelichting voor de gebruiker,
    maar filtering/export moet niet op losse tekst zoeken. Met ``sort_severity``
    kunnen we echte waarschuwingen scheiden van informatieve aandachtspunten.
    """
    if not messages:
        return ""

    if any(str(message).startswith("WAARSCHUWING") for message in messages):
        return "waarschuwing"

    if any(str(message).startswith("INFO") for message in messages):
        return "info"

    return "aandachtspunt"


ROUTE_SPAN_OUTLIER_THRESHOLD_M = 1000.0
ROUTE_PRIMARY_ALL_DELTA_THRESHOLD_M = 250.0


def _round_or_none(value: Any, digits: int = 2) -> float | None:
    """Rond een getal veilig af of geef ``None`` terug."""
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError, OverflowError):
        return None


def _numeric_values(values: pd.Series) -> pd.Series:
    """Zet een serie tolerant om naar numerieke waarden zonder lege waarden."""
    if values is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(values, errors="coerce").dropna()


def _route_stats_from_projected(projected: pd.DataFrame) -> dict[str, float | None]:
    """
    Vat routeposities van een set objecten samen.

    ``start`` en ``end`` gebruiken de buitenste projectiepunten. ``mid`` gebruikt
    de mediaan van de objectmiddens, zodat één extreem secundair object zichtbaar
    blijft maar niet stil de hele diagnose domineert.
    """
    if projected is None or projected.empty:
        return {"start": None, "mid": None, "end": None, "span": None}

    starts = _numeric_values(projected.get("route_start_m", pd.Series(dtype=float)))
    mids = _numeric_values(projected.get("route_mid_m", pd.Series(dtype=float)))
    ends = _numeric_values(projected.get("route_end_m", pd.Series(dtype=float)))

    route_start = float(starts.min()) if not starts.empty else None
    route_mid = float(mids.median()) if not mids.empty else None
    route_end = float(ends.max()) if not ends.empty else None

    if route_start is not None and route_end is not None:
        route_span = max(0.0, route_end - route_start)
    else:
        route_span = None

    return {
        "start": route_start,
        "mid": route_mid,
        "end": route_end,
        "span": route_span,
    }


def _sort_value_from_route_stats(
    route_stats: dict[str, float | None],
    route_sort_bron: str,
) -> float | None:
    """
    Bereken de route-sortwaarde uit een diagnose-samenvatting.

    De Project Adviseur gebruikt normaal ``route_mid_m`` en binnen
    overlapclusters meestal ``route_start_m``. Door dezelfde bronnaam toe te
    passen op primaire en alle-object-routevelden kunnen we eerlijk vergelijken
    zonder de sortering zelf te wijzigen.
    """
    source = normalize_text(route_sort_bron)

    if source == "route_start_m":
        return route_stats.get("start")
    if source == "route_end_m":
        return route_stats.get("end")

    # Default bewust op mid: dit is de normale Project Adviseur-sorteerwaarde.
    return route_stats.get("mid")


def _route_delta(left: float | None, right: float | None) -> float | None:
    """Geef het absolute verschil tussen twee routewaarden terug."""
    if left is None or right is None:
        return None
    return abs(float(left) - float(right))


def _route_outlier_messages(
    *,
    hm_min: float | None,
    hm_max: float | None,
    route_start: float | None,
    route_mid: float | None,
    route_end: float | None,
    primary_route_sort: float | None,
    all_route_sort: float | None,
    route_basis: str,
) -> list[str]:
    """
    Signaleer route-anomalieën zonder de sortering te wijzigen.

    Dit is bewust diagnose. Grote sprongen kunnen ontstaan door secundaire
    objecten, foutieve geometrie of route-asprojectie rond kruisingen/rotondes.
    De databeheerder moet de oorzaak kunnen controleren voordat v0.15 de
    primaire ruggengraat als harde sorteersleutel gebruikt.
    """
    messages: list[str] = []

    start_mid_delta = _route_delta(route_start, route_mid)
    mid_end_delta = _route_delta(route_mid, route_end)
    route_span = _route_delta(route_start, route_end)

    hm_span = None
    if hm_min is not None and hm_max is not None:
        hm_span = abs(float(hm_max) - float(hm_min))

    compact_hm = hm_span is None or hm_span <= 0.2

    if (
        compact_hm
        and (
            (start_mid_delta is not None and start_mid_delta > ROUTE_SPAN_OUTLIER_THRESHOLD_M)
            or (mid_end_delta is not None and mid_end_delta > ROUTE_SPAN_OUTLIER_THRESHOLD_M)
            or (route_span is not None and route_span > ROUTE_SPAN_OUTLIER_THRESHOLD_M)
        )
    ):
        messages.append(
            "WAARSCHUWING: groot verschil tussen route_start_m, route_mid_m en/of route_end_m "
            "binnen een compact hm-bereik; controleer objectgeometrie of gekoppelde objecten"
        )

    primary_all_delta = _route_delta(primary_route_sort, all_route_sort)
    if (
        route_basis == "primary_ids"
        and primary_all_delta is not None
        and primary_all_delta > ROUTE_PRIMARY_ALL_DELTA_THRESHOLD_M
    ):
        messages.append(
            "INFO: primaire-route en alle-object-route verschillen duidelijk; "
            "controleer of secundaire objecten de groepsroute vertekenen"
        )

    return messages


def _value_counts_as_text(values: pd.Series) -> str:
    """Maak een compacte, stabiele lijst van unieke waarden met aantallen."""
    cleaned = [normalize_text(value) for value in values if clean_display_value(value)]
    if not cleaned:
        return ""

    counts: dict[str, int] = {}
    for value in cleaned:
        counts[value] = counts.get(value, 0) + 1

    return ", ".join(
        f"{value} ({count})" if count > 1 else value
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def _dominant_subthema(primary_subset: gpd.GeoDataFrame, subset: gpd.GeoDataFrame, group_data: dict[str, Any]) -> str:
    """
    Bepaal het meest representatieve subthema van een adviesgroep.

    In de parallel-laag zitten parallelweg, landbouwpad en busbaan samen. Het
    oude label gebruikte dan vaak blind ``parallelweg`` terwijl de groep in de
    praktijk bijvoorbeeld alleen een landbouwpad bevatte. Voor diagnose en UI is
    het veiliger om het dominante primaire subthema uit de data te tonen.
    """
    source = primary_subset if primary_subset is not None and not primary_subset.empty else subset

    if source is not None and not source.empty and "subthema_clean" in source.columns:
        values = [normalize_text(value) for value in source["subthema_clean"] if normalize_text(value)]
        if values:
            counts: dict[str, int] = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    return normalize_text(group_data.get("subthema", "")) or clean_display_value(group_data.get("subthema", ""))


def _project_prefix_for_subthema(subthema_clean: str) -> str:
    """Vertaal een primair subthema naar het gebruikelijke onderhoudsprojectprefix."""
    mapping = {
        "rijstrook": "HRB",
        "parallelweg": "PW",
        "landbouwpad": "LBP",
        "busbaan": "BB",
        "fietspad": "FP",
    }
    return mapping.get(normalize_text(subthema_clean), "")


def _fietspad_classification_text(group_data: dict[str, Any], dominant_subthema: str, subset: gpd.GeoDataFrame) -> str:
    """
    Geef een korte diagnose van de fietspadlogica voor een groep.

    De Project Adviseur koppelt haakse/rotonde-/kruispuntfietspaden aan het
    hoofdproject, terwijl parallelfietspaden een eigen groep kunnen vormen. Deze
    tekst maakt zichtbaar waarom er minder fietspadgroepen zijn dan fietspad-
    objecten in de brondata.
    """
    fietspad_count = 0
    if subset is not None and not subset.empty and "subthema_clean" in subset.columns:
        fietspad_count = int((subset["subthema_clean"].map(normalize_text) == "fietspad").sum())

    attached_count = len(group_data.get("attached_fietspad_ids", []) or [])

    if dominant_subthema == "fietspad":
        if group_data.get("review_needed"):
            return "fietspad eigen project: classificatie onzeker, handmatig controleren"
        return "parallelfietspad / eigen onderhoudsproject"

    if attached_count:
        return f"{attached_count} fietspadobject(en) gekoppeld aan hoofdproject"

    if fietspad_count:
        return f"{fietspad_count} fietspadobject(en) in groep; controleer classificatie"

    return ""


def build_sort_diagnostics(
    gdf: gpd.GeoDataFrame,
    groups: dict[str, dict[str, Any]] | None = None,
    *,
    selected_road: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, LocalAxisResult]:
    """
    Bouw object- en groepsdiagnose voor de huidige sortering.

    De functie is fouttolerant: ontbrekende kolommen leveren lege/diagnostische
    waarden op in plaats van crashes. Daardoor blijft de diagnose bruikbaar bij
    wisselende iASSET-exportbestanden.
    """
    empty_objects = pd.DataFrame(
        columns=[
            "sys_id",
            "Wegnummer",
            "nummer",
            "subthema",
            "is_primair",
            "Wegvaknum",
            "Metrering",
            "Situering",
            "hm_sort",
            "bucket_count",
            "route_start_m",
            "route_mid_m",
            "route_end_m",
            "object_route_start_m",
            "object_route_mid_m",
            "object_route_end_m",
            "object_route_span_m",
            "object_route_outlier",
            "dwarsafstand_m",
            "sort_severity",
            "sort_warning",
        ]
    )
    empty_groups = pd.DataFrame(
        columns=[
            "volgorde_nr",
            "groep",
            "laag",
            "dominant_subthema",
            "subthema_lijst",
            "project_prefix",
            "fietspad_classificatie",
            "rank",
            "objecten",
            "primaire_objecten",
            "sort_mode",
            "sort_quality",
            "hm_min",
            "hm_max",
            "hm_overlap_vorige",
            "route_terugval_vorige",
            "route_mid_terugval_vorige",
            "route_sort_terugval_vorige",
            "routepositie_onderscheidend",
            "route_basis",
            "route_sort_m",
            "route_sort_bron",
            "route_sort_verklaarbaar",
            "advisor_sort_m",
            "advisor_sort_raw_m",
            "advisor_sort_correctie",
            "advisor_sort_basis",
            "advisor_sort_fallback_m",
            "advisor_sort_terugval_vorige",
            "primary_route_start_m",
            "primary_route_mid_m",
            "primary_route_end_m",
            "primary_route_sort_m",
            "primary_route_span_m",
            "all_route_start_m",
            "all_route_mid_m",
            "all_route_end_m",
            "all_route_sort_m",
            "all_route_span_m",
            "primary_all_route_delta_m",
            "route_span_warning",
            "route_outlier_warning",
            "hm_route_conflict",
            "hm_route_conflict_resolved",
            "route_conflict_cluster_id",
            "route_conflict_sort_applied",
            "route_conflict_cluster_size",
            "fallback_sort_m",
            "overlap_cluster_id",
            "overlap_sort_applied",
            "overlap_cluster_size",
            "route_start_m",
            "route_mid_m",
            "route_end_m",
            "tie_breaker",
            "tie_breaker_source",
            "waarschuwing",
        ]
    )

    if gdf is None or gdf.empty:
        return empty_objects, empty_groups, LocalAxisResult(axis=None, anchor_count=0, source="geen_data")

    axis_result = build_local_axis(gdf, selected_road=selected_road)
    axis = axis_result.axis

    wegvak_col, metrering_col, situering_col = _bucket_columns(gdf)

    working = gdf.copy()
    working["_diag_is_primary"] = _primary_mask(working)

    if wegvak_col:
        working["_diag_wegvak"] = working[wegvak_col].map(_clean_bucket_value)
    else:
        working["_diag_wegvak"] = "<geen_kolom>"

    if "hm_sort" in working.columns:
        working["_diag_hm"] = pd.to_numeric(working["hm_sort"], errors="coerce").fillna(99999.9)
    elif metrering_col:
        working["_diag_hm"] = working[metrering_col].map(parse_hm_sort)
    else:
        working["_diag_hm"] = 99999.9

    if metrering_col:
        working["_diag_metrering"] = working[metrering_col].map(_clean_bucket_value)
    else:
        working["_diag_metrering"] = "<geen_kolom>"

    if situering_col:
        working["_diag_situering"] = working[situering_col].map(_clean_bucket_value)
    else:
        working["_diag_situering"] = "<geen_kolom>"

    bucket_key_columns = ["_diag_is_primary", "subthema_clean", "_diag_wegvak", "_diag_hm", "_diag_situering"]
    if "subthema_clean" not in working.columns:
        if "subthema" in working.columns:
            working["subthema_clean"] = working["subthema"].map(normalize_text)
        else:
            working["subthema_clean"] = ""

    bucket_counts = working.groupby(bucket_key_columns, dropna=False).size().rename("_diag_bucket_count")
    working = working.join(bucket_counts, on=bucket_key_columns)

    # Projecteer alle objecten één keer op de lokale route-as. Daardoor kunnen
    # we duplicaten binnen hetzelfde wegvak/metrering niet alleen signaleren,
    # maar ook beoordelen of de routepositie ze echt van elkaar onderscheidt.
    projected_ranges = {
        object_id: _project_geometry_range(row.geometry, axis)
        for object_id, row in working.iterrows()
    }
    working["_diag_route_start"] = [projected_ranges[idx][0] for idx in working.index]
    working["_diag_route_mid"] = [projected_ranges[idx][1] for idx in working.index]
    working["_diag_route_end"] = [projected_ranges[idx][2] for idx in working.index]
    working["_diag_lateral"] = [projected_ranges[idx][3] for idx in working.index]

    duplicate_route_distinguishes: dict[tuple[Any, ...], bool] = {}
    duplicate_route_warning: dict[tuple[Any, ...], str] = {}

    for bucket_key, bucket in working.groupby(bucket_key_columns, dropna=False):
        if len(bucket) <= 1:
            continue

        route_values = pd.to_numeric(bucket["_diag_route_mid"], errors="coerce").dropna()
        if axis is None or len(route_values) < 2:
            duplicate_route_distinguishes[bucket_key] = False
            duplicate_route_warning[bucket_key] = "routepositie niet beschikbaar"
            continue

        route_spread = float(route_values.max() - route_values.min())
        if route_spread > 1.0:
            duplicate_route_distinguishes[bucket_key] = True
            duplicate_route_warning[bucket_key] = "lokale routepositie onderscheidt binnenvakvolgorde"
        else:
            duplicate_route_distinguishes[bucket_key] = False
            duplicate_route_warning[bucket_key] = "routepositie niet onderscheidend"

    object_rows: list[dict[str, Any]] = []
    for object_id, row in working.iterrows():
        start_m = row["_diag_route_start"]
        mid_m = row["_diag_route_mid"]
        end_m = row["_diag_route_end"]
        lateral_m = row["_diag_lateral"]
        object_route_span = _route_delta(start_m, end_m)
        object_route_outlier = (
            object_route_span is not None
            and object_route_span > ROUTE_SPAN_OUTLIER_THRESHOLD_M
        )

        warnings: list[str] = []
        if row["_diag_hm"] >= 90000:
            warnings.append("WAARSCHUWING: mist metrering of metrering is ongeldig")
        if row["_diag_wegvak"] in {"<leeg>", "<geen_kolom>"}:
            warnings.append("WAARSCHUWING: mist wegvak")
        if int(row.get("_diag_bucket_count", 1)) > 1:
            row_bucket_key = tuple(row[column] for column in bucket_key_columns)
            route_note = duplicate_route_warning.get(row_bucket_key, "binnenvakvolgorde vraagt controle")

            if bool(row["_diag_is_primary"]):
                prefix = "INFO" if duplicate_route_distinguishes.get(row_bucket_key, False) else "WAARSCHUWING"
                warnings.append(
                    f"{prefix}: meerdere primaire objecten in zelfde wegvak/metrering/situering (meerdere objecten); {route_note}"
                )
            else:
                warnings.append(
                    f"INFO: meerdere secundaire objecten in zelfde wegvak/metrering/situering (meerdere objecten); {route_note}"
                )
        if object_route_outlier:
            if bool(row["_diag_is_primary"]):
                warnings.append(
                    "WAARSCHUWING: primair object heeft extreem grote route-span; "
                    "controleer geometrie of routeprojectie"
                )
            else:
                warnings.append(
                    "AANDACHTSPUNT: secundair object heeft extreem grote route-span; "
                    "controleer koppeling aan onderhoudscomplex"
                )
        if axis is None:
            warnings.append("WAARSCHUWING: geen lokale route-as")

        object_rows.append(
            {
                "sys_id": object_id,
                "Wegnummer": clean_display_value(row.get("Wegnummer", "")),
                "nummer": clean_display_value(row.get("nummer", "")),
                "subthema": clean_display_value(row.get("subthema", "")),
                "is_primair": bool(row["_diag_is_primary"]),
                "Wegvaknum": row["_diag_wegvak"],
                "Metrering": row["_diag_metrering"],
                "Situering": row["_diag_situering"],
                "hm_sort": None if row["_diag_hm"] >= 90000 else float(row["_diag_hm"]),
                "bucket_count": int(row.get("_diag_bucket_count", 1)),
                "route_start_m": _round_or_none(start_m),
                "route_mid_m": _round_or_none(mid_m),
                "route_end_m": _round_or_none(end_m),
                "object_route_start_m": _round_or_none(start_m),
                "object_route_mid_m": _round_or_none(mid_m),
                "object_route_end_m": _round_or_none(end_m),
                "object_route_span_m": _round_or_none(object_route_span),
                "object_route_outlier": bool(object_route_outlier),
                "dwarsafstand_m": _round_or_none(lateral_m),
                "sort_severity": _attention_severity(warnings),
                "sort_warning": "; ".join(warnings),
            }
        )

    object_df = pd.DataFrame(object_rows)
    if not object_df.empty:
        object_df = object_df.sort_values(
            by=["is_primair", "Wegvaknum", "hm_sort", "route_mid_m", "sys_id"],
            ascending=[False, True, True, True, True],
            na_position="last",
        ).reset_index(drop=True)

    group_rows: list[dict[str, Any]] = []
    groups = groups or {}

    for volgorde_nr, (group_id, group_data) in enumerate(groups.items(), start=1):
        ids = [object_id for object_id in group_data.get("ids", []) if object_id in working.index]
        primary_ids = [object_id for object_id in group_data.get("primary_ids", []) if object_id in working.index]

        if not ids:
            continue

        subset = working.loc[ids]
        primary_subset = working.loc[primary_ids] if primary_ids else subset[subset["_diag_is_primary"]]

        dominant_subthema = _dominant_subthema(primary_subset, subset, group_data)
        subthema_lijst = _value_counts_as_text(primary_subset.get("subthema_clean", pd.Series(dtype=str)))
        layer_label = clean_display_value(group_data.get("layer_label", ""))
        if not layer_label:
            layer_label = clean_display_value(group_data.get("subthema", ""))

        hm_values = pd.to_numeric(primary_subset["_diag_hm"], errors="coerce")
        valid_hm = hm_values[hm_values < 90000]

        all_projected = object_df[object_df["sys_id"].isin(ids)]
        primary_projected = object_df[object_df["sys_id"].isin(primary_ids)] if primary_ids else object_df.iloc[0:0]
        projected = primary_projected if not primary_projected.empty else all_projected
        route_values = _numeric_values(projected.get("route_mid_m", pd.Series(dtype=float)))

        # Gebruik voor de groepsdiagnose bij voorkeur exact dezelfde routewaarden
        # als de Project Adviseur heeft gebruikt. Daarnaast tonen we expliciet
        # de primaire route én de alle-object-route, zodat secundaire uitschieters
        # zichtbaar blijven nu v0.15 de Project Adviseur-sleutel expliciet maakt.
        route_start_from_group = group_data.get("route_start_m", None)
        route_mid_from_group = group_data.get("route_mid_m", None)
        route_end_from_group = group_data.get("route_end_m", None)
        route_sort_from_group = group_data.get("route_sort_m", None)
        route_sort_bron = clean_display_value(group_data.get("route_sort_bron", ""))
        fallback_sort = group_data.get("fallback_sort_m", group_data.get("fallback_tie_breaker_dist", None))
        advisor_sort_from_group = group_data.get("advisor_sort_m", route_sort_from_group)
        advisor_sort_raw_from_group = group_data.get("advisor_sort_raw_m", advisor_sort_from_group)
        advisor_sort_correctie = clean_display_value(group_data.get("advisor_sort_correctie", ""))
        advisor_sort_basis = clean_display_value(group_data.get("advisor_sort_basis", ""))
        advisor_sort_fallback = group_data.get("advisor_sort_fallback_m", fallback_sort)

        route_basis = "primary_ids" if primary_ids else "all_ids_fallback"
        primary_route_stats = _route_stats_from_projected(primary_projected)
        all_route_stats = _route_stats_from_projected(all_projected)

        duplicate_primary = primary_subset[primary_subset.get("_diag_bucket_count", pd.Series(dtype=int)) > 1]
        has_duplicate_bucket = not duplicate_primary.empty

        duplicate_keys = {
            tuple(row[column] for column in bucket_key_columns)
            for _, row in duplicate_primary.iterrows()
        }
        duplicate_route_ok = [
            duplicate_route_distinguishes.get(bucket_key, False)
            for bucket_key in duplicate_keys
        ]
        has_non_distinguishing_duplicate = bool(duplicate_route_ok) and not all(duplicate_route_ok)

        sort_mode = clean_display_value(group_data.get("sort_mode", ""))
        tie_breaker = group_data.get("tie_breaker_dist", None)
        tie_breaker_source = clean_display_value(group_data.get("tie_breaker_source", ""))

        warnings: list[str] = []
        if valid_hm.empty:
            sort_quality = "laag"
            warnings.append("WAARSCHUWING: geen geldige metrering in primaire objecten")
        elif has_non_distinguishing_duplicate:
            sort_quality = "middel"
            warnings.append(
                "WAARSCHUWING: dubbele primaire objecten binnen wegvak/metrering/situering; "
                "lokale routepositie onderscheidt de binnenvakvolgorde niet betrouwbaar"
            )
        elif has_duplicate_bucket:
            sort_quality = "middel"
            warnings.append(
                "INFO: meerdere primaire objecten binnen hetzelfde wegvak/metrering/situering; "
                "lokale routepositie lijkt bruikbaar als toekomstige tie-breaker"
            )
        else:
            sort_quality = "hoog"

        if sort_mode == "axis":
            sort_quality = "laag"
            warnings.append("WAARSCHUWING: huidige volgorde gebruikt globale asfallback")
        elif sort_mode == "hm_route" and has_duplicate_bucket:
            warnings.append("INFO: huidige Project Adviseur gebruikt lokale route-as als tie-breaker binnen dezelfde metrering")
        elif sort_mode == "hm" and has_duplicate_bucket:
            warnings.append("INFO: huidige Project Adviseur gebruikt nog globale X/Y-tie-breaker, niet de lokale route-as")

        if axis is None:
            warnings.append("lokale route-as niet beschikbaar")

        def _diag_float(value):
            try:
                if pd.isna(value):
                    return None
                return float(value)
            except (TypeError, ValueError, OverflowError):
                return None

        route_start = _diag_float(route_start_from_group)
        if route_start is None:
            route_start = route_values.min() if not route_values.empty else None

        route_mid = _diag_float(route_mid_from_group)
        if route_mid is None:
            route_mid = route_values.median() if not route_values.empty else None

        route_end = _diag_float(route_end_from_group)
        if route_end is None:
            route_end = route_values.max() if not route_values.empty else None

        route_sort = _diag_float(route_sort_from_group)
        fallback_sort_value = _diag_float(fallback_sort)
        advisor_sort_value = _diag_float(advisor_sort_from_group)
        advisor_sort_raw_value = _diag_float(advisor_sort_raw_from_group)
        advisor_sort_fallback_value = _diag_float(advisor_sort_fallback)
        if advisor_sort_value is None and route_sort is not None:
            advisor_sort_value = route_sort
        if advisor_sort_raw_value is None:
            advisor_sort_raw_value = advisor_sort_value
        if not advisor_sort_correctie:
            advisor_sort_correctie = "geen_correctie"
        if not advisor_sort_basis:
            advisor_sort_basis = "primary_route_sort_m" if route_sort is not None else "globale_richting_fallback"

        if route_sort is not None and route_start is not None and route_end is not None:
            route_sort_verklaarbaar = min(route_start, route_end) - 0.01 <= route_sort <= max(route_start, route_end) + 0.01
        else:
            route_sort_verklaarbaar = False

        if route_sort is not None and not route_sort_bron:
            route_sort_bron = "route_mid_m"

        primary_route_sort = _sort_value_from_route_stats(primary_route_stats, route_sort_bron)
        all_route_sort = _sort_value_from_route_stats(all_route_stats, route_sort_bron)
        primary_all_route_delta = _route_delta(primary_route_sort, all_route_sort)

        hm_min_value = round(float(valid_hm.min()), 3) if not valid_hm.empty else None
        hm_max_value = round(float(valid_hm.max()), 3) if not valid_hm.empty else None

        route_outlier_messages = _route_outlier_messages(
            hm_min=hm_min_value,
            hm_max=hm_max_value,
            route_start=route_start,
            route_mid=route_mid,
            route_end=route_end,
            primary_route_sort=primary_route_sort,
            all_route_sort=all_route_sort,
            route_basis=route_basis,
        )
        warnings.extend(route_outlier_messages)
        route_outlier_warning = "; ".join(dict.fromkeys(route_outlier_messages))
        route_span_warning = next(
            (
                message
                for message in route_outlier_messages
                if "groot verschil tussen route_start_m" in message
            ),
            "",
        )

        group_rows.append(
            {
                "volgorde_nr": group_data.get("volgorde_nr", volgorde_nr),
                "groep": group_id,
                "laag": layer_label,
                "dominant_subthema": dominant_subthema,
                "subthema_lijst": subthema_lijst,
                "project_prefix": _project_prefix_for_subthema(dominant_subthema),
                "fietspad_classificatie": _fietspad_classification_text(group_data, dominant_subthema, subset),
                "rank": group_data.get("rank", ""),
                "objecten": len(ids),
                "primaire_objecten": len(primary_ids) if primary_ids else int(subset["_diag_is_primary"].sum()),
                "sort_mode": sort_mode or "?",
                "sort_quality": sort_quality,
                "hm_min": hm_min_value,
                "hm_max": hm_max_value,
                # Deze diagnosevelden worden na het opbouwen van alle groepen
                # gevuld, omdat ze afhangen van de vorige groep in de zichtbare volgorde.
                "hm_overlap_vorige": False,
                "route_terugval_vorige": False,
                "route_mid_terugval_vorige": False,
                "route_sort_terugval_vorige": False,
                "routepositie_onderscheidend": bool(group_data.get("routepositie_onderscheidend", True)),
                "route_basis": route_basis,
                "route_sort_m": _round_or_none(route_sort),
                "route_sort_bron": route_sort_bron,
                "route_sort_verklaarbaar": bool(route_sort_verklaarbaar),
                "advisor_sort_m": _round_or_none(advisor_sort_value),
                "advisor_sort_raw_m": _round_or_none(advisor_sort_raw_value),
                "advisor_sort_correctie": advisor_sort_correctie,
                "advisor_sort_basis": advisor_sort_basis,
                "advisor_sort_fallback_m": _round_or_none(advisor_sort_fallback_value),
                "advisor_sort_terugval_vorige": False,
                "primary_route_start_m": _round_or_none(primary_route_stats.get("start")),
                "primary_route_mid_m": _round_or_none(primary_route_stats.get("mid")),
                "primary_route_end_m": _round_or_none(primary_route_stats.get("end")),
                "primary_route_sort_m": _round_or_none(primary_route_sort),
                "primary_route_span_m": _round_or_none(primary_route_stats.get("span")),
                "all_route_start_m": _round_or_none(all_route_stats.get("start")),
                "all_route_mid_m": _round_or_none(all_route_stats.get("mid")),
                "all_route_end_m": _round_or_none(all_route_stats.get("end")),
                "all_route_sort_m": _round_or_none(all_route_sort),
                "all_route_span_m": _round_or_none(all_route_stats.get("span")),
                "primary_all_route_delta_m": _round_or_none(primary_all_route_delta),
                "route_span_warning": route_span_warning,
                "route_outlier_warning": route_outlier_warning,
                "hm_route_conflict": False,
                "hm_route_conflict_resolved": bool(group_data.get("hm_route_conflict_resolved", False)),
                "route_conflict_cluster_id": clean_display_value(group_data.get("route_conflict_cluster_id", "")),
                "route_conflict_sort_applied": bool(group_data.get("route_conflict_sort_applied", False)),
                "route_conflict_cluster_size": int(group_data.get("route_conflict_cluster_size", 1) or 1),
                "fallback_sort_m": _round_or_none(fallback_sort_value),
                "overlap_cluster_id": clean_display_value(group_data.get("overlap_cluster_id", "")),
                "overlap_sort_applied": bool(group_data.get("overlap_sort_applied", False)),
                "overlap_cluster_size": int(group_data.get("overlap_cluster_size", 1) or 1),
                "route_start_m": _round_or_none(route_start),
                "route_mid_m": _round_or_none(route_mid),
                "route_end_m": _round_or_none(route_end),
                "tie_breaker": _round_or_none(tie_breaker),
                "tie_breaker_source": tie_breaker_source,
                "waarschuwing": "; ".join(dict.fromkeys(warnings)),
            }
        )

    group_df = pd.DataFrame(group_rows, columns=empty_groups.columns)
    if not group_df.empty:
        # Bewaar bewust de volgorde waarin de Project Adviseur de groepen aanlevert.
        # Groepsnummers zoals GRP_RIJBAAN_1 kunnen historisch/technisch zijn; de
        # kolom volgorde_nr is de veilige zichtbare volgorde voor de gebruiker.
        group_df = group_df.sort_values(by=["volgorde_nr"], na_position="last").reset_index(drop=True)

        # Diagnoseer per laag of de huidige volgorde opvallende sprongen bevat.
        # Dit verandert de sortering niet; het maakt alleen zichtbaar waar
        # overlapcluster-sortering de volgorde moet verklaren.
        route_duplicate_keys: set[tuple[Any, Any, Any]] = set()
        if "advisor_sort_m" in group_df.columns:
            route_key_column = "advisor_sort_m"
        elif "route_sort_m" in group_df.columns:
            route_key_column = "route_sort_m"
        else:
            route_key_column = "route_mid_m"
        for (rank_value, hm_value), part in group_df.groupby(["rank", "hm_min"], dropna=False):
            rounded_routes = pd.to_numeric(part[route_key_column], errors="coerce").round(2)
            duplicated = rounded_routes.duplicated(keep=False)
            for index in part.index[duplicated]:
                route_duplicate_keys.add((group_df.at[index, "rank"], group_df.at[index, "hm_min"], group_df.at[index, route_key_column]))

        previous_by_rank: dict[Any, pd.Series] = {}
        for index, row in group_df.iterrows():
            rank_value = row.get("rank", "")
            previous = previous_by_rank.get(rank_value)
            warnings = [text for text in str(row.get("waarschuwing", "")).split("; ") if text]

            route_key = (row.get("rank"), row.get("hm_min"), row.get(route_key_column))
            route_is_distinguishing = route_key not in route_duplicate_keys
            if bool(row.get("routepositie_onderscheidend", True)) is False:
                route_is_distinguishing = False
            group_df.at[index, "routepositie_onderscheidend"] = bool(route_is_distinguishing)

            if not bool(row.get("route_sort_verklaarbaar", False)) and pd.notna(row.get("route_sort_m")):
                warnings.append(
                    "WAARSCHUWING: route_sort_m ligt buiten route_start_m/route_end_m; "
                    "controleer diagnose of asprojectie"
                )

            if row.get("tie_breaker_source") in {"lokale_route_as", "lokale_route_as_overlapcluster"} and not route_is_distinguishing:
                group_df.at[index, "tie_breaker_source"] = "stabiele_fallback"
                warnings.append(
                    "WAARSCHUWING: lokale routepositie is niet onderscheidend; "
                    "stabiele fallback bepaalt de onderlinge volgorde"
                )
            elif row.get("tie_breaker_source") in {"globale_as_fallback", "globale_richting_fallback"}:
                group_df.at[index, "tie_breaker_source"] = "globale_richting_fallback"
            elif not row.get("tie_breaker_source"):
                group_df.at[index, "tie_breaker_source"] = "stabiele_fallback"

            if bool(row.get("overlap_sort_applied", False)):
                if row.get("tie_breaker_source") == "stabiele_fallback":
                    warnings.append(
                        "INFO: overlapcluster-sortering actief, maar binnen deze groep is de lokale routepositie niet onderscheidend"
                    )
                else:
                    warnings.append(
                        "INFO: overlapcluster-sortering actief: lokale routepositie gebruikt in plaats van alleen hm_min"
                    )

            if clean_display_value(row.get("advisor_sort_correctie", "")) not in {"", "geen_correctie"}:
                warnings.append(
                    "INFO: Project Adviseur-sorteersleutel gecorrigeerd omdat route_start_m "
                    "een uitschieter is binnen een compact hm-bereik"
                )
                if group_df.at[index, "sort_quality"] == "hoog":
                    group_df.at[index, "sort_quality"] = "middel"

            if bool(row.get("route_conflict_sort_applied", False)):
                group_df.at[index, "hm_route_conflict"] = True
                warnings.append(
                    "INFO: hm/route-conflictcluster is op lokale routepositie hersorteerd"
                )
                if group_df.at[index, "sort_quality"] == "hoog":
                    group_df.at[index, "sort_quality"] = "middel"

            if previous is not None:
                hm_min = row.get("hm_min")
                hm_max = row.get("hm_max")
                prev_hm_min = previous.get("hm_min")
                prev_hm_max = previous.get("hm_max")
                if pd.notna(hm_min) and pd.notna(hm_max) and pd.notna(prev_hm_min) and pd.notna(prev_hm_max):
                    # Alleen echte overlap of exact dezelfde start-hectometrering is
                    # verdacht. Twee groepen die netjes aansluiten op hetzelfde
                    # eind-/beginpunt (bijv. 0.4-0.8 na 0.2-0.4) zijn normaal.
                    hm_overlap = (
                        float(hm_min) < float(prev_hm_max) - 0.0001
                        or abs(float(hm_min) - float(prev_hm_min)) <= 0.0001
                    )
                    group_df.at[index, "hm_overlap_vorige"] = bool(hm_overlap)
                    if hm_overlap:
                        warnings.append(
                            "INFO: hm-bereik overlapt met vorige groep in dezelfde laag; "
                            "controleer of lokale routepositie de volgorde moet verklaren"
                        )

                route_mid = row.get("route_mid_m")
                prev_route_mid = previous.get("route_mid_m")
                route_sort = row.get("route_sort_m")
                prev_route_sort = previous.get("route_sort_m")
                advisor_sort = row.get("advisor_sort_m")
                prev_advisor_sort = previous.get("advisor_sort_m")

                advisor_sort_terugval = False
                if pd.notna(advisor_sort) and pd.notna(prev_advisor_sort):
                    advisor_sort_terugval = float(advisor_sort) + 0.01 < float(prev_advisor_sort)
                    group_df.at[index, "advisor_sort_terugval_vorige"] = bool(advisor_sort_terugval)

                route_mid_terugval = False
                if pd.notna(route_mid) and pd.notna(prev_route_mid):
                    route_mid_terugval = float(route_mid) + 0.01 < float(prev_route_mid)
                    group_df.at[index, "route_mid_terugval_vorige"] = bool(route_mid_terugval)

                route_sort_terugval = False
                if pd.notna(route_sort) and pd.notna(prev_route_sort):
                    route_sort_terugval = float(route_sort) + 0.01 < float(prev_route_sort)
                    group_df.at[index, "route_sort_terugval_vorige"] = bool(route_sort_terugval)
                    # Backwards-compatible alias: vanaf v0.14.2 volgt deze kolom
                    # de feitelijke sorteersleutel, niet meer blind route_mid_m.
                    group_df.at[index, "route_terugval_vorige"] = bool(route_sort_terugval)

                if advisor_sort_terugval and not route_sort_terugval:
                    group_df.at[index, "hm_route_conflict"] = True
                    warnings.append(
                        "WAARSCHUWING: advisor_sort_m ligt vóór de vorige groep; "
                        "controleer primaire ruggengraatvolgorde"
                    )
                    if group_df.at[index, "sort_quality"] == "hoog":
                        group_df.at[index, "sort_quality"] = "middel"

                if route_sort_terugval:
                    group_df.at[index, "hm_route_conflict"] = True
                    if bool(row.get("route_conflict_sort_applied", False)):
                        warnings.append(
                            "INFO: route_sort_m lag vóór de vorige groep, maar het lokale conflict is hersorteerd"
                        )
                    else:
                        warnings.append(
                            "WAARSCHUWING: route_sort_m ligt vóór de vorige groep; "
                            "huidige zichtbare volgorde en sorteersleutel spreken elkaar tegen"
                        )
                    if group_df.at[index, "sort_quality"] == "hoog":
                        group_df.at[index, "sort_quality"] = "middel"
                elif route_mid_terugval:
                    warnings.append(
                        "INFO: route_mid_m ligt vóór de vorige groep, maar route_sort_m blijft oplopend; "
                        "dit is een diagnoseverschil, geen sorteertegenstrijdigheid"
                    )
                    if group_df.at[index, "sort_quality"] == "hoog":
                        group_df.at[index, "sort_quality"] = "middel"

            group_df.at[index, "waarschuwing"] = "; ".join(dict.fromkeys(warnings))
            previous_by_rank[rank_value] = group_df.loc[index]

    return object_df, group_df, axis_result
