"""
Experimentele referentieasdiagnose voor v0.32.

Deze module is bewust diagnostisch. De berekende referentie-metrering wordt
niet gebruikt als productiewaarheid, niet als automatische iASSET-mutatie en
niet als vervanging van de bestaande v0.31-trajectlengtekeuze.

Doel van v0.32:
- PDOK-hectometerpunten of een vergelijkbare referentielaag veilig als proef
  naast iASSET-data leggen;
- begin/eind van rijstrookobjecten indicatief op een referentieas projecteren;
- verschillen met onderhoudsprojectnaam en paspoortmetrering zichtbaar maken;
- foute of incomplete brondata loggen in diagnosetabellen in plaats van crashen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from .sorting_diagnostics import project_geometry_range_on_axis
from .trajectory import parse_project_range, round_km_to_nearest_5m
from .utils import clean_display_value, normalize_text, parse_hm_sort


REFERENCE_AXIS_SCHEMA_VERSION = "refaxis-v0.32.0"
EXPERIMENTAL_ROADS = {"N354", "N398"}

HECTOMETER_COLUMN_CANDIDATES = (
    "hm_val",
    "hectometrering",
    "hectomtrng",
    "hectometer",
    "hm",
)

ROAD_COLUMN_CANDIDATES = (
    "wegnummer",
    "Wegnummer",
    "weg",
    "route",
    "routenummer",
    "n_weg",
    "n-weg",
)


@dataclass(frozen=True)
class ReferenceAxisResult:
    """
    Resultaat van het opbouwen van een experimentele referentieas.

    ``anchors`` bevat de gebruikte hectometerankerpunten. ``axis`` is alleen
    bedoeld voor projectie binnen deze module en niet voor mutaties.
    """

    axis: LineString | None
    anchors: pd.DataFrame
    source: str
    warning: str = ""

    @property
    def anchor_count(self) -> int:
        """Aantal bruikbare ankerpunten."""
        return len(self.anchors)


@dataclass(frozen=True)
class ReferenceDiagnosticsResult:
    """
    Resultaat van de referentieasproef.

    De object- en projecttabellen zijn gewone DataFrames, zodat Streamlit ze
    direct kan tonen en als CSV kan exporteren.
    """

    object_diagnostics: pd.DataFrame
    project_summary: pd.DataFrame
    axis_result: ReferenceAxisResult
    warning: str = ""


def _empty_anchor_frame() -> pd.DataFrame:
    """Maak een lege ankertabel met vaste kolommen."""
    return pd.DataFrame(columns=["hm_km", "route_m", "x", "y", "point_count"])


def _empty_object_frame() -> pd.DataFrame:
    """Maak een lege objectdiagnosetabel met vaste kolommen."""
    return pd.DataFrame(
        columns=[
            "sys_id",
            "Wegnummer",
            "nummer",
            "subthema",
            "Onderhoudsproject",
            "Metrering",
            "referentie_begin_km",
            "referentie_begin_5m",
            "referentie_midden_km",
            "referentie_eind_km",
            "referentie_eind_5m",
            "referentie_lengte_m",
            "afstand_tot_as_m",
            "project_begin_km",
            "project_eind_km",
            "project_lengte_m",
            "verschil_met_projectnaam_m",
            "bronkwaliteit",
            "status",
            "waarschuwing",
        ]
    )


def _empty_project_frame() -> pd.DataFrame:
    """Maak een lege projectsamenvatting met vaste kolommen."""
    return pd.DataFrame(
        columns=[
            "Onderhoudsproject",
            "objecten",
            "objecten_met_projectie",
            "referentie_begin_km",
            "referentie_begin_5m",
            "referentie_eind_km",
            "referentie_eind_5m",
            "referentie_lengte_m",
            "project_lengte_m",
            "verschil_met_projectnaam_m",
            "max_afstand_tot_as_m",
            "waarschuwingen",
            "bronkwaliteit",
        ]
    )


def _first_existing_column(gdf: gpd.GeoDataFrame | pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Geef de eerste bestaande kandidaatkolom terug."""
    if gdf is None:
        return None

    for column in candidates:
        if column in gdf.columns:
            return column
    return None


def _safe_to_rd(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Zet een GeoDataFrame veilig om naar EPSG:28992.

    Als CRS ontbreekt of conversie mislukt, houden we de geometrie zoals die is.
    De diagnose blijft dan bruikbaar voor lokale tests, maar de UI toont de
    bronkwaliteit nog steeds als experimenteel.
    """
    if gdf is None:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:28992")

    working = gdf.copy()
    if "geometry" not in working.columns:
        working["geometry"] = [None] * len(working)
        return gpd.GeoDataFrame(working, geometry="geometry", crs="EPSG:28992")

    try:
        if working.crs is not None and str(working.crs).upper() not in {"EPSG:28992", "28992"}:
            return working.to_crs(epsg=28992)
    except Exception:
        return working

    return working


def _valid_geometry_mask(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Selecteer rijen met een bruikbare geometrie."""
    if gdf is None or gdf.empty or "geometry" not in gdf.columns:
        return pd.Series(dtype=bool)

    mask_values: list[bool] = []
    for geometry in gdf.geometry:
        try:
            mask_values.append(geometry is not None and not geometry.is_empty)
        except Exception:
            mask_values.append(False)

    return pd.Series(mask_values, index=gdf.index, dtype=bool)


def _parse_hectometer_to_km(value: Any) -> float | None:
    """
    Parseer een PDOK-hectometerwaarde naar kilometers.

    De bestaande PDOK-laag zet ``hm_val`` zo dat 143 overeenkomt met 14,3 km.
    Als een bron al decimale kilometers bevat (bijvoorbeeld 14.3), blijft die
    waarde behouden.
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

    if numeric < 0:
        return None

    # Waarden zoals 143 of 14.0 uit PDOK zijn hectometerwaarden. Een al als
    # kilometers aangeleverde waarde met drie decimalen laten we intact.
    if numeric >= 100 or float(numeric).is_integer():
        return numeric / 10.0

    return numeric


def _round_km_or_none(value: Any, digits: int = 3) -> float | None:
    """Rond een kilometerwaarde veilig af of geef None terug."""
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError, OverflowError):
        return None


def _round_m_or_none(value: Any, digits: int = 1) -> float | None:
    """Rond een meterwaarde veilig af of geef None terug."""
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError, OverflowError):
        return None


def _five_meter_or_none(value: Any) -> float | None:
    """Pas de vijfmeterregel veilig toe."""
    try:
        if value is None or pd.isna(value):
            return None
        return round_km_to_nearest_5m(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _road_filter_mask(gdf: gpd.GeoDataFrame, selected_road: str | None) -> pd.Series:
    """
    Filter PDOK-punten op wegnummer als de bronkolom beschikbaar is.

    Ontbreekt de kolom of levert filtering te weinig ankerpunten op, dan wordt
    de hele bbox-set gebruikt. Dat is bewust: v0.32 is een proeflaag en moet
    bij schema-afwijkingen niet stuklopen.
    """
    if gdf is None or gdf.empty or not selected_road:
        return pd.Series(True, index=getattr(gdf, "index", []), dtype=bool)

    road_column = _first_existing_column(gdf, ROAD_COLUMN_CANDIDATES)
    if not road_column:
        return pd.Series(True, index=gdf.index, dtype=bool)

    target = normalize_text(selected_road).replace(" ", "")
    target_digits = "".join(char for char in target if char.isdigit())

    values = gdf[road_column].map(lambda value: normalize_text(value).replace(" ", ""))
    mask = values.str.contains(target, na=False)

    if target_digits:
        digit_mask = values.str.contains(target_digits, na=False)
        mask = mask | digit_mask

    if int(mask.sum()) >= 2:
        return mask

    return pd.Series(True, index=gdf.index, dtype=bool)


def build_reference_axis_from_hectopoints(
    hectopoints: gpd.GeoDataFrame | None,
    *,
    selected_road: str | None = None,
) -> ReferenceAxisResult:
    """
    Bouw een experimentele referentieas uit hectometerpunten.

    De punten worden per hectometerwaarde samengevat tot één ankerpunt. Dit
    dempt dubbele punten van linker-/rechterrijbaan, maar is nog steeds een
    proefbenadering. Daarom krijgt de bron altijd het label ``experimenteel``.
    """
    if hectopoints is None or hectopoints.empty:
        return ReferenceAxisResult(
            axis=None,
            anchors=_empty_anchor_frame(),
            source="geen_hectometerpunten",
            warning="Geen PDOK-hectometerpunten beschikbaar. Zet de PDOK-laag aan of controleer de verbinding.",
        )

    working = _safe_to_rd(hectopoints)
    valid_mask = _valid_geometry_mask(working)
    if valid_mask.empty or int(valid_mask.sum()) == 0:
        return ReferenceAxisResult(
            axis=None,
            anchors=_empty_anchor_frame(),
            source="geen_geldige_geometrie",
            warning="PDOK-hectometerpunten bevatten geen bruikbare geometrieën.",
        )

    working = working.loc[valid_mask].copy()
    working = working.loc[_road_filter_mask(working, selected_road)].copy()

    hm_column = _first_existing_column(working, HECTOMETER_COLUMN_CANDIDATES)
    if not hm_column:
        return ReferenceAxisResult(
            axis=None,
            anchors=_empty_anchor_frame(),
            source="geen_hectometerkolom",
            warning="Geen herkenbare hectometerkolom gevonden in de referentielaag.",
        )

    working["_ref_hm_km"] = working[hm_column].map(_parse_hectometer_to_km)
    working = working.dropna(subset=["_ref_hm_km"]).copy()

    if working.empty:
        return ReferenceAxisResult(
            axis=None,
            anchors=_empty_anchor_frame(),
            source="geen_geldige_hectometrering",
            warning="Geen bruikbare hectometerwaarden gevonden in de referentielaag.",
        )

    anchor_rows: list[dict[str, Any]] = []
    # Afronden op meters voorkomt dat waarden als 14.3000000002 aparte buckets
    # worden. Binnen één hm-bucket nemen we het ruimtelijke gemiddelde.
    working["_ref_hm_bucket"] = working["_ref_hm_km"].round(3)

    for hm_km, group in working.groupby("_ref_hm_bucket", sort=True):
        points: list[Point] = []
        for geometry in group.geometry:
            try:
                points.append(geometry.centroid)
            except Exception:
                continue

        if not points:
            continue

        x = sum(point.x for point in points) / len(points)
        y = sum(point.y for point in points) / len(points)
        anchor_rows.append(
            {
                "hm_km": float(hm_km),
                "x": float(x),
                "y": float(y),
                "point_count": int(len(points)),
            }
        )

    if len(anchor_rows) < 2:
        return ReferenceAxisResult(
            axis=None,
            anchors=pd.DataFrame(anchor_rows) if anchor_rows else _empty_anchor_frame(),
            source="onvoldoende_ankerpunten",
            warning="Te weinig unieke hectometerpunten om een referentieas te maken.",
        )

    anchor_rows = sorted(anchor_rows, key=lambda row: row["hm_km"])
    unique_points: list[Point] = []
    clean_rows: list[dict[str, Any]] = []
    for row in anchor_rows:
        point = Point(row["x"], row["y"])
        if unique_points and point.distance(unique_points[-1]) <= 0.01:
            continue
        unique_points.append(point)
        clean_rows.append(row)

    if len(unique_points) < 2:
        return ReferenceAxisResult(
            axis=None,
            anchors=pd.DataFrame(clean_rows) if clean_rows else _empty_anchor_frame(),
            source="onvoldoende_unieke_punten",
            warning="De hectometerpunten vallen ruimtelijk samen; referentieas niet bruikbaar.",
        )

    try:
        axis = LineString([(point.x, point.y) for point in unique_points])
    except Exception:
        return ReferenceAxisResult(
            axis=None,
            anchors=pd.DataFrame(clean_rows) if clean_rows else _empty_anchor_frame(),
            source="as_fout",
            warning="Referentieas kon niet worden opgebouwd uit hectometerpunten.",
        )

    if axis.length <= 0:
        return ReferenceAxisResult(
            axis=None,
            anchors=pd.DataFrame(clean_rows) if clean_rows else _empty_anchor_frame(),
            source="as_lengte_nul",
            warning="Referentieas heeft lengte 0.",
        )

    for row, point in zip(clean_rows, unique_points, strict=False):
        row["route_m"] = float(axis.project(point))

    anchors = pd.DataFrame(clean_rows, columns=["hm_km", "route_m", "x", "y", "point_count"])
    anchors = anchors.sort_values("route_m").reset_index(drop=True)

    # Als sortering op hm en sortering langs de as duidelijk botsen, melden we
    # dat zonder de diagnose te blokkeren.
    warning = ""
    hm_monotonic = anchors["hm_km"].is_monotonic_increasing or anchors["hm_km"].is_monotonic_decreasing
    if not hm_monotonic:
        warning = (
            "Hectometerpunten zijn langs de opgebouwde as niet monotone oplopend. "
            "Controleer of er punten van parallelle of kruisende wegen zijn meegekomen."
        )

    return ReferenceAxisResult(
        axis=axis,
        anchors=anchors,
        source="pdok_hectometerpunten_experimenteel",
        warning=warning,
    )


def _interpolate_route_to_km(
    route_m: float | None,
    anchors: pd.DataFrame,
) -> tuple[float | None, bool]:
    """
    Vertaal een routepositie langs de as naar hectometrering.

    We gebruiken lineaire interpolatie tussen de gebruikte ankerpunten. Buiten
    de ankerrange wordt de waarde geklemd en markeren we ``in_range=False``.
    """
    if route_m is None or anchors is None or anchors.empty or len(anchors) < 2:
        return None, False

    try:
        route_value = float(route_m)
    except (TypeError, ValueError, OverflowError):
        return None, False

    route_values = anchors["route_m"].astype(float).tolist()
    hm_values = anchors["hm_km"].astype(float).tolist()

    if not route_values or not hm_values:
        return None, False

    if route_value < route_values[0]:
        return hm_values[0], False
    if route_value == route_values[0]:
        return hm_values[0], True
    if route_value > route_values[-1]:
        return hm_values[-1], False
    if route_value == route_values[-1]:
        return hm_values[-1], True

    for index in range(1, len(route_values)):
        left_route = route_values[index - 1]
        right_route = route_values[index]
        if route_value > right_route:
            continue

        left_hm = hm_values[index - 1]
        right_hm = hm_values[index]
        span = right_route - left_route
        if span <= 0:
            return left_hm, False

        fraction = (route_value - left_route) / span
        return left_hm + fraction * (right_hm - left_hm), True

    return hm_values[-1], False


def _rijstrook_mask(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Selecteer alleen rijstroken voor de eerste v0.32-proef."""
    if gdf is None or gdf.empty:
        return pd.Series(dtype=bool)

    if "subthema_clean" in gdf.columns:
        values = gdf["subthema_clean"].map(normalize_text)
    elif "subthema" in gdf.columns:
        values = gdf["subthema"].map(normalize_text)
    else:
        values = pd.Series([""] * len(gdf), index=gdf.index)

    return values == "rijstrook"


def _row_value(row: pd.Series, column: str) -> str:
    """Haal een rijwaarde op als nette tekst."""
    if column not in row.index:
        return ""
    return clean_display_value(row.get(column))


def _project_name_metrics(project_name: Any) -> tuple[float | None, float | None, float | None]:
    """Haal begin/eind/lengte uit een onderhoudsprojectnaam."""
    project_range = parse_project_range(project_name)
    if project_range is None:
        return None, None, None
    return project_range.start_km, project_range.end_km, project_range.length_m


def _object_warning(
    *,
    start_in_range: bool,
    mid_in_range: bool,
    end_in_range: bool,
    offset_m: float | None,
    max_offset_m: float,
    ref_length_m: float | None,
) -> str:
    """Maak een compacte waarschuwingstekst voor één object."""
    messages: list[str] = []

    if not (start_in_range and mid_in_range and end_in_range):
        messages.append("buiten ankerrange")

    if offset_m is not None and offset_m > max_offset_m:
        messages.append(f"afstand tot as > {max_offset_m:g} m")

    if ref_length_m is not None and ref_length_m <= 0:
        messages.append("referentielengte 0")

    return "; ".join(messages)


def _object_status(warning: str, has_projection: bool) -> str:
    """Vertaal projectieresultaat naar een simpele status."""
    if not has_projection:
        return "geen projectie"
    if warning:
        return "controleer"
    return "projectie"


def build_reference_axis_diagnostics(
    road_gdf: gpd.GeoDataFrame,
    hectopoints: gpd.GeoDataFrame | None,
    *,
    selected_road: str | None = None,
    max_offset_m: float = 25.0,
) -> ReferenceDiagnosticsResult:
    """
    Bouw de experimentele referentieasdiagnose voor één geselecteerde weg.

    Alleen rijstroken worden meegenomen. Parallelwegen, fietspaden, rotondes en
    kruispunten blijven bewust buiten de eerste proef, omdat hun relatie met een
    centrale as domeingevoelig is.
    """
    if road_gdf is None or road_gdf.empty:
        axis_result = build_reference_axis_from_hectopoints(hectopoints, selected_road=selected_road)
        return ReferenceDiagnosticsResult(
            object_diagnostics=_empty_object_frame(),
            project_summary=_empty_project_frame(),
            axis_result=axis_result,
            warning="Geen iASSET-objecten beschikbaar voor de geselecteerde weg.",
        )

    axis_result = build_reference_axis_from_hectopoints(hectopoints, selected_road=selected_road)

    general_warnings: list[str] = []
    if selected_road and selected_road not in EXPERIMENTAL_ROADS:
        general_warnings.append(
            f"Deze proef is bedoeld voor N354 en N398; {selected_road} wordt alleen indicatief doorgerekend."
        )

    if axis_result.warning:
        general_warnings.append(axis_result.warning)

    if axis_result.axis is None or axis_result.anchors.empty:
        if axis_result.warning:
            general_warnings.append(axis_result.warning)
        return ReferenceDiagnosticsResult(
            object_diagnostics=_empty_object_frame(),
            project_summary=_empty_project_frame(),
            axis_result=axis_result,
            warning=" ".join(dict.fromkeys(general_warnings)),
        )

    working = _safe_to_rd(road_gdf)
    valid_mask = _valid_geometry_mask(working)
    if valid_mask.empty or int(valid_mask.sum()) == 0:
        return ReferenceDiagnosticsResult(
            object_diagnostics=_empty_object_frame(),
            project_summary=_empty_project_frame(),
            axis_result=axis_result,
            warning="Geen bruikbare objectgeometrieën gevonden.",
        )

    working = working.loc[valid_mask & _rijstrook_mask(working)].copy()
    if working.empty:
        return ReferenceDiagnosticsResult(
            object_diagnostics=_empty_object_frame(),
            project_summary=_empty_project_frame(),
            axis_result=axis_result,
            warning="Geen rijstrookobjecten gevonden voor deze proef.",
        )

    rows: list[dict[str, Any]] = []
    for object_index, row in working.iterrows():
        geometry = row.geometry
        route_start, route_mid, route_end, offset_m = project_geometry_range_on_axis(
            geometry,
            axis_result.axis,
        )

        start_km, start_in_range = _interpolate_route_to_km(route_start, axis_result.anchors)
        mid_km, mid_in_range = _interpolate_route_to_km(route_mid, axis_result.anchors)
        end_km, end_in_range = _interpolate_route_to_km(route_end, axis_result.anchors)

        has_projection = start_km is not None and end_km is not None
        if has_projection:
            ref_begin_km = min(float(start_km), float(end_km))
            ref_end_km = max(float(start_km), float(end_km))
            ref_length_m = max(0.0, (ref_end_km - ref_begin_km) * 1000.0)
        else:
            ref_begin_km = None
            ref_end_km = None
            ref_length_m = None

        project_begin_km, project_end_km, project_length_m = _project_name_metrics(row.get("Onderhoudsproject"))

        delta_project_m = None
        if ref_length_m is not None and project_length_m is not None:
            delta_project_m = ref_length_m - project_length_m

        warning = _object_warning(
            start_in_range=start_in_range,
            mid_in_range=mid_in_range,
            end_in_range=end_in_range,
            offset_m=offset_m,
            max_offset_m=max_offset_m,
            ref_length_m=ref_length_m,
        )

        rows.append(
            {
                "sys_id": row.get("sys_id", object_index),
                "Wegnummer": _row_value(row, "Wegnummer"),
                "nummer": _row_value(row, "nummer"),
                "subthema": _row_value(row, "subthema"),
                "Onderhoudsproject": _row_value(row, "Onderhoudsproject"),
                "Metrering": _row_value(row, "Metrering"),
                "referentie_begin_km": _round_km_or_none(ref_begin_km),
                "referentie_begin_5m": _round_km_or_none(_five_meter_or_none(ref_begin_km)),
                "referentie_midden_km": _round_km_or_none(mid_km),
                "referentie_eind_km": _round_km_or_none(ref_end_km),
                "referentie_eind_5m": _round_km_or_none(_five_meter_or_none(ref_end_km)),
                "referentie_lengte_m": _round_m_or_none(ref_length_m),
                "afstand_tot_as_m": _round_m_or_none(offset_m),
                "project_begin_km": _round_km_or_none(project_begin_km),
                "project_eind_km": _round_km_or_none(project_end_km),
                "project_lengte_m": _round_m_or_none(project_length_m),
                "verschil_met_projectnaam_m": _round_m_or_none(delta_project_m),
                "bronkwaliteit": "experimenteel",
                "status": _object_status(warning, has_projection),
                "waarschuwing": warning,
            }
        )

    object_df = pd.DataFrame(rows, columns=_empty_object_frame().columns)
    project_df = _build_project_summary(object_df)

    return ReferenceDiagnosticsResult(
        object_diagnostics=object_df,
        project_summary=project_df,
        axis_result=axis_result,
        warning=" ".join(dict.fromkeys(general_warnings)),
    )


def _build_project_summary(object_df: pd.DataFrame) -> pd.DataFrame:
    """Vat objectprojecties per onderhoudsproject samen."""
    if object_df is None or object_df.empty:
        return _empty_project_frame()

    rows: list[dict[str, Any]] = []
    group_column = "Onderhoudsproject"
    working = object_df.copy()
    working[group_column] = working[group_column].map(lambda value: clean_display_value(value) or "<geen projectnaam>")

    for project_name, group in working.groupby(group_column, dropna=False, sort=True):
        start_values = pd.to_numeric(group["referentie_begin_km"], errors="coerce").dropna()
        end_values = pd.to_numeric(group["referentie_eind_km"], errors="coerce").dropna()
        offset_values = pd.to_numeric(group["afstand_tot_as_m"], errors="coerce").dropna()

        if not start_values.empty and not end_values.empty:
            ref_begin = float(start_values.min())
            ref_end = float(end_values.max())
            ref_length = max(0.0, (ref_end - ref_begin) * 1000.0)
        else:
            ref_begin = None
            ref_end = None
            ref_length = None

        project_lengths = pd.to_numeric(group["project_lengte_m"], errors="coerce").dropna()
        project_length = float(project_lengths.iloc[0]) if not project_lengths.empty else None

        delta_project_m = None
        if ref_length is not None and project_length is not None:
            delta_project_m = ref_length - project_length

        warnings = [
            clean_display_value(value)
            for value in group["waarschuwing"].tolist()
            if clean_display_value(value)
        ]

        rows.append(
            {
                "Onderhoudsproject": project_name,
                "objecten": int(len(group)),
                "objecten_met_projectie": int((group["status"] != "geen projectie").sum()),
                "referentie_begin_km": _round_km_or_none(ref_begin),
                "referentie_begin_5m": _round_km_or_none(_five_meter_or_none(ref_begin)),
                "referentie_eind_km": _round_km_or_none(ref_end),
                "referentie_eind_5m": _round_km_or_none(_five_meter_or_none(ref_end)),
                "referentie_lengte_m": _round_m_or_none(ref_length),
                "project_lengte_m": _round_m_or_none(project_length),
                "verschil_met_projectnaam_m": _round_m_or_none(delta_project_m),
                "max_afstand_tot_as_m": _round_m_or_none(float(offset_values.max())) if not offset_values.empty else None,
                "waarschuwingen": "; ".join(sorted(set(warnings))),
                "bronkwaliteit": "experimenteel",
            }
        )

    return pd.DataFrame(rows, columns=_empty_project_frame().columns)
