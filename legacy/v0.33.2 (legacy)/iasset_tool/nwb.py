"""
NWB-referentieproef voor v0.33.

Deze module gebruikt de officiële NWB OGC API Features als experimentele
diagnosebron naast iASSET. De uitkomsten worden nooit gebruikt om automatisch
iASSET-data te muteren.

Waarom deze module naast ``reference_axis.py`` bestaat:
- v0.32 bouwde een proefas uit losse hectometerpunten.
- v0.33 onderzoekt een robuustere bron: NWB-wegvakken én NWB-hectopunten.
- De iASSET/Dielplak-wegas kan daarmee als interne beheer-as naast de
  officiële NWB-referentie worden gelegd.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box
from shapely.ops import unary_union

from .utils import clean_display_value, normalize_text


NWB_REFERENCE_SCHEMA_VERSION = "nwb-ref-v0.33.2"
NWB_OGC_API_BASE_URL = "https://api.pdok.nl/rws/nationaal-wegenbestand-wegen/ogc/v1"

NWB_ROAD_COLUMNS = (
    "wegnummer",
    "wegnr_hmp",
    "wegnr_aw",
    "routenr",
    "routenr2",
    "routenr3",
    "routenr4",
)


@dataclass(frozen=True)
class NwbReferenceResult:
    """Resultaat van de experimentele NWB-bronverkenning."""

    wegvakken: gpd.GeoDataFrame
    hectopunten: gpd.GeoDataFrame
    source_summary: pd.DataFrame
    warning: str = ""


def _empty_gdf(geometry_type: str = "geometry") -> gpd.GeoDataFrame:
    """Maak een lege GeoDataFrame in RD New."""
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:28992")


def _empty_source_summary() -> pd.DataFrame:
    """Maak een lege bronsamenvatting met vaste kolommen."""
    return pd.DataFrame(
        columns=[
            "Wegnummer",
            "nwb_wegvakken",
            "nwb_hectopunten",
            "unieke_wvk_ids",
            "wegvak_beginkm_min",
            "wegvak_eindkm_max",
            "hectopunt_min_km",
            "hectopunt_max_km",
            "bronkwaliteit",
            "status",
            "waarschuwing",
        ]
    )


def _empty_wegas_comparison() -> pd.DataFrame:
    """Maak een lege vergelijkingstabel voor iASSET-wegassen versus NWB."""
    return pd.DataFrame(
        columns=[
            "nummer",
            "naam",
            "Wegnummer",
            "iasset_lengte_m",
            "afstand_min_tot_nwb_m",
            "afstand_gem_tot_nwb_m",
            "afstand_max_sample_tot_nwb_m",
            "bronkwaliteit",
            "status",
            "waarschuwing",
        ]
    )


def _empty_wegas_comparison_detail() -> pd.DataFrame:
    """Maak een lege detailtabel voor iASSET-wegas-samplepunten versus NWB."""
    return pd.DataFrame(
        columns=[
            "nummer",
            "naam",
            "Wegnummer",
            "sample_nr",
            "afstand_langs_iasset_wegas_m",
            "sample_fractie",
            "x_rd",
            "y_rd",
            "afstand_tot_nwb_m",
            "dichtstbijzijnde_nwb_wvk_id",
            "dichtstbijzijnde_nwb_wegnummer",
            "dichtstbijzijnde_nwb_routenr",
            "dichtstbijzijnde_nwb_beginkm",
            "dichtstbijzijnde_nwb_eindkm",
            "bronkwaliteit",
            "status",
            "waarschuwing",
        ]
    )


def _ensure_rd(gdf: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    """Zorg dat geometrie in EPSG:28992 staat; lege frames blijven veilig leeg."""
    if gdf is None or gdf.empty:
        return _empty_gdf()

    result = gdf.copy()
    if result.crs is None:
        # iASSET-exports met RD-coördinaten hebben soms geen CRS-metadata.
        # Voor GeoJSON is dat meestal WGS84; daarom alleen defaulten als er geen
        # crs is én de coördinaten duidelijk op lon/lat lijken.
        minx, miny, maxx, maxy = result.total_bounds
        if -180 <= minx <= 180 and -90 <= miny <= 90 and -180 <= maxx <= 180 and -90 <= maxy <= 90:
            result = result.set_crs(epsg=4326, allow_override=True)
        else:
            result = result.set_crs(epsg=28992, allow_override=True)

    if result.crs.to_epsg() != 28992:
        result = result.to_crs(epsg=28992)

    return result


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Converteer waarden robuust naar numeriek."""
    return pd.to_numeric(series, errors="coerce")


def _normalise_road_query(road: str) -> tuple[str, str]:
    """Geef zowel 'N354' als '354' terug voor flexibele NWB-vergelijking."""
    raw = clean_display_value(road).upper().replace(" ", "").replace("-", "")
    if not raw:
        return "", ""
    digits = "".join(ch for ch in raw if ch.isdigit())
    if raw.startswith("N"):
        return raw, digits
    if digits:
        return f"N{digits}", digits
    return raw, digits


def _normalise_road_value(value: Any) -> tuple[str, str]:
    """Normaliseer een NWB/iASSET-wegnummer of routenummer voor vergelijking.

    iASSET-wegassen kunnen objectnamen hebben zoals ``WA-N354_1`` of
    ``N354_1``. Die suffix betekent: tweede wegasdeel/variant, niet weg N3541.
    Daarom zoeken we eerst naar een herkenbaar N-wegpatroon en pas daarna naar
    volledig numerieke waarden zoals ``354`` of ``354.0``.
    """
    text = clean_display_value(value).upper().replace(" ", "").replace("-", "")
    if not text or text.lower() == "nan":
        return "", ""

    road_match = re.search(r"N(\d{1,4})(?:\D|$)", text)
    if road_match:
        digits = road_match.group(1)
        return f"N{digits}", digits

    # Routenummers komen vaak numeriek binnen, bijvoorbeeld 354 of 354.0.
    try:
        number = int(float(text))
        return f"N{number}", str(number)
    except Exception:
        pass

    digits = "".join(ch for ch in text if ch.isdigit())
    if digits and text == digits:
        return f"N{digits}", digits
    return text, digits


def filter_nwb_wegvakken_for_road(wegvakken: gpd.GeoDataFrame, selected_road: str) -> gpd.GeoDataFrame:
    """
    Filter NWB-wegvakken op wegnummer/routenummer.

    Waarom meerdere kolommen?
    NWB bevat zowel administratieve wegnummerkolommen als routenummerkolommen.
    In de bronverkenning willen we bewust ruim zoeken, zodat we niet te vroeg
    relevante wegvakken kwijtraken.
    """
    if wegvakken is None or wegvakken.empty:
        return _empty_gdf()

    road_n, road_digits = _normalise_road_query(selected_road)
    if not road_n and not road_digits:
        return _empty_gdf()

    mask = pd.Series(False, index=wegvakken.index)

    for column in NWB_ROAD_COLUMNS:
        if column not in wegvakken.columns:
            continue
        normalised = wegvakken[column].map(_normalise_road_value)
        col_mask = normalised.map(lambda pair: pair[0] == road_n or (road_digits and pair[1] == road_digits))
        mask = mask | col_mask.fillna(False)

    result = wegvakken.loc[mask].copy()
    return _ensure_rd(result)


def _wvk_id_key(value: Any) -> str:
    """Normaliseer ``wvk_id`` zodat 123, 123.0 en '123' gelijk behandeld worden."""
    if value is None:
        return ""
    text = clean_display_value(value)
    if not text or text.lower() == "nan":
        return ""
    try:
        return str(int(float(text)))
    except Exception:
        return text


def filter_nwb_hectopunten_for_wegvakken(
    hectopunten: gpd.GeoDataFrame,
    wegvakken: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Filter hectopunten via de ``wvk_id``-koppeling met NWB-wegvakken."""
    if hectopunten is None or hectopunten.empty or wegvakken is None or wegvakken.empty:
        return _empty_gdf()
    if "wvk_id" not in hectopunten.columns or "wvk_id" not in wegvakken.columns:
        return _empty_gdf()

    allowed_ids = {_wvk_id_key(value) for value in wegvakken["wvk_id"].dropna().tolist()}
    allowed_ids.discard("")
    if not allowed_ids:
        return _empty_gdf()

    keys = hectopunten["wvk_id"].map(_wvk_id_key)
    result = hectopunten.loc[keys.isin(allowed_ids)].copy()
    return _ensure_rd(result)


def _bbox_crs84_from_rd_gdf(gdf: gpd.GeoDataFrame, buffer_meters: float = 500.0) -> tuple[float, float, float, float] | None:
    """Maak een CRS84-bbox voor OGC API-vragen op basis van RD-geometrie."""
    if gdf is None or gdf.empty:
        return None

    rd = _ensure_rd(gdf)
    minx, miny, maxx, maxy = rd.total_bounds
    bbox_geom = box(
        minx - buffer_meters,
        miny - buffer_meters,
        maxx + buffer_meters,
        maxy + buffer_meters,
    )
    bbox_wgs84 = gpd.GeoSeries([bbox_geom], crs="EPSG:28992").to_crs(epsg=4326).total_bounds
    return tuple(float(value) for value in bbox_wgs84)


def _features_to_gdf(features: list[dict[str, Any]]) -> gpd.GeoDataFrame:
    """Zet OGC API-features veilig om naar GeoDataFrame in RD New."""
    if not features:
        return _empty_gdf()

    try:
        result = gpd.GeoDataFrame.from_features(features)
    except Exception:
        return _empty_gdf()

    if result.empty:
        return _empty_gdf()

    if result.crs is None:
        result = result.set_crs(epsg=4326, allow_override=True)

    return _ensure_rd(result)


def fetch_nwb_collection_by_bbox(
    collection: str,
    bbox_crs84: tuple[float, float, float, float],
    *,
    limit: int = 10000,
    max_pages: int = 10,
    timeout_seconds: int = 15,
) -> tuple[gpd.GeoDataFrame, str]:
    """
    Haal een NWB-collectie op via OGC API Features binnen een bbox.

    Bij netwerkfouten of onverwachte responses geeft deze functie een leeg
    GeoDataFrame terug met een waarschuwing. De Streamlit-app mag hier nooit
    op crashen.
    """
    if not bbox_crs84:
        return _empty_gdf(), "Geen bbox beschikbaar voor NWB-opvraag."

    url = f"{NWB_OGC_API_BASE_URL}/collections/{collection}/items"
    params: dict[str, Any] | None = {
        "f": "json",
        "bbox": ",".join(f"{value:.8f}" for value in bbox_crs84),
        "limit": int(limit),
    }

    features: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_urls: set[str] = set()

    for _page in range(max_pages):
        try:
            response = requests.get(url, params=params, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            warnings.append(f"NWB {collection} kon niet worden opgehaald: {exc}")
            break

        features.extend(payload.get("features") or [])

        next_url = ""
        for link in payload.get("links") or []:
            if link.get("rel") == "next" and link.get("href"):
                next_url = link["href"]
                break

        if not next_url or next_url in seen_urls:
            break

        seen_urls.add(next_url)
        url = next_url
        params = None

    warning = " ".join(warnings)
    return _features_to_gdf(features), warning


def build_nwb_source_summary(
    selected_road: str,
    wegvakken: gpd.GeoDataFrame,
    hectopunten: gpd.GeoDataFrame,
    *,
    warning: str = "",
) -> pd.DataFrame:
    """Maak een compacte bronsamenvatting voor Streamlit en CSV-export."""
    if wegvakken is None:
        wegvakken = _empty_gdf()
    if hectopunten is None:
        hectopunten = _empty_gdf()

    if wegvakken.empty:
        status = "geen_wegvakken"
        local_warning = "Geen NWB-wegvakken gevonden voor deze weg en bbox."
    elif hectopunten.empty:
        status = "geen_hectopunten"
        local_warning = "NWB-wegvakken gevonden, maar geen gekoppelde hectopunten via wvk_id."
    else:
        status = "bron_gevonden"
        local_warning = ""

    warnings = "; ".join(part for part in [local_warning, warning] if part)

    def minmax(column: str, frame: pd.DataFrame) -> tuple[float | None, float | None]:
        if column not in frame.columns or frame.empty:
            return None, None
        values = _safe_numeric(frame[column]).dropna()
        if values.empty:
            return None, None
        return float(values.min()), float(values.max())

    begin_min, _ = minmax("beginkm", wegvakken)
    _, eind_max = minmax("eindkm", wegvakken)
    hm_min_raw, hm_max_raw = minmax("hectomtrng", hectopunten)

    # NWB-hectomtrng is een hectometerwaarde als integer: 86 betekent 8.6 km.
    hm_min_km = hm_min_raw / 10.0 if hm_min_raw is not None else None
    hm_max_km = hm_max_raw / 10.0 if hm_max_raw is not None else None

    unique_wvk_ids = 0
    if "wvk_id" in wegvakken.columns and not wegvakken.empty:
        unique_wvk_ids = len({_wvk_id_key(value) for value in wegvakken["wvk_id"].dropna().tolist()} - {""})

    return pd.DataFrame(
        [
            {
                "Wegnummer": selected_road,
                "nwb_wegvakken": int(len(wegvakken)),
                "nwb_hectopunten": int(len(hectopunten)),
                "unieke_wvk_ids": int(unique_wvk_ids),
                "wegvak_beginkm_min": begin_min,
                "wegvak_eindkm_max": eind_max,
                "hectopunt_min_km": hm_min_km,
                "hectopunt_max_km": hm_max_km,
                "bronkwaliteit": "extern_nwb_ogc_api_experimenteel",
                "status": status,
                "waarschuwing": warnings,
            }
        ]
    )


def build_nwb_reference_for_road(
    road_gdf: gpd.GeoDataFrame,
    selected_road: str,
    *,
    buffer_meters: float = 500.0,
    limit: int = 10000,
    max_pages: int = 10,
    timeout_seconds: int = 15,
) -> NwbReferenceResult:
    """
    Haal NWB-wegvakken en gekoppelde NWB-hectopunten op voor één weg.

    De selectie gebeurt bewust in twee stappen:
    1. Wegvakken binnen de iASSET-bbox ophalen en filteren op wegnummer.
    2. Hectopunten binnen dezelfde bbox ophalen en koppelen via ``wvk_id``.
    """
    if road_gdf is None or road_gdf.empty:
        warning = "Geen iASSET-objecten beschikbaar om een NWB-bbox te bepalen."
        return NwbReferenceResult(_empty_gdf(), _empty_gdf(), _empty_source_summary(), warning)

    bbox_crs84 = _bbox_crs84_from_rd_gdf(road_gdf, buffer_meters=buffer_meters)
    if bbox_crs84 is None:
        warning = "Geen geldige bbox beschikbaar voor NWB-opvraag."
        return NwbReferenceResult(_empty_gdf(), _empty_gdf(), _empty_source_summary(), warning)

    raw_wegvakken, warning_wegvakken = fetch_nwb_collection_by_bbox(
        "wegvakken",
        bbox_crs84,
        limit=limit,
        max_pages=max_pages,
        timeout_seconds=timeout_seconds,
    )
    wegvakken = filter_nwb_wegvakken_for_road(raw_wegvakken, selected_road)

    raw_hectopunten, warning_hectopunten = fetch_nwb_collection_by_bbox(
        "hectopunten",
        bbox_crs84,
        limit=limit,
        max_pages=max_pages,
        timeout_seconds=timeout_seconds,
    )
    hectopunten = filter_nwb_hectopunten_for_wegvakken(raw_hectopunten, wegvakken)

    warning = "; ".join(part for part in [warning_wegvakken, warning_hectopunten] if part)
    summary = build_nwb_source_summary(selected_road, wegvakken, hectopunten, warning=warning)
    return NwbReferenceResult(wegvakken, hectopunten, summary, warning)


def read_wegassen_geojson_bytes(data: bytes | str) -> gpd.GeoDataFrame:
    """Lees een iASSET-wegassen-GeoJSON veilig in."""
    if not data:
        return _empty_gdf()

    try:
        if isinstance(data, bytes):
            payload = json.loads(data.decode("utf-8-sig"))
        elif isinstance(data, str):
            payload = json.loads(data)
        else:
            payload = data
        features = payload.get("features") or []
        result = gpd.GeoDataFrame.from_features(features)
    except Exception:
        return _empty_gdf()

    if result.empty:
        return _empty_gdf()

    result = result.set_crs(epsg=4326, allow_override=True)
    return _ensure_rd(result)


def filter_iasset_wegassen_for_road(wegassen_gdf: gpd.GeoDataFrame, selected_road: str) -> gpd.GeoDataFrame:
    """Filter iASSET-wegassen op wegnummer/objectnummer/naam."""
    if wegassen_gdf is None or wegassen_gdf.empty:
        return _empty_gdf()

    road_n, road_digits = _normalise_road_query(selected_road)
    result = _ensure_rd(wegassen_gdf)

    candidate_columns = [col for col in ["Wegnummer", "nummer", "naam", "Object nummer", "Object naam"] if col in result.columns]
    if not candidate_columns:
        return result.iloc[0:0].copy()

    mask = pd.Series(False, index=result.index)
    for column in candidate_columns:
        normalised = result[column].map(_normalise_road_value)
        col_mask = normalised.map(lambda pair: pair[0] == road_n or (road_digits and pair[1] == road_digits))
        mask = mask | col_mask.fillna(False)

    return result.loc[mask].copy()


def _sample_geometry_distances(geometry: Any, target: Any, sample_count: int = 25) -> list[float]:
    """Neem punten over een lijn en meet afstand tot een doelgeometrie."""
    if geometry is None or geometry.is_empty or target is None or target.is_empty:
        return []

    try:
        length = float(geometry.length)
    except Exception:
        return []

    if length <= 0:
        try:
            return [float(geometry.distance(target))]
        except Exception:
            return []

    distances: list[float] = []
    for index in range(sample_count + 1):
        try:
            point = geometry.interpolate(index / sample_count, normalized=True)
            distances.append(float(point.distance(target)))
        except Exception:
            continue
    return distances




def _sample_positions_along_geometry(
    geometry: Any,
    *,
    sample_count: int = 25,
    sample_step_m: float | None = None,
    max_samples_per_axis: int = 500,
) -> list[float]:
    """Bepaal sampleposities langs een wegas in meters.

    We nemen altijd de genormaliseerde punten uit de asvergelijking mee
    (0/25, 1/25, ..., 25/25). Als een sampleafstand is opgegeven, voegen we
    ook regelmatige meterstappen toe. Zo blijft de detail-export vergelijkbaar
    met de samenvatting én krijgen beheerders meer lokale aanwijzingen.
    """
    if geometry is None or geometry.is_empty:
        return []

    try:
        length = float(geometry.length)
    except Exception:
        return []

    if length <= 0:
        return [0.0]

    positions = {0.0, length}

    safe_sample_count = max(int(sample_count), 1)
    for index in range(safe_sample_count + 1):
        positions.add(length * index / safe_sample_count)

    if sample_step_m is not None:
        try:
            step = float(sample_step_m)
        except Exception:
            step = 0.0
        if step > 0:
            step_count = int(length // step)
            for index in range(step_count + 1):
                positions.add(min(length, index * step))

    sorted_positions = sorted(positions)
    if len(sorted_positions) <= max_samples_per_axis:
        return sorted_positions

    # Bij heel lange assen beperken we de hoeveelheid detailregels, maar we
    # behouden begin/eind en verdelen de rest gelijkmatig.
    keep_indices = sorted(
        {
            round(index * (len(sorted_positions) - 1) / (max_samples_per_axis - 1))
            for index in range(max_samples_per_axis)
        }
    )
    return [sorted_positions[index] for index in keep_indices]


def _nearest_nwb_row(point: Any, nwb_wegvakken: gpd.GeoDataFrame) -> tuple[pd.Series | None, float | None]:
    """Zoek het dichtstbijzijnde NWB-wegvak voor één samplepunt."""
    if point is None or point.is_empty or nwb_wegvakken is None or nwb_wegvakken.empty:
        return None, None

    best_index = None
    best_distance = None

    for index, geom in nwb_wegvakken.geometry.items():
        if geom is None or geom.is_empty:
            continue
        try:
            distance = float(point.distance(geom))
        except Exception:
            continue
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = index

    if best_index is None:
        return None, None
    return nwb_wegvakken.loc[best_index], best_distance


def compare_iasset_wegassen_to_nwb_detail(
    wegassen_gdf: gpd.GeoDataFrame,
    nwb_wegvakken: gpd.GeoDataFrame,
    selected_road: str,
    *,
    max_distance_m: float = 25.0,
    sample_step_m: float = 100.0,
    sample_count: int = 25,
    max_samples_per_axis: int = 500,
) -> pd.DataFrame:
    """
    Maak detailregels per samplepunt voor de iASSET-wegas versus NWB.

    Waarom deze detail-export?
    De samenvatting zegt dát een wegas afwijkt. Deze tabel laat zien wáár langs
    de iASSET-wegas de afwijking optreedt. Dat is bedoeld voor visuele controle
    en overleg, niet voor automatische mutaties in iASSET.
    """
    if nwb_wegvakken is None or nwb_wegvakken.empty:
        return _empty_wegas_comparison_detail()

    filtered_wegassen = filter_iasset_wegassen_for_road(wegassen_gdf, selected_road)
    if filtered_wegassen.empty:
        return _empty_wegas_comparison_detail()

    nwb_rd = _ensure_rd(nwb_wegvakken)
    rows: list[dict[str, Any]] = []

    for _, wegas_row in filtered_wegassen.iterrows():
        geom = wegas_row.geometry
        if geom is None or geom.is_empty:
            continue

        try:
            length = float(geom.length)
        except Exception:
            continue

        positions = _sample_positions_along_geometry(
            geom,
            sample_count=sample_count,
            sample_step_m=sample_step_m,
            max_samples_per_axis=max_samples_per_axis,
        )

        for sample_nr, position_m in enumerate(positions, start=1):
            try:
                point = geom.interpolate(position_m)
            except Exception:
                continue

            nearest_row, distance = _nearest_nwb_row(point, nwb_rd)
            warnings: list[str] = []
            if distance is None:
                warnings.append("Geen dichtstbijzijnd NWB-wegvak gevonden.")
            elif distance > max_distance_m:
                warnings.append("samplepunt ligt verder dan de maximale afstand tot NWB")

            status = "vergelijking" if not warnings else "controleer"

            rows.append(
                {
                    "nummer": clean_display_value(wegas_row.get("nummer", "")),
                    "naam": clean_display_value(wegas_row.get("naam", "")),
                    "Wegnummer": clean_display_value(wegas_row.get("Wegnummer", "")),
                    "sample_nr": int(sample_nr),
                    "afstand_langs_iasset_wegas_m": round(float(position_m), 2),
                    "sample_fractie": round(float(position_m / length), 6) if length > 0 else None,
                    "x_rd": round(float(point.x), 3),
                    "y_rd": round(float(point.y), 3),
                    "afstand_tot_nwb_m": round(float(distance), 2) if distance is not None else None,
                    "dichtstbijzijnde_nwb_wvk_id": clean_display_value(nearest_row.get("wvk_id", "")) if nearest_row is not None else "",
                    "dichtstbijzijnde_nwb_wegnummer": clean_display_value(nearest_row.get("wegnummer", "")) if nearest_row is not None else "",
                    "dichtstbijzijnde_nwb_routenr": clean_display_value(nearest_row.get("routenr", "")) if nearest_row is not None else "",
                    "dichtstbijzijnde_nwb_beginkm": nearest_row.get("beginkm", None) if nearest_row is not None else None,
                    "dichtstbijzijnde_nwb_eindkm": nearest_row.get("eindkm", None) if nearest_row is not None else None,
                    "bronkwaliteit": "experimenteel",
                    "status": status,
                    "waarschuwing": "; ".join(warnings),
                }
            )

    if not rows:
        return _empty_wegas_comparison_detail()

    result = pd.DataFrame(rows, columns=_empty_wegas_comparison_detail().columns)
    return result

def compare_iasset_wegassen_to_nwb(
    wegassen_gdf: gpd.GeoDataFrame,
    nwb_wegvakken: gpd.GeoDataFrame,
    selected_road: str,
    *,
    max_distance_m: float = 25.0,
) -> pd.DataFrame:
    """
    Vergelijk iASSET/Dielplak-wegassen met NWB-wegvakken.

    Dit is een diagnose op asniveau. De uitkomst zegt niet dat een bron fout is,
    maar markeert waar de interne iASSET-wegas en externe NWB-wegvakken ruimtelijk
    sterk van elkaar afwijken.
    """
    if nwb_wegvakken is None or nwb_wegvakken.empty:
        frame = _empty_wegas_comparison()
        return frame

    filtered_wegassen = filter_iasset_wegassen_for_road(wegassen_gdf, selected_road)
    if filtered_wegassen.empty:
        return _empty_wegas_comparison()

    nwb_rd = _ensure_rd(nwb_wegvakken)
    try:
        nwb_union = unary_union([geom for geom in nwb_rd.geometry if geom is not None and not geom.is_empty])
    except Exception:
        return _empty_wegas_comparison()

    rows: list[dict[str, Any]] = []
    for _, row in filtered_wegassen.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            rows.append(
                {
                    "nummer": clean_display_value(row.get("nummer", "")),
                    "naam": clean_display_value(row.get("naam", "")),
                    "Wegnummer": clean_display_value(row.get("Wegnummer", "")),
                    "iasset_lengte_m": None,
                    "afstand_min_tot_nwb_m": None,
                    "afstand_gem_tot_nwb_m": None,
                    "afstand_max_sample_tot_nwb_m": None,
                    "bronkwaliteit": "experimenteel",
                    "status": "controleer",
                    "waarschuwing": "Lege wegasgeometrie.",
                }
            )
            continue

        distances = _sample_geometry_distances(geom, nwb_union)
        min_distance = float(geom.distance(nwb_union)) if nwb_union is not None and not nwb_union.is_empty else None
        avg_distance = float(sum(distances) / len(distances)) if distances else None
        max_sample_distance = float(max(distances)) if distances else None

        warnings: list[str] = []
        if max_sample_distance is not None and max_sample_distance > max_distance_m:
            warnings.append("iASSET-wegas wijkt ruimtelijk af van NWB-wegvakken")

        status = "vergelijking" if not warnings else "controleer"
        rows.append(
            {
                "nummer": clean_display_value(row.get("nummer", "")),
                "naam": clean_display_value(row.get("naam", "")),
                "Wegnummer": clean_display_value(row.get("Wegnummer", "")),
                "iasset_lengte_m": round(float(geom.length), 2),
                "afstand_min_tot_nwb_m": round(min_distance, 2) if min_distance is not None else None,
                "afstand_gem_tot_nwb_m": round(avg_distance, 2) if avg_distance is not None else None,
                "afstand_max_sample_tot_nwb_m": round(max_sample_distance, 2) if max_sample_distance is not None else None,
                "bronkwaliteit": "experimenteel",
                "status": status,
                "waarschuwing": "; ".join(warnings),
            }
        )

    return pd.DataFrame(rows, columns=_empty_wegas_comparison().columns)
