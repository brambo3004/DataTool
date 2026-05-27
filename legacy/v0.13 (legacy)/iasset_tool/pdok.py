"""
PDOK/NWB-koppeling voor hectometerpunten.

Deze laag is visueel ondersteunend. De projectlogica gebruikt deze punten
nog niet als harde bron voor knippen of sorteren.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import requests


def get_pdok_hectopunten_visual_only(
    road_gdf: gpd.GeoDataFrame,
    buffer_meters: int = 200,
    chunk_size: int = 50,
    timeout_seconds: int = 5,
) -> gpd.GeoDataFrame:
    """
    Haal hectometerpunten op bij PDOK voor de omgeving van de geselecteerde weg.

    Bij netwerkfouten geeft de functie een lege GeoDataFrame terug. De app mag
    hier niet op crashen, omdat PDOK alleen visuele ondersteuning biedt.
    """
    wfs_url = "https://service.pdok.nl/rws/nwbwegen/wfs/v1_0"

    if road_gdf is None or road_gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:28992")

    road_sorted = road_gdf.copy()
    road_sorted["sort_x"] = road_sorted.geometry.centroid.x
    road_sorted = road_sorted.sort_values("sort_x")

    all_features: list[dict] = []

    for start in range(0, len(road_sorted), chunk_size):
        chunk = road_sorted.iloc[start : start + chunk_size]
        minx, miny, maxx, maxy = chunk.total_bounds

        bbox = (
            f"{minx - buffer_meters},"
            f"{miny - buffer_meters},"
            f"{maxx + buffer_meters},"
            f"{maxy + buffer_meters}"
        )

        params = {
            "service": "WFS",
            "version": "1.0.0",
            "request": "GetFeature",
            "typeName": "hectopunten",
            "outputFormat": "json",
            "bbox": bbox,
            "maxFeatures": 5000,
        }

        try:
            response = requests.get(wfs_url, params=params, timeout=timeout_seconds)
            if response.status_code == 200:
                data = response.json()
                all_features.extend(data.get("features") or [])
        except Exception:
            continue

    if not all_features:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:28992")

    result = gpd.GeoDataFrame.from_features(all_features)
    result.set_crs(epsg=28992, inplace=True, allow_override=True)

    hm_col = None
    if "hectometrering" in result.columns:
        hm_col = "hectometrering"
    elif "hectomtrng" in result.columns:
        hm_col = "hectomtrng"

    if "id" in result.columns:
        result = result.drop_duplicates(subset=["id"])
    elif hm_col:
        result = result.drop_duplicates(subset=[hm_col])

    if hm_col:
        result["hm_val"] = pd.to_numeric(result[hm_col], errors="coerce").fillna(0)
    else:
        result["hm_val"] = 0.0

    return result
