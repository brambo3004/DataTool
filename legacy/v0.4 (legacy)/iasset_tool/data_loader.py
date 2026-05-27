\
"""
Data-inleeslaag voor iASSET-exportbestanden.

Deze module doet drie dingen:
1. bronbestanden lezen;
2. geometrie veilig omzetten van WKT naar Shapely;
3. de tabel voorbereiden voor analyse in RD New (EPSG:28992).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import pandas as pd
from shapely import wkt

from .config import ALL_META_COLS, HIERARCHY_RANK, INPUT_FILES
from .utils import clean_display_value, is_empty_value, normalize_text, parse_date_info, parse_hm_sort


@dataclass
class LoadResult:
    """
    Resultaat van het inlezen.

    Attributes
    ----------
    gdf:
        Geldige objecten met geometrie in EPSG:28992.
    invalid_geometry_rows:
        Rijen waarvan de WKT-geometrie niet bruikbaar was.
    warnings:
        Meldingen die in de UI getoond kunnen worden.
    """

    gdf: gpd.GeoDataFrame
    invalid_geometry_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)


def read_csv_safely(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Lees een CSV-bestand robuust in.

    Waarom deze functie?
    Exports verschillen soms in scheidingsteken. We proberen eerst standaard CSV.
    Als er maar één kolom uitkomt, proberen we opnieuw met puntkomma.
    """
    warnings: list[str] = []

    if not path.exists():
        warnings.append(f"Bestand niet gevonden: {path}")
        return pd.DataFrame(), warnings

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        warnings.append(f"Kon {path} niet lezen met standaardinstellingen: {exc}")
        try:
            df = pd.read_csv(path, sep=";", low_memory=False)
        except Exception as exc2:
            warnings.append(f"Kon {path} ook niet lezen met puntkomma als scheidingsteken: {exc2}")
            return pd.DataFrame(), warnings

    if df.shape[1] == 1:
        try:
            df_semicolon = pd.read_csv(path, sep=";", low_memory=False)
            if df_semicolon.shape[1] > df.shape[1]:
                df = df_semicolon
        except Exception as exc:
            warnings.append(f"Puntkomma-herlezing van {path} is overgeslagen: {exc}")

    return df, warnings


def parse_wkt_geometry(value: object) -> tuple[object | None, str | None]:
    """
    Zet een WKT-string om naar geometrie.

    Foute of lege geometrie geeft geen exception terug naar de app.
    De rij wordt later gelogd in invalid_geometry_rows.
    """
    if is_empty_value(value):
        return None, "Lege WKT-geometrie"

    try:
        geom = wkt.loads(str(value))
    except Exception as exc:
        return None, f"Ongeldige WKT-geometrie: {exc}"

    if geom is None or geom.is_empty:
        return None, "Lege geometrie na WKT-parser"

    return geom, None


def _make_empty_gdf(columns: Iterable[str]) -> gpd.GeoDataFrame:
    """Maak een lege GeoDataFrame met de kolommen die de app verwacht."""
    all_columns = list(dict.fromkeys([*columns, "geometry"]))
    return gpd.GeoDataFrame(columns=all_columns, geometry="geometry", crs="EPSG:28992")


def _ensure_expected_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Zorg dat verwachte kolommen bestaan.

    iASSET-exportbestanden zijn niet altijd identiek. Ontbrekende kolommen vullen
    we met lege tekst, zodat latere modules niet op KeyError stuklopen.
    """
    for col in ALL_META_COLS:
        if col not in gdf.columns:
            gdf[col] = ""

    return gdf


def _prepare_domain_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Maak afgeleide kolommen voor analyse en sortering.
    """
    if "Situering" in gdf.columns:
        situering = gdf["Situering"].astype(str).str.strip().str.title()
        gdf["Situering"] = situering.replace({"Nan": "Onbekend", "None": "Onbekend", "": "Onbekend"})
    else:
        gdf["Situering"] = "Onbekend"

    gdf["subthema_clean"] = gdf["subthema"].apply(normalize_text)
    gdf["Rank"] = gdf["subthema_clean"].apply(lambda value: HIERARCHY_RANK.get(value, 4))

    if "Metrering" in gdf.columns:
        gdf["hm_sort"] = gdf["Metrering"].apply(parse_hm_sort)
    else:
        gdf["hm_sort"] = 99999.9

    if "tijdstipRegistratie" in gdf.columns:
        parsed = gdf["tijdstipRegistratie"].apply(parse_date_info)
        gdf["reg_jaar"] = [item[0] for item in parsed]
        gdf["reg_maand"] = [item[1] for item in parsed]
    else:
        gdf["reg_jaar"] = 0
        gdf["reg_maand"] = 0

    for col in ["Jaar aanleg", "Jaar deklaag"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].apply(clean_display_value)

    return gdf


def load_iasset_data(input_files: Sequence[str | Path] = INPUT_FILES) -> LoadResult:
    """
    Lees de iASSET-exportbestanden en bouw een GeoDataFrame.

    De geometrie komt uit de kolom 'gps coordinaten' en wordt geïnterpreteerd
    als EPSG:4326. Daarna wordt de geometrie omgezet naar EPSG:28992, omdat
    ruimtelijke buffers en afstanden in meters moeten worden berekend.
    """
    warnings: list[str] = []
    frames: list[pd.DataFrame] = []

    for file_path in input_files:
        df_part, file_warnings = read_csv_safely(Path(file_path))
        warnings.extend(file_warnings)
        if not df_part.empty:
            frames.append(df_part)

    if not frames:
        return LoadResult(gdf=_make_empty_gdf(ALL_META_COLS), warnings=warnings)

    df = pd.concat(frames, ignore_index=True)

    # Hernoem bron-id om verwarring met de technische sys_id te voorkomen.
    if "id" in df.columns and "bron_id" not in df.columns:
        df = df.rename(columns={"id": "bron_id"})

    df["sys_id"] = range(len(df))

    if "gps coordinaten" not in df.columns:
        warnings.append("Kolom 'gps coordinaten' ontbreekt. Er kan geen kaartlaag worden opgebouwd.")
        return LoadResult(gdf=_make_empty_gdf([*df.columns, *ALL_META_COLS]), warnings=warnings)

    geometries = []
    invalid_rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        geom, error = parse_wkt_geometry(row.get("gps coordinaten"))
        geometries.append(geom)

        if error:
            invalid_rows.append(
                {
                    "sys_id": row.get("sys_id"),
                    "bron_id": row.get("bron_id", ""),
                    "nummer": row.get("nummer", ""),
                    "Wegnummer": row.get("Wegnummer", ""),
                    "fout": error,
                    "gps coordinaten": row.get("gps coordinaten", ""),
                }
            )

    df["geometry"] = geometries
    df_valid = df[df["geometry"].notna()].copy()

    invalid_geometry_rows = pd.DataFrame(invalid_rows)

    if df_valid.empty:
        warnings.append("Geen geldige geometrieën gevonden.")
        return LoadResult(
            gdf=_make_empty_gdf([*df.columns, *ALL_META_COLS]),
            invalid_geometry_rows=invalid_geometry_rows,
            warnings=warnings,
        )

    gdf = gpd.GeoDataFrame(df_valid, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=28992)

    gdf = _ensure_expected_columns(gdf)
    gdf = _prepare_domain_columns(gdf)

    # sys_id is de stabiele technische sleutel binnen deze run.
    gdf = gdf.set_index("sys_id", drop=False)
    gdf.index.name = None

    return LoadResult(gdf=gdf, invalid_geometry_rows=invalid_geometry_rows, warnings=warnings)
