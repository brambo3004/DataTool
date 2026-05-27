"""
Data-inleeslaag voor iASSET-exportbestanden.

Deze module doet drie dingen:
1. bronbestanden lezen vanaf schijf óf uit een Streamlit-upload;
2. geometrie veilig omzetten van WKT naar Shapely;
3. de tabel voorbereiden voor analyse in RD New (EPSG:28992).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence

import geopandas as gpd
import pandas as pd
from shapely import wkt

from .config import ALL_META_COLS, HIERARCHY_RANK, INPUT_FILES
from .utils import clean_display_value, is_empty_value, normalize_text, parse_date_info, parse_hm_sort


# Een bronbestand kan een pad zijn, of een tuple uit de Streamlit-uploader:
# ("bestandsnaam.csv", b"...inhoud...").
FileInput = str | Path | tuple[str, bytes]


@dataclass(frozen=True)
class ResolvedInputFile:
    """
    Genormaliseerde verwijzing naar een bronbestand.

    path:
        Pad naar een bestand op schijf, bijvoorbeeld de vaste CSV's naast app.py.
    content:
        Bytes uit een upload. Die hoeven dus niet eerst op schijf gezet te worden.
    """

    name: str
    path: Path | None = None
    content: bytes | None = None

    @property
    def suffix(self) -> str:
        """Bestandsextensie in kleine letters, inclusief punt."""
        return Path(self.name).suffix.lower()


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


def _resolve_input_file(input_file: FileInput | Any) -> ResolvedInputFile:
    """
    Zet een pad of uploadbestand om naar één intern formaat.

    We importeren Streamlit hier bewust niet. Een Streamlit UploadedFile heeft
    een ``name`` en ``getvalue()``, en dat is voldoende om hem generiek te lezen.
    """
    if isinstance(input_file, ResolvedInputFile):
        return input_file

    if isinstance(input_file, tuple) and len(input_file) == 2:
        name, content = input_file
        return ResolvedInputFile(name=str(name), content=bytes(content))

    if hasattr(input_file, "name") and hasattr(input_file, "getvalue"):
        return ResolvedInputFile(name=str(input_file.name), content=bytes(input_file.getvalue()))

    path = Path(input_file)
    return ResolvedInputFile(name=str(path), path=path)


def _open_for_pandas(input_file: ResolvedInputFile):
    """
    Geef een object terug dat pandas kan lezen.

    Voor uploads maken we telkens een nieuwe BytesIO, omdat pandas de stream
    tijdens het lezen verplaatst.
    """
    if input_file.content is not None:
        return BytesIO(input_file.content)

    return input_file.path


def read_csv_safely(input_file: FileInput | Any) -> tuple[pd.DataFrame, list[str]]:
    """
    Lees een CSV-bestand robuust in.

    Waarom deze functie?
    Exports verschillen soms in scheidingsteken. We proberen eerst standaard CSV.
    Als er maar één kolom uitkomt, proberen we opnieuw met puntkomma.
    """
    resolved = _resolve_input_file(input_file)
    warnings: list[str] = []

    if resolved.path is not None and not resolved.path.exists():
        warnings.append(f"Bestand niet gevonden: {resolved.name}")
        return pd.DataFrame(), warnings

    try:
        df = pd.read_csv(_open_for_pandas(resolved), low_memory=False)
    except Exception as exc:
        warnings.append(f"Kon {resolved.name} niet lezen met standaard CSV-instellingen: {exc}")
        try:
            df = pd.read_csv(_open_for_pandas(resolved), sep=";", low_memory=False)
        except Exception as exc2:
            warnings.append(f"Kon {resolved.name} ook niet lezen met puntkomma als scheidingsteken: {exc2}")
            return pd.DataFrame(), warnings

    if df.shape[1] == 1:
        try:
            df_semicolon = pd.read_csv(_open_for_pandas(resolved), sep=";", low_memory=False)
            if df_semicolon.shape[1] > df.shape[1]:
                df = df_semicolon
        except Exception as exc:
            warnings.append(f"Puntkomma-herlezing van {resolved.name} is overgeslagen: {exc}")

    return df, warnings


def read_excel_safely(input_file: FileInput | Any) -> tuple[pd.DataFrame, list[str]]:
    """
    Lees een Excelbestand in en kies het meest waarschijnlijke datasheet.

    iASSET-exports horen in ieder geval geometrie te bevatten in de kolom
    ``gps coordinaten``. Als een werkboek meerdere tabbladen heeft, kiezen we
    daarom het tabblad met de meeste verwachte iASSET-kolommen.
    """
    resolved = _resolve_input_file(input_file)
    warnings: list[str] = []

    if resolved.path is not None and not resolved.path.exists():
        warnings.append(f"Bestand niet gevonden: {resolved.name}")
        return pd.DataFrame(), warnings

    try:
        sheets = pd.read_excel(_open_for_pandas(resolved), sheet_name=None)
    except Exception as exc:
        warnings.append(f"Kon Excelbestand {resolved.name} niet lezen: {exc}")
        return pd.DataFrame(), warnings

    if not sheets:
        warnings.append(f"Excelbestand {resolved.name} bevat geen leesbare tabbladen.")
        return pd.DataFrame(), warnings

    expected_lower = {column.lower() for column in ALL_META_COLS}
    expected_lower.update({"gps coordinaten", "wegnummer", "subthema", "onderhoudsproject"})

    best_sheet_name: str | None = None
    best_sheet_df = pd.DataFrame()
    best_score = -1

    for sheet_name, sheet_df in sheets.items():
        if sheet_df.empty:
            continue

        columns_lower = {str(column).strip().lower() for column in sheet_df.columns}
        score = len(columns_lower & expected_lower)

        # De geometriekolom is doorslaggevend, want zonder geometrie kan de app
        # geen kaartobjecten maken.
        if "gps coordinaten" in columns_lower:
            score += 100

        if score > best_score:
            best_sheet_name = str(sheet_name)
            best_sheet_df = sheet_df
            best_score = score

    if best_sheet_name is None:
        warnings.append(f"Excelbestand {resolved.name} bevat geen gevuld tabblad.")
        return pd.DataFrame(), warnings

    if len(sheets) > 1:
        warnings.append(f"Excelbestand {resolved.name}: tabblad '{best_sheet_name}' gebruikt voor import.")

    return best_sheet_df, warnings


def read_table_safely(input_file: FileInput | Any) -> tuple[pd.DataFrame, list[str]]:
    """
    Lees CSV of Excel, afhankelijk van de bestandsextensie.

    Onbekende extensies proberen we eerst als CSV en daarna als Excel.
    """
    resolved = _resolve_input_file(input_file)
    suffix = resolved.suffix

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return read_excel_safely(resolved)

    if suffix in {".csv", ".txt"}:
        return read_csv_safely(resolved)

    df, warnings = read_csv_safely(resolved)
    if not df.empty:
        warnings.append(f"Bestandstype van {resolved.name} is onbekend; succesvol als CSV gelezen.")
        return df, warnings

    df_excel, excel_warnings = read_excel_safely(resolved)
    return df_excel, [*warnings, *excel_warnings]


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


def load_iasset_data(input_files: Sequence[FileInput | Any] = INPUT_FILES) -> LoadResult:
    """
    Lees iASSET-exportbestanden en bouw een GeoDataFrame.

    De geometrie komt uit de kolom ``gps coordinaten`` en wordt geïnterpreteerd
    als EPSG:4326. Daarna wordt de geometrie omgezet naar EPSG:28992, omdat
    ruimtelijke buffers en afstanden in meters moeten worden berekend.

    ``input_files`` mag bestaan uit paden én uit upload-tuples:
    ``("export.csv", b"...")``.
    """
    warnings: list[str] = []
    frames: list[pd.DataFrame] = []

    for input_file in input_files:
        df_part, file_warnings = read_table_safely(input_file)
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
