"""
Data-inleeslaag voor iASSET-exportbestanden.

Deze module doet drie dingen:
1. bronbestanden lezen vanaf schijf óf uit een Streamlit-upload;
2. kolomkoppen uit wisselende iASSET-exports gelijk trekken;
3. geometrie veilig omzetten van WKT naar Shapely/RD New (EPSG:28992).

De functies crashen bewust niet op corrupte of lege rijen. Onbruikbare rijen
komen in ``invalid_geometry_rows`` terecht, zodat de databeheerder kan zien wat
uit de kaartlaag is overgeslagen.
"""

from __future__ import annotations

import math
import re
import unicodedata
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


# Geometriekolommen in voorkeursvolgorde.
# GPS is de normale kaartbron. RD is een veilige fallback als een export geen
# GPS-WKT bevat, maar wel RD-WKT.
GEOMETRY_SOURCE_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("gps coordinaten", "EPSG:4326"),
    ("rds coordinaten", "EPSG:28992"),
)


# Kolomaliasen voor bekende iASSET-varianten. De sleutel is de kolomnaam die de
# rest van de applicatie verwacht. De waarden zijn varianten die we bij import
# naar die sleutel vertalen.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "object id", "object_id", "objectid", "asset id", "asset_id"),
    "bron_id": ("bron id", "bron_id"),
    "gps coordinaten": (
        "gps coordinaten",
        "gps coordinaat",
        "gps coördinaten",
        "gps coördinaat",
        "gpscoordinaten",
        "gpscoordinaat",
        "gps_coordinate",
        "gps coordinates",
        "gps geometrie",
        "gps geometry",
    ),
    "rds coordinaten": (
        "rds coordinaten",
        "rds coordinaat",
        "rd coordinaten",
        "rd coordinaat",
        "rd-coordinaten",
        "rdscoordinaten",
        "rd geometry",
        "rd geometrie",
    ),
    "Wegnummer": ("wegnummer", "weg nummer", "n weg", "n-weg", "route"),
    "subthema": ("subthema", "sub thema"),
    "Onderhoudsproject": (
        "onderhoudsproject",
        "onderhoud project",
        "onderhouds project",
        "onderhoudscomplex",
        "onderhoud complex",
    ),
    "Situering": ("situering",),
    "Metrering": ("metrering", "hectometrering", "hm", "hm waarde", "hm_waarde"),
    "verhardingssoort": ("verhardingssoort", "verhardingsoort", "soort verharding", "soort verharding n"),
    "Soort verharding_N": ("soort verharding_n", "soort verharding n"),
    "Soort deklaag specifiek": ("soort deklaag specifiek", "deklaag specifiek", "soort deklaag"),
    "Jaar aanleg": ("jaar aanleg", "aanlegjaar"),
    "Jaar deklaag": ("jaar deklaag", "deklaagjaar"),
    "Jaar conservering": ("jaar conservering", "conserveringsjaar"),
    "Jaar herstrating": ("jaar herstrating", "herstratingsjaar"),
    "Besteknummer": ("besteknummer", "bestek nummer", "bestek"),
    "tijdstipRegistratie": ("tijdstipregistratie", "tijdstip registratie", "registratietijdstip"),
    "nummer": ("nummer", "object nummer", "objectnummer"),
    "naam": ("naam", "object naam", "objectnaam"),
    "Gebruikersfunctie": ("gebruikersfunctie", "gebruikers functie"),
    "Type onderdeel": ("type onderdeel", "type_onderdeel"),
    "Wegvaknum": ("wegvaknum", "wegvaknummer", "wegvak nummer"),
    "Wegvak V,G": ("wegvak v,g", "wegvak vg"),
    "Wegvaknum V,G": ("wegvaknum v,g", "wegvaknummer v,g", "wegvaknum vg"),
}


EXPECTED_IMPORT_COLUMNS = {
    "gps coordinaten",
    "rds coordinaten",
    "Wegnummer",
    "subthema",
    "Onderhoudsproject",
    "Metrering",
    "nummer",
    "bron_id",
    "id",
}


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


def normalize_column_key(value: Any) -> str:
    """
    Normaliseer een kolomkop voor herkenning.

    iASSET-exports en Excelbewerkingen kunnen verschillen in hoofdletters,
    accenten, underscores, koppeltekens en extra spaties. Voor het herkennen van
    kolommen trekken we die verschillen gelijk. De uiteindelijke DataFrame houdt
    daarna weer nette canonieke kolomnamen.
    """
    if is_empty_value(value):
        return ""

    text = str(value).replace("\ufeff", "").strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_alias_lookup() -> dict[str, str]:
    """Maak een lookup van genormaliseerde alias naar canonieke kolomnaam."""
    lookup: dict[str, str] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in (canonical, *aliases):
            key = normalize_column_key(alias)
            if key:
                lookup[key] = canonical

    return lookup


_ALIAS_LOOKUP = _build_alias_lookup()


def canonical_column_name(value: Any) -> str:
    """
    Vertaal een kolomkop naar de naam die de applicatie verwacht.

    Onbekende kolommen blijven behouden, maar worden wel opgeschoond. Daardoor
    raakt broninformatie niet kwijt wanneer een export extra iASSET-velden bevat.
    """
    key = normalize_column_key(value)
    if key in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[key]

    text = "" if is_empty_value(value) else str(value).replace("\ufeff", "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _make_unique_column_names(columns: Iterable[Any]) -> list[str]:
    """
    Maak kolomnamen uniek zonder informatie weg te gooien.

    Pandas kan duplicate headers toestaan of automatisch ``.1`` toevoegen. Voor
    onze alias-samenvoeging is het veiliger om vooraf expliciet unieke namen te
    maken.
    """
    result: list[str] = []
    seen: dict[str, int] = {}

    for index, column in enumerate(columns):
        name = clean_display_value(column)
        if not name or name.lower().startswith("unnamed:"):
            name = f"onbenoemde_kolom_{index + 1}"

        count = seen.get(name, 0)
        seen[name] = count + 1

        if count:
            result.append(f"{name}_{count + 1}")
        else:
            result.append(name)

    return result


def _score_column_labels(labels: Iterable[Any]) -> int:
    """
    Geef een score aan een mogelijke set kolomkoppen.

    De geometriekolom telt zwaar mee, omdat de kaart zonder geometrie niet kan
    worden opgebouwd. RD-geometrie krijgt ook veel punten, maar GPS blijft de
    voorkeursbron.
    """
    canonical_columns = {canonical_column_name(label) for label in labels}
    score = len(canonical_columns & EXPECTED_IMPORT_COLUMNS)

    if "gps coordinaten" in canonical_columns:
        score += 100
    if "rds coordinaten" in canonical_columns:
        score += 50

    return score


def _score_dataframe_columns(df: pd.DataFrame) -> int:
    """Score helper voor CSV- en Excelbladselectie."""
    if df is None or df.empty and len(df.columns) == 0:
        return 0
    return _score_column_labels(df.columns)


def canonicalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Trek bekende kolomvarianten gelijk en voeg dubbele aliaskolommen samen.

    Waarom samenvoegen?
    Soms bevat een export zowel een oude als nieuwe kolomkop. Dan willen we de
    eerste gevulde waarde behouden in plaats van één van de kolommen blind weg te
    gooien.
    """
    warnings: list[str] = []

    if df is None or df.empty and len(df.columns) == 0:
        return pd.DataFrame(), warnings

    working = df.copy()
    working.columns = _make_unique_column_names(working.columns)

    column_data: dict[str, pd.Series] = {}

    for original_column in working.columns:
        canonical = canonical_column_name(original_column)
        if not canonical:
            canonical = f"onbenoemde_kolom_{len(column_data) + 1}"

        series = working[original_column]

        if canonical not in column_data:
            column_data[canonical] = series.copy()
            if str(original_column).strip() != canonical:
                warnings.append(f"Kolom '{original_column}' herkend als '{canonical}'.")
            continue

        existing = column_data[canonical].copy()
        empty_mask = existing.apply(is_empty_value)
        existing.loc[empty_mask] = series.loc[empty_mask]
        column_data[canonical] = existing
        warnings.append(
            f"Kolom '{original_column}' is samengevoegd met bestaande kolom '{canonical}'."
        )

    return pd.DataFrame(column_data, index=working.index), warnings


def _drop_fully_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verwijder rijen die helemaal leeg zijn.

    Vooral Excelbestanden bevatten soms lege regels vóór of na de exporttabel.
    Die hoeven niet als foutieve objecten in het inleesrapport te verschijnen.
    """
    if df.empty:
        return df

    mask = df.apply(lambda row: all(is_empty_value(value) for value in row), axis=1)
    return df.loc[~mask].copy()


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


def _payload_is_empty(input_file: ResolvedInputFile) -> bool:
    """Controleer of een pad/upload inhoud heeft."""
    if input_file.content is not None:
        return len(input_file.content) == 0

    if input_file.path is not None and input_file.path.exists():
        try:
            return input_file.path.stat().st_size == 0
        except OSError:
            return False

    return False


def _read_csv_candidate(
    input_file: ResolvedInputFile,
    encoding: str,
    separator: str | None,
) -> pd.DataFrame:
    """Lees één CSV-poging met vaste encoding en separator."""
    kwargs: dict[str, Any] = {
        "encoding": encoding,
        "dtype": str,
        "keep_default_na": False,
        "low_memory": False,
    }

    if separator is None:
        # sep=None laat Python de delimiter snuffelen. Dat is nuttig bij
        # wisselende exportinstellingen.
        kwargs["sep"] = None
        kwargs["engine"] = "python"
    else:
        kwargs["sep"] = separator

    return pd.read_csv(_open_for_pandas(input_file), **kwargs)


def read_csv_safely(input_file: FileInput | Any) -> tuple[pd.DataFrame, list[str]]:
    """
    Lees een CSV-bestand robuust in.

    We proberen meerdere encodings en scheidingstekens. De beste kandidaat is de
    tabel met de meeste herkenbare iASSET-kolommen; daarna telt het aantal
    kolommen. Zo kiezen we bij een puntkomma-export niet per ongeluk de
    éénkoloms-variant van de komma-parser.
    """
    resolved = _resolve_input_file(input_file)
    warnings: list[str] = []

    if resolved.path is not None and not resolved.path.exists():
        warnings.append(f"Bestand niet gevonden: {resolved.name}")
        return pd.DataFrame(), warnings

    if _payload_is_empty(resolved):
        warnings.append(f"Bestand {resolved.name} is leeg en is overgeslagen.")
        return pd.DataFrame(), warnings

    encodings = ("utf-8-sig", "utf-8", "cp1252", "latin1")
    separators: tuple[str | None, ...] = (None, ";", ",", "\t")

    best_df = pd.DataFrame()
    best_score = -1
    best_encoding = ""
    best_separator: str | None = None
    errors: list[str] = []

    for encoding in encodings:
        for separator in separators:
            try:
                candidate = _read_csv_candidate(resolved, encoding, separator)
            except Exception as exc:
                errors.append(f"encoding={encoding}, sep={separator!r}: {exc}")
                continue

            score = (_score_dataframe_columns(candidate), len(candidate.columns), len(candidate))
            if score > (best_score, len(best_df.columns), len(best_df)):
                best_df = candidate
                best_score = score[0]
                best_encoding = encoding
                best_separator = separator

    if best_df.empty and len(best_df.columns) == 0:
        detail = errors[0] if errors else "geen leesbare tabel gevonden"
        warnings.append(f"Kon CSV-bestand {resolved.name} niet lezen ({detail}).")
        return pd.DataFrame(), warnings

    if best_encoding not in {"utf-8-sig", "utf-8"}:
        warnings.append(f"CSV-bestand {resolved.name} is gelezen met encoding {best_encoding}.")

    if best_separator is not None and best_separator != ";":
        warnings.append(f"CSV-bestand {resolved.name} is gelezen met scheidingsteken {best_separator!r}.")

    if len(best_df.columns) <= 1:
        warnings.append(
            f"CSV-bestand {resolved.name} lijkt maar één kolom te hebben. "
            "Controleer het scheidingsteken of de exportinstellingen."
        )

    return best_df, warnings


def _excel_sheet_from_raw(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """
    Zoek de meest waarschijnlijke kopregel in een Excel-tabblad.

    iASSET-Excelbestanden hebben soms een infoblok boven de echte tabel. Daarom
    scoren we de eerste regels op herkenbare kolomkoppen en gebruiken we de beste
    regel als header.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(), 0, -1

    max_header_scan_rows = min(25, len(raw_df))
    best_row = -1
    best_score = -1

    for row_index in range(max_header_scan_rows):
        values = list(raw_df.iloc[row_index])
        score = _score_column_labels(values)

        if score > best_score:
            best_score = score
            best_row = row_index

    if best_row < 0 or best_score <= 0:
        return pd.DataFrame(), 0, -1

    columns = _make_unique_column_names(raw_df.iloc[best_row].tolist())
    data = raw_df.iloc[best_row + 1 :].copy()
    data.columns = columns
    data = _drop_fully_empty_rows(data)

    return data, best_score, best_row


def read_excel_safely(input_file: FileInput | Any) -> tuple[pd.DataFrame, list[str]]:
    """
    Lees een Excelbestand in en kies het meest waarschijnlijke datasheet.

    iASSET-exports horen geometrie te bevatten in ``gps coordinaten`` of
    ``rds coordinaten``. Als een werkboek meerdere tabbladen heeft, kiezen we het
    tabblad met de meeste verwachte iASSET-kolommen. Een kopregel hoeft niet per
    se op rij 1 te staan.
    """
    resolved = _resolve_input_file(input_file)
    warnings: list[str] = []

    if resolved.path is not None and not resolved.path.exists():
        warnings.append(f"Bestand niet gevonden: {resolved.name}")
        return pd.DataFrame(), warnings

    if _payload_is_empty(resolved):
        warnings.append(f"Excelbestand {resolved.name} is leeg en is overgeslagen.")
        return pd.DataFrame(), warnings

    try:
        raw_sheets = pd.read_excel(
            _open_for_pandas(resolved),
            sheet_name=None,
            header=None,
            dtype=str,
            keep_default_na=False,
        )
    except Exception as exc:
        warnings.append(f"Kon Excelbestand {resolved.name} niet lezen: {exc}")
        return pd.DataFrame(), warnings

    if not raw_sheets:
        warnings.append(f"Excelbestand {resolved.name} bevat geen leesbare tabbladen.")
        return pd.DataFrame(), warnings

    best_sheet_name: str | None = None
    best_sheet_df = pd.DataFrame()
    best_score = -1
    best_header_row = -1

    for sheet_name, raw_df in raw_sheets.items():
        candidate_df, score, header_row = _excel_sheet_from_raw(raw_df)
        if candidate_df.empty and len(candidate_df.columns) == 0:
            continue

        # Bij gelijke herkenningsscore wint het tabblad met echte datarijen.
        # Dit voorkomt dat een leeg thematabblad met veel kolommen boven het
        # gevulde tabblad "Verhardingen" wordt gekozen.
        score_tuple = (score, len(candidate_df), len(candidate_df.columns))
        best_tuple = (best_score, len(best_sheet_df), len(best_sheet_df.columns))

        if score_tuple > best_tuple:
            best_sheet_name = str(sheet_name)
            best_sheet_df = candidate_df
            best_score = score
            best_header_row = header_row

    if best_sheet_name is None:
        warnings.append(
            f"Excelbestand {resolved.name} bevat geen tabblad met herkenbare iASSET-kolomkoppen."
        )
        return pd.DataFrame(), warnings

    if len(raw_sheets) > 1:
        warnings.append(f"Excelbestand {resolved.name}: tabblad '{best_sheet_name}' gebruikt voor import.")

    if best_header_row > 0:
        warnings.append(
            f"Excelbestand {resolved.name}: kopregel gevonden op rij {best_header_row + 1} "
            f"van tabblad '{best_sheet_name}'."
        )

    return best_sheet_df, warnings


def _prepare_imported_table(df: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Maak een ingelezen tabel klaar voor samenvoegen.

    Hier doen we brononafhankelijke stappen: kolomkoppen gelijk trekken, lege
    regels verwijderen en de bronbestandsnaam toevoegen voor foutmeldingen.
    """
    warnings: list[str] = []
    prepared, column_warnings = canonicalize_columns(df)
    warnings.extend(column_warnings)

    prepared = _drop_fully_empty_rows(prepared)

    if not prepared.empty:
        prepared["bronbestand"] = source_name

    return prepared, warnings


def read_table_safely(input_file: FileInput | Any) -> tuple[pd.DataFrame, list[str]]:
    """
    Lees CSV of Excel, afhankelijk van de bestandsextensie.

    Onbekende extensies proberen we eerst als CSV en daarna als Excel.
    """
    resolved = _resolve_input_file(input_file)
    suffix = resolved.suffix

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        df, warnings = read_excel_safely(resolved)
        prepared, prep_warnings = _prepare_imported_table(df, Path(resolved.name).name)
        return prepared, [*warnings, *prep_warnings]

    if suffix in {".csv", ".txt"}:
        df, warnings = read_csv_safely(resolved)
        prepared, prep_warnings = _prepare_imported_table(df, Path(resolved.name).name)
        return prepared, [*warnings, *prep_warnings]

    df, warnings = read_csv_safely(resolved)
    if not df.empty:
        prepared, prep_warnings = _prepare_imported_table(df, Path(resolved.name).name)
        warnings.append(f"Bestandstype van {resolved.name} is onbekend; succesvol als CSV gelezen.")
        return prepared, [*warnings, *prep_warnings]

    df_excel, excel_warnings = read_excel_safely(resolved)
    prepared, prep_warnings = _prepare_imported_table(df_excel, Path(resolved.name).name)
    return prepared, [*warnings, *excel_warnings, *prep_warnings]


def parse_wkt_geometry(value: object) -> tuple[object | None, str | None]:
    """
    Zet een WKT-string om naar geometrie.

    Foute of lege geometrie geeft geen exception terug naar de app. De rij wordt
    later gelogd in ``invalid_geometry_rows``.

    Ook EWKT zoals ``SRID=4326;POINT (...)`` wordt ondersteund, omdat sommige
    GIS-exportpaden die prefix toevoegen.
    """
    if is_empty_value(value):
        return None, "Lege WKT-geometrie"

    text = clean_display_value(value).strip().strip('"').strip("'")
    if text.upper().startswith("SRID=") and ";" in text:
        text = text.split(";", 1)[1].strip()

    try:
        geom = wkt.loads(text)
    except Exception as exc:
        return None, f"Ongeldige WKT-geometrie: {exc}"

    if geom is None or geom.is_empty:
        return None, "Lege geometrie na WKT-parser"

    return geom, None


def _geometry_bounds_are_finite(geometry: Any) -> bool:
    """Controleer of de geometrie eindige bounds heeft."""
    try:
        bounds = geometry.bounds
    except Exception:
        return False

    if not bounds or len(bounds) != 4:
        return False

    return all(math.isfinite(float(value)) for value in bounds)


def _geometry_fits_declared_crs(geometry: Any, crs: str) -> bool:
    """
    Voorkom dat RD-coördinaten per ongeluk als GPS worden getransformeerd.

    Voor EPSG:4326 moeten x/y binnen lon/lat-grenzen vallen. Voor RD controleren
    we alleen op eindige waarden; iASSET kan objecten buiten de provinciegrens
    bevatten, maar niet met oneindige coördinaten.
    """
    if not _geometry_bounds_are_finite(geometry):
        return False

    if crs != "EPSG:4326":
        return True

    minx, miny, maxx, maxy = geometry.bounds
    return -180 <= minx <= 180 and -180 <= maxx <= 180 and -90 <= miny <= 90 and -90 <= maxy <= 90


def _parse_geometry_from_row(row: pd.Series) -> tuple[object | None, str, str, str | None]:
    """
    Parse de eerste bruikbare geometrie uit een rij.

    Retourneert ``(geometry, bronkolom, bron_crs, foutmelding)``. Als GPS leeg of
    ongeldig is, proberen we RD als fallback. Daardoor blijven bruikbare objecten
    zichtbaar, ook als één geometriekolom corrupt is.
    """
    errors: list[str] = []
    available_geometry_columns = [column for column, _ in GEOMETRY_SOURCE_CANDIDATES if column in row.index]

    if not available_geometry_columns:
        return None, "", "", "Geen geometriekolom gevonden"

    for column, source_crs in GEOMETRY_SOURCE_CANDIDATES:
        if column not in row.index:
            continue

        value = row.get(column)
        if is_empty_value(value):
            errors.append(f"{column}: leeg")
            continue

        geometry, error = parse_wkt_geometry(value)
        if error:
            errors.append(f"{column}: {error}")
            continue

        if not _geometry_fits_declared_crs(geometry, source_crs):
            errors.append(f"{column}: geometrie past niet bij {source_crs}")
            continue

        return geometry, column, source_crs, None

    return None, "", "", "; ".join(errors) if errors else "Geen bruikbare geometrie gevonden"


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

    for col in ["Jaar aanleg", "Jaar deklaag", "Jaar herstrating", "Jaar conservering"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].apply(clean_display_value)

    return gdf


def _warn_for_missing_columns(df: pd.DataFrame) -> list[str]:
    """
    Maak gerichte waarschuwingen voor kolommen die de app hard nodig heeft.
    """
    warnings: list[str] = []

    geometry_columns = [column for column, _ in GEOMETRY_SOURCE_CANDIDATES if column in df.columns]
    if not geometry_columns:
        warnings.append(
            "Kolommen 'gps coordinaten' en 'rds coordinaten' ontbreken. "
            "Er kan geen kaartlaag worden opgebouwd."
        )

    for column in ["Wegnummer", "subthema"]:
        if column not in df.columns:
            warnings.append(f"Kolom '{column}' ontbreekt. De app vult deze leeg, maar analyse kan beperkt zijn.")

    return warnings


def _build_invalid_geometry_row(row: pd.Series, error: str) -> dict[str, object]:
    """Maak een compacte foutregel voor het inleesrapport."""
    return {
        "bronbestand": row.get("bronbestand", ""),
        "sys_id": row.get("sys_id"),
        "bron_id": row.get("bron_id", ""),
        "nummer": row.get("nummer", ""),
        "Wegnummer": row.get("Wegnummer", ""),
        "fout": error,
        "gps coordinaten": row.get("gps coordinaten", ""),
        "rds coordinaten": row.get("rds coordinaten", ""),
    }


def _transform_valid_geometries(
    df_valid: pd.DataFrame,
    warnings: list[str],
    invalid_rows: list[dict[str, object]],
) -> gpd.GeoDataFrame:
    """
    Transformeer geldige geometrieën per bron-CRS naar EPSG:28992.

    Omdat GPS- en RD-fallbackrijen in één upload kunnen voorkomen, transformeren
    we per CRS-groep en voegen we daarna weer samen.
    """
    parts: list[gpd.GeoDataFrame] = []

    for source_crs, part in df_valid.groupby("_geometry_source_crs", sort=False):
        try:
            gdf_part = gpd.GeoDataFrame(part.copy(), geometry="geometry", crs=source_crs)
            if str(source_crs).upper() not in {"EPSG:28992", "28992"}:
                gdf_part = gdf_part.to_crs(epsg=28992)
            else:
                gdf_part = gdf_part.set_crs(epsg=28992, allow_override=True)
            parts.append(gdf_part)
        except Exception as exc:
            warnings.append(f"Transformatie van geometrieën uit {source_crs} is mislukt: {exc}")
            for _, row in part.iterrows():
                invalid_rows.append(_build_invalid_geometry_row(row, f"CRS-transformatie mislukt: {exc}"))

    if not parts:
        return _make_empty_gdf([*df_valid.columns, *ALL_META_COLS])

    combined = pd.concat(parts, ignore_index=False)
    combined = combined.sort_values("sys_id")
    return gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:28992")


def load_iasset_data(input_files: Sequence[FileInput | Any] = INPUT_FILES) -> LoadResult:
    """
    Lees iASSET-exportbestanden en bouw een GeoDataFrame.

    De geometrie komt bij voorkeur uit ``gps coordinaten`` (EPSG:4326). Als die
    ontbreekt of per rij leeg/ongeldig is, gebruikt de loader ``rds coordinaten``
    (EPSG:28992) als fallback. Alle geometrie eindigt in RD New, omdat ruimtelijke
    buffers en afstanden in meters moeten worden berekend.

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
    warnings.extend(_warn_for_missing_columns(df))

    # Hernoem bron-id om verwarring met de technische sys_id te voorkomen.
    if "id" in df.columns and "bron_id" not in df.columns:
        df = df.rename(columns={"id": "bron_id"})
    elif "id" in df.columns and "bron_id" in df.columns:
        empty_bron_id = df["bron_id"].apply(is_empty_value)
        df.loc[empty_bron_id, "bron_id"] = df.loc[empty_bron_id, "id"]
        df = df.drop(columns=["id"])

    df["sys_id"] = range(len(df))

    geometries: list[object | None] = []
    geometry_source_columns: list[str] = []
    geometry_source_crs_values: list[str] = []
    invalid_rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        geometry, source_column, source_crs, error = _parse_geometry_from_row(row)
        geometries.append(geometry)
        geometry_source_columns.append(source_column)
        geometry_source_crs_values.append(source_crs)

        if error:
            invalid_rows.append(_build_invalid_geometry_row(row, error))

    df["geometry"] = geometries
    df["_geometry_source_column"] = geometry_source_columns
    df["_geometry_source_crs"] = geometry_source_crs_values

    df_valid = df[df["geometry"].notna()].copy()
    invalid_geometry_rows = pd.DataFrame(invalid_rows)

    if not df_valid.empty:
        source_counts = df_valid["_geometry_source_column"].value_counts().to_dict()
        if "rds coordinaten" in source_counts:
            warnings.append(
                f"{source_counts['rds coordinaten']} rij(en) met RD-geometrie uit 'rds coordinaten' gebruikt."
            )

    if invalid_rows:
        warnings.append(f"{len(invalid_rows)} rij(en) met ongeldige of lege geometrie overgeslagen.")

    if df_valid.empty:
        warnings.append("Geen geldige geometrieën gevonden.")
        return LoadResult(
            gdf=_make_empty_gdf([*df.columns, *ALL_META_COLS]),
            invalid_geometry_rows=invalid_geometry_rows,
            warnings=warnings,
        )

    gdf = _transform_valid_geometries(df_valid, warnings, invalid_rows)
    invalid_geometry_rows = pd.DataFrame(invalid_rows)

    if gdf.empty:
        warnings.append("Geen geometrieën konden naar EPSG:28992 worden omgezet.")
        return LoadResult(
            gdf=_make_empty_gdf([*df.columns, *ALL_META_COLS]),
            invalid_geometry_rows=invalid_geometry_rows,
            warnings=warnings,
        )

    gdf = _ensure_expected_columns(gdf)
    gdf = _prepare_domain_columns(gdf)

    # sys_id is de stabiele technische sleutel binnen deze run.
    gdf = gdf.set_index("sys_id", drop=False)
    gdf.index.name = None

    return LoadResult(gdf=gdf, invalid_geometry_rows=invalid_geometry_rows, warnings=warnings)
