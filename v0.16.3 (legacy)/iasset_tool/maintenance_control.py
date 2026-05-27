"""
Controle tussen paspoortexport en onderhoudsexport.

Deze module bevat de eerste Fase-4-controle: objecten uit de paspoortexport
worden naast onderhoudsregels uit de onderhoudsexport gelegd.

Belangrijk ontwerpprincipe:
- Deze module voert géén mutaties uit.
- De controle is bedoeld als veilig overleg-/acceptatiepakket voordat iets in
  iASSET of het maatregeltoetsdocument wordt verwerkt.
- Inlezen is robuust: afwijkende kopregels of lege bestanden leveren
  waarschuwingen op in plaats van crashes.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .config import BACKBONE_TYPES
from .domain import is_maintenance_project_exempt
from .utils import clean_display_value, is_empty_value, normalize_text, parse_hm_sort


# Een bronbestand kan een pad zijn, of een tuple uit de Streamlit-uploader:
# ("bestandsnaam.csv", b"...inhoud...").
FileInput = str | Path | tuple[str, bytes]


@dataclass
class MaintenanceReadResult:
    """Resultaat van het veilig inlezen van onderhoudsexportbestanden."""

    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)


@dataclass
class MaintenanceControlResult:
    """Alle tabellen die de Fase-4-controle oplevert."""

    summary: dict[str, int] = field(default_factory=dict)
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    passport_projects: pd.DataFrame = field(default_factory=pd.DataFrame)
    maintenance_projects: pd.DataFrame = field(default_factory=pd.DataFrame)
    object_differences: pd.DataFrame = field(default_factory=pd.DataFrame)
    action_list: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)


MAINTENANCE_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "objectnummer": (
        "object nr",
        "object nr:",
        "objectnummer",
        "object nummer",
        "nummer",
        "asset nummer",
    ),
    "objectnaam": (
        "object naam",
        "object naam:",
        "naam",
    ),
    "thema": ("thema",),
    "subthema": (
        "subthema",
        "sub thema",
        "subthema:",
    ),
    "project_id": (
        "project id",
        "projectid",
        "onderhoudsproject id",
    ),
    "Onderhoudsproject": (
        "project",
        "project:",
        "onderhoudsproject",
        "onderhoud project",
        "onderhoudscomplex",
        "onderhoud complex",
        "onderhoudscomplex nieuw",
        "onderhoudscomplex nieuw 2025",
        "onderhoudsproject nieuw",
    ),
    "project_type": (
        "type",
        "project type",
        "onderhoudstype",
    ),
    "maatregel": (
        "maatregel",
        "maatregel omschrijving",
        "maatregelomschrijving",
        "maatregel omschrijving:",
        "omschrijving maatregel",
    ),
    "eenheid": (
        "eenheid",
        "eenheid:",
        "hoeveelheid eenheid",
    ),
    "hoeveelheid": (
        "hoeveelh",
        "hoeveelh.",
        "hoeveelheid",
        "aantal",
    ),
    "prijs_per_eenheid": (
        "prijs/eenh",
        "prijs per eenheid",
        "prijs_per_eenheid",
    ),
    "totaalprijs": (
        "totaalprijs",
        "totaal prijs",
    ),
    "opmerking": (
        "opmerking",
        "opmerking:",
        "opmerking2",
        "opmerking 2",
    ),
    "project_jaar_gepland": (
        "project_jaar_gepland",
        "project jaar gepland",
        "jaar gepland",
        "gepland jaar",
        "planjaar",
    ),
}


_EXPECTED_MAINTENANCE_COLUMNS = set(MAINTENANCE_COLUMN_ALIASES)
_EXPECTED_HEADER_KEYS: set[str] = set()
for canonical, aliases in MAINTENANCE_COLUMN_ALIASES.items():
    _EXPECTED_HEADER_KEYS.add(canonical)
    _EXPECTED_HEADER_KEYS.update(aliases)


def _normalize_header_key(value: Any) -> str:
    """Normaliseer een kolomkop voor aliasvergelijking."""
    if is_empty_value(value):
        return ""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("\ufeff", " ").strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _alias_lookup() -> dict[str, str]:
    """Maak een lookup van genormaliseerde kolomkop naar canonieke naam."""
    lookup: dict[str, str] = {}

    for canonical, aliases in MAINTENANCE_COLUMN_ALIASES.items():
        lookup[_normalize_header_key(canonical)] = canonical
        for alias in aliases:
            lookup[_normalize_header_key(alias)] = canonical

    return lookup


_ALIAS_LOOKUP = _alias_lookup()


def _canonical_maintenance_column(value: Any) -> str:
    """Vertaal bekende onderhoudsexport-kolommen naar de appnaam."""
    key = _normalize_header_key(value)
    if key in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[key]

    text = clean_display_value(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _make_unique_column_names(columns: Iterable[Any]) -> list[str]:
    """Maak kolomnamen uniek, zodat dubbele Excelkoppen geen data overschrijven."""
    result: list[str] = []
    seen: dict[str, int] = {}

    for index, value in enumerate(columns):
        name = clean_display_value(value)
        if not name or name.lower().startswith("unnamed"):
            name = f"onbenoemde_kolom_{index + 1}"

        canonical = _canonical_maintenance_column(name)
        count = seen.get(canonical, 0)
        seen[canonical] = count + 1

        if count:
            result.append(f"{canonical}_{count + 1}")
        else:
            result.append(canonical)

    return result


def _drop_fully_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Verwijder lege Excel-/CSV-regels."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    mask = df.apply(lambda row: all(is_empty_value(value) for value in row), axis=1)
    return df.loc[~mask].copy()


def _score_header_values(values: Iterable[Any]) -> int:
    """
    Score een mogelijke kopregel.

    De onderhoudsexport heeft vaak geen geometrie. Daarom scoren we op Project,
    Object nr en Maatregel in plaats van op WKT-kolommen.
    """
    keys = {_normalize_header_key(value) for value in values}
    canonical_hits = {_ALIAS_LOOKUP[key] for key in keys if key in _ALIAS_LOOKUP}

    score = 0
    if "Onderhoudsproject" in canonical_hits:
        score += 100
    if "objectnummer" in canonical_hits:
        score += 40
    if "maatregel" in canonical_hits:
        score += 30
    if "project_id" in canonical_hits:
        score += 15
    if "hoeveelheid" in canonical_hits:
        score += 10
    if "totaalprijs" in canonical_hits:
        score += 10

    score += len(canonical_hits)
    return score


def _table_from_raw_with_header_scan(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """
    Zoek de meest waarschijnlijke kopregel in een onderhoudsexport.

    iASSET-overzichten hebben vaak titelregels boven de echte tabel. We scannen
    de eerste regels en kiezen de regel met Project/Object/Maatregel-kolommen.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(), 0, -1

    max_header_scan_rows = min(40, len(raw_df))
    best_row = -1
    best_score = -1

    for row_index in range(max_header_scan_rows):
        values = list(raw_df.iloc[row_index])
        score = _score_header_values(values)

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


def _resolve_input_file(input_file: FileInput | Any) -> tuple[str, Path | None, bytes | None]:
    """Zet pad/upload om naar naam, pad en optionele bytes."""
    if isinstance(input_file, tuple) and len(input_file) == 2:
        name, content = input_file
        return str(name), None, bytes(content)

    if hasattr(input_file, "name") and hasattr(input_file, "getvalue"):
        return str(input_file.name), None, bytes(input_file.getvalue())

    path = Path(input_file)
    return str(path), path, None


def _open_for_pandas(path: Path | None, content: bytes | None):
    """Geef pandas een pad of nieuwe BytesIO-stream."""
    if content is not None:
        return BytesIO(content)
    return path



def _read_raw_csv_loose(path: Path | None, content: bytes | None, encoding: str, separator: str | None) -> pd.DataFrame:
    """
    Lees CSV-regels zonder te crashen op titelregels met minder kolommen.

    Pandas behandelt wisselende kolomaantallen soms als "bad lines". Juist bij
    iASSET-overzichten hebben de eerste titelregels vaak minder kolommen dan de
    echte tabel. Daarom lezen we de regels hier zelf en vullen we kortere regels
    aan met lege cellen.
    """
    if content is not None:
        text = content.decode(encoding, errors="replace")
    else:
        text = Path(path).read_text(encoding=encoding, errors="replace") if path is not None else ""

    if separator is None:
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=";,\t")
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ";"
    else:
        delimiter = separator

    rows = list(csv.reader(StringIO(text), delimiter=delimiter))
    if not rows:
        return pd.DataFrame()

    width = max(len(row) for row in rows)
    if width == 0:
        return pd.DataFrame()

    padded = [row + [""] * (width - len(row)) for row in rows]
    return pd.DataFrame(padded, dtype=str)


def read_maintenance_csv_safely(input_file: FileInput | Any) -> tuple[pd.DataFrame, list[str]]:
    """Lees een onderhoudsexport uit CSV, inclusief kopregelscan."""
    name, path, content = _resolve_input_file(input_file)
    warnings: list[str] = []

    if path is not None and not path.exists():
        return pd.DataFrame(), [f"Onderhoudsexport niet gevonden: {name}"]

    if content is not None and len(content) == 0:
        return pd.DataFrame(), [f"Onderhoudsexport {Path(name).name} is leeg en is overgeslagen."]

    candidates: list[tuple[int, int, pd.DataFrame, list[str]]] = []

    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        for separator in (";", ",", "\t", None):
            try:
                raw_df = _read_raw_csv_loose(path, content, encoding, separator)
            except Exception:
                continue

            table, score, header_row = _table_from_raw_with_header_scan(raw_df)
            if not table.empty or len(table.columns) > 0:
                candidates.append((score, len(table.columns), table, [f"CSV {Path(name).name}: kopregel gevonden op rij {header_row + 1}."]))

    if not candidates:
        return pd.DataFrame(), [f"CSV {Path(name).name} bevat geen herkenbare onderhoudsexport-kolommen."]

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2], candidates[0][3]


def read_maintenance_excel_safely(input_file: FileInput | Any) -> tuple[pd.DataFrame, list[str]]:
    """Lees een onderhoudsexport uit Excel, inclusief tabblad- en kopregelscan."""
    name, path, content = _resolve_input_file(input_file)
    warnings: list[str] = []

    if path is not None and not path.exists():
        return pd.DataFrame(), [f"Onderhoudsexport niet gevonden: {name}"]

    if content is not None and len(content) == 0:
        return pd.DataFrame(), [f"Onderhoudsexport {Path(name).name} is leeg en is overgeslagen."]

    try:
        excel_file = pd.ExcelFile(_open_for_pandas(path, content))
    except Exception as exc:
        return pd.DataFrame(), [f"Kon onderhoudsexport {Path(name).name} niet lezen: {exc}"]

    candidates: list[tuple[int, int, int, str, pd.DataFrame, list[str]]] = []

    for sheet_name in excel_file.sheet_names:
        try:
            raw_preview = pd.read_excel(
                excel_file,
                sheet_name=sheet_name,
                header=None,
                dtype=str,
                keep_default_na=False,
                nrows=80,
            )
        except Exception as exc:
            warnings.append(f"Onderhoudsexport {Path(name).name}: tabblad '{sheet_name}' kon niet worden gescand ({exc}).")
            continue

        table_preview, score, header_row = _table_from_raw_with_header_scan(raw_preview)
        if table_preview.empty and len(table_preview.columns) == 0:
            continue

        candidates.append((score, len(table_preview), len(table_preview.columns), str(sheet_name), table_preview, [f"Onderhoudsexport {Path(name).name}: tabblad '{sheet_name}' gebruikt."]))
        candidates[-1][5].append(
            f"Onderhoudsexport {Path(name).name}: kopregel gevonden op rij {header_row + 1} van tabblad '{sheet_name}'."
        )

    if not candidates:
        warnings.append(f"Onderhoudsexport {Path(name).name} bevat geen tabblad met herkenbare projectkolommen.")
        return pd.DataFrame(), warnings

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    _score, _rows, _cols, best_sheet_name, _preview, best_warnings = candidates[0]

    try:
        raw_selected = pd.read_excel(
            excel_file,
            sheet_name=best_sheet_name,
            header=None,
            dtype=str,
            keep_default_na=False,
        )
    except Exception as exc:
        return pd.DataFrame(), [*warnings, f"Kon tabblad '{best_sheet_name}' uit onderhoudsexport {Path(name).name} niet volledig lezen: {exc}"]

    table, _score, _header_row = _table_from_raw_with_header_scan(raw_selected)
    table = _drop_fully_empty_rows(table)
    return table, [*warnings, *best_warnings]


def read_maintenance_table_safely(input_file: FileInput | Any) -> tuple[pd.DataFrame, list[str]]:
    """Lees een onderhoudsexportbestand uit CSV of Excel."""
    name, path, content = _resolve_input_file(input_file)
    suffix = Path(name).suffix.lower()

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        df, warnings = read_maintenance_excel_safely((name, content) if content is not None else path or name)
    elif suffix in {".csv", ".txt"}:
        df, warnings = read_maintenance_csv_safely((name, content) if content is not None else path or name)
    else:
        df, warnings = read_maintenance_csv_safely((name, content) if content is not None else path or name)
        if df.empty:
            excel_df, excel_warnings = read_maintenance_excel_safely((name, content) if content is not None else path or name)
            return excel_df, [*warnings, *excel_warnings]
        warnings.append(f"Bestandstype van {Path(name).name} is onbekend; succesvol als CSV gelezen.")

    if not df.empty:
        df = df.copy()
        df["bronbestand_onderhoud"] = Path(name).name

    return df, warnings


def read_maintenance_exports(input_files: Sequence[FileInput | Any]) -> MaintenanceReadResult:
    """Lees één of meer onderhoudsexportbestanden samen in."""
    warnings: list[str] = []
    frames: list[pd.DataFrame] = []

    for input_file in input_files:
        df_part, file_warnings = read_maintenance_table_safely(input_file)
        warnings.extend(file_warnings)
        if not df_part.empty:
            frames.append(df_part)

    if not frames:
        return MaintenanceReadResult(dataframe=pd.DataFrame(), warnings=warnings)

    combined = pd.concat(frames, ignore_index=True)
    return MaintenanceReadResult(dataframe=combined, warnings=warnings)


def normalize_project_name(value: Any) -> str:
    """
    Normaliseer onderhoudsprojectnamen voor vergelijking.

    We zijn bewust conservatief: spaties en streepjesvarianten worden gelijk
    getrokken, maar inhoudelijke verschillen zoals voorloopnullen blijven
    zichtbaar. Zo voorkomen we dat de controle verschillende projecten stil
    samenvoegt.
    """
    if is_empty_value(value):
        return ""

    text = clean_display_value(value).strip()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.upper()


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Geef de eerste bestaande kolom uit een lijst kandidaten terug."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _join_preview(values: Iterable[Any], max_items: int = 6) -> str:
    """Maak een korte, stabiele samenvatting van unieke waarden."""
    cleaned = []
    seen = set()
    for value in values:
        text = clean_display_value(value)
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)

    if len(cleaned) > max_items:
        return ", ".join(cleaned[:max_items]) + f" (+{len(cleaned) - max_items})"
    return ", ".join(cleaned)





def normalize_object_number(value: Any) -> str:
    """
    Normaliseer objectnummers voor Fase-4-vergelijking.

    We gebruiken hoofdletters en verwijderen loze spaties, maar veranderen de
    inhoud niet. Een verkeerd wegnummer in een objectnummer moet dus zichtbaar
    blijven en mag niet door normalisatie verdwijnen.
    """
    if is_empty_value(value):
        return ""

    text = clean_display_value(value).strip().upper()
    if text in {"NAN", "NONE", "NULL", "<NA>"}:
        return ""

    text = re.sub(r"\s+", "", text)
    return text


def _extract_road_from_text(value: Any) -> str:
    """Haal een N-wegnummer uit een project- of objecttekst, als dat aanwezig is."""
    text = clean_display_value(value).upper()
    match = re.search(r"\bN\d{3,4}\b", text)
    return match.group(0) if match else ""


def _hm_value_for_control(row: pd.Series) -> float:
    """
    Bepaal een betrouwbare hectometerwaarde voor Fase-4-samenvattingen.

    Waarom niet blind op ``hm_sort`` vertrouwen?
    De algemene sorteerfallback gebruikt 99999.9 bij ongeldige metrering. Dat is
    handig om de app niet te laten crashen, maar ongeschikt voor rapportage:
    één waarde zoals ``4,,9`` mag de samenvatting niet naar hm 99999.9 trekken.
    """
    if "Metrering" in row.index:
        hm_value = parse_hm_sort(row.get("Metrering"), fallback=float("nan"))
    elif "hm_sort" in row.index:
        hm_value = pd.to_numeric(row.get("hm_sort"), errors="coerce")
    else:
        return float("nan")

    if pd.isna(hm_value):
        return float("nan")

    try:
        hm_float = float(hm_value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")

    # 99999.9 is de generieke sorteerfallback voor ongeldige waarden. Voor deze
    # controle behandelen we zulke waarden als ongeldig en rapporteren we ze los.
    if hm_float >= 90000:
        return float("nan")

    return hm_float


def _prepare_passport_project_rows(passport_df: pd.DataFrame, selected_road: str | None = None) -> pd.DataFrame:
    """Maak paspoortregels klaar voor projectcontrole."""
    if passport_df is None or passport_df.empty or "Onderhoudsproject" not in passport_df.columns:
        return pd.DataFrame()

    working = passport_df.copy()
    if selected_road and "Wegnummer" in working.columns:
        selected_road_text = clean_display_value(selected_road)
        working = working[working["Wegnummer"].astype(str).str.strip() == selected_road_text].copy()

    working["project_norm"] = working["Onderhoudsproject"].apply(normalize_project_name)
    working = working[working["project_norm"] != ""].copy()

    if working.empty:
        return working

    working["_hm_sort_control"] = working.apply(_hm_value_for_control, axis=1)
    working["_hm_valid_control"] = working["_hm_sort_control"].notna()
    return working


def _prepare_maintenance_project_rows(maintenance_df: pd.DataFrame, selected_road: str | None = None) -> pd.DataFrame:
    """Maak onderhoudsregels klaar voor projectcontrole."""
    if maintenance_df is None or maintenance_df.empty:
        return pd.DataFrame()

    project_column = _first_existing_column(maintenance_df, ["Onderhoudsproject", "project"])
    if project_column is None:
        return pd.DataFrame()

    working = maintenance_df.copy()
    working["project_norm"] = working[project_column].apply(normalize_project_name)
    working = working[working["project_norm"] != ""].copy()

    if selected_road:
        selected = clean_display_value(selected_road).upper()
        if selected:
            working = working[working["project_norm"].str.contains(re.escape(selected), na=False)].copy()

    return working


def _object_display_map(values: Iterable[Any]) -> dict[str, str]:
    """Bewaar per genormaliseerd objectnummer de eerste leesbare schrijfwijze."""
    result: dict[str, str] = {}
    for value in values:
        normalized = normalize_object_number(value)
        if not normalized:
            continue
        result.setdefault(normalized, clean_display_value(value))
    return result


def _passport_project_object_maps(
    passport_df: pd.DataFrame,
    selected_road: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Verzamel paspoortobjecten per onderhoudsproject."""
    working = _prepare_passport_project_rows(passport_df, selected_road=selected_road)
    if working.empty:
        return {}

    object_column = _first_existing_column(working, ["nummer", "bron_id", "objectnummer", "sys_id"])
    result: dict[str, dict[str, Any]] = {}

    for project_norm, group in working.groupby("project_norm", dropna=False):
        project_names = [clean_display_value(value) for value in group["Onderhoudsproject"] if clean_display_value(value)]
        project_name = project_names[0] if project_names else project_norm

        object_values = group[object_column] if object_column else group.index.to_series()
        display_map = _object_display_map(object_values)

        invalid_hm_records: list[dict[str, str]] = []
        invalid_group = group[~group["_hm_valid_control"]]
        for row_index, row in invalid_group.iterrows():
            raw_object = row.get(object_column, row_index) if object_column else row_index
            normalized_object = normalize_object_number(raw_object)
            if not normalized_object:
                continue

            invalid_hm_records.append(
                {
                    "objectnummer_norm": normalized_object,
                    "objectnummer": clean_display_value(raw_object),
                    "metrering": clean_display_value(row.get("Metrering", "")),
                }
            )

        result[str(project_norm)] = {
            "onderhoudsproject": project_name,
            "objects": set(display_map),
            "display": display_map,
            "invalid_hm": invalid_hm_records,
        }

    return result


def _maintenance_project_object_maps(
    maintenance_df: pd.DataFrame,
    selected_road: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Verzamel onderhoudsexportobjecten per onderhoudsproject."""
    working = _prepare_maintenance_project_rows(maintenance_df, selected_road=selected_road)
    if working.empty:
        return {}

    project_column = _first_existing_column(working, ["Onderhoudsproject", "project"])
    object_column = _first_existing_column(working, ["objectnummer", "nummer", "bron_id"])
    result: dict[str, dict[str, Any]] = {}

    for project_norm, group in working.groupby("project_norm", dropna=False):
        project_names = [clean_display_value(value) for value in group[project_column] if clean_display_value(value)]
        project_name = project_names[0] if project_names else project_norm

        if object_column:
            object_values = group[object_column]
            display_map = _object_display_map(object_values)
        else:
            display_map = {}

        result[str(project_norm)] = {
            "onderhoudsproject": project_name,
            "objects": set(display_map),
            "display": display_map,
        }

    return result


def build_object_differences(
    passport_df: pd.DataFrame,
    maintenance_df: pd.DataFrame,
    selected_road: str | None = None,
) -> pd.DataFrame:
    """
    Vergelijk objectsets per onderhoudsproject.

    Deze tabel is de verdiepende Fase-4-controle: projectnamen kunnen in beide
    exports bestaan, terwijl de onderliggende objecten toch verschillen.
    """
    passport_map = _passport_project_object_maps(passport_df, selected_road=selected_road)
    maintenance_map = _maintenance_project_object_maps(maintenance_df, selected_road=selected_road)
    keys = sorted(set(passport_map) | set(maintenance_map))
    records: list[dict[str, Any]] = []
    selected_road_text = clean_display_value(selected_road).upper() if selected_road else ""

    for key in keys:
        passport_record = passport_map.get(key, {})
        maintenance_record = maintenance_map.get(key, {})
        project_name = (
            passport_record.get("onderhoudsproject")
            or maintenance_record.get("onderhoudsproject")
            or key
        )

        passport_objects = set(passport_record.get("objects", set()))
        maintenance_objects = set(maintenance_record.get("objects", set()))
        passport_display = passport_record.get("display", {})
        maintenance_display = maintenance_record.get("display", {})

        for object_norm in sorted(passport_objects - maintenance_objects):
            records.append(
                {
                    "onderhoudsproject": project_name,
                    "project_norm": key,
                    "objectnummer": passport_display.get(object_norm, object_norm),
                    "objectnummer_norm": object_norm,
                    "verschiltype": "ALLEEN_IN_PASPOORT",
                    "bron": "paspoortexport",
                    "ernst": "waarschuwing",
                    "object_wegnummer_vermoed": _extract_road_from_text(object_norm),
                    "geselecteerde_weg": selected_road_text,
                    "metrering": "",
                    "melding": "Object staat in de paspoortexport, maar niet in de onderhoudsexport.",
                }
            )

        for object_norm in sorted(maintenance_objects - passport_objects):
            object_road = _extract_road_from_text(object_norm)
            records.append(
                {
                    "onderhoudsproject": project_name,
                    "project_norm": key,
                    "objectnummer": maintenance_display.get(object_norm, object_norm),
                    "objectnummer_norm": object_norm,
                    "verschiltype": "ALLEEN_IN_ONDERHOUD",
                    "bron": "onderhoudsexport",
                    "ernst": "waarschuwing",
                    "object_wegnummer_vermoed": object_road,
                    "geselecteerde_weg": selected_road_text,
                    "metrering": "",
                    "melding": "Object staat in de onderhoudsexport, maar niet in de paspoortexport.",
                }
            )

        for object_norm in sorted(maintenance_objects):
            object_road = _extract_road_from_text(object_norm)
            if selected_road_text and object_road and object_road != selected_road_text:
                records.append(
                    {
                        "onderhoudsproject": project_name,
                        "project_norm": key,
                        "objectnummer": maintenance_display.get(object_norm, object_norm),
                        "objectnummer_norm": object_norm,
                        "verschiltype": "OBJECT_WEGNUMMER_VERDACHT",
                        "bron": "onderhoudsexport",
                        "ernst": "waarschuwing",
                        "object_wegnummer_vermoed": object_road,
                        "geselecteerde_weg": selected_road_text,
                        "metrering": "",
                        "melding": (
                            f"Objectnummer lijkt bij {object_road} te horen, "
                            f"maar de controle draait voor {selected_road_text}."
                        ),
                    }
                )

        for invalid_hm in passport_record.get("invalid_hm", []):
            records.append(
                {
                    "onderhoudsproject": project_name,
                    "project_norm": key,
                    "objectnummer": invalid_hm.get("objectnummer", ""),
                    "objectnummer_norm": invalid_hm.get("objectnummer_norm", ""),
                    "verschiltype": "ONGELDIGE_METRERING_PASPOORT",
                    "bron": "paspoortexport",
                    "ernst": "aandachtspunt",
                    "object_wegnummer_vermoed": _extract_road_from_text(invalid_hm.get("objectnummer_norm", "")),
                    "geselecteerde_weg": selected_road_text,
                    "metrering": invalid_hm.get("metrering", ""),
                    "melding": "Object heeft een ongeldige metrering; genegeerd in hm_min/hm_max.",
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "onderhoudsproject",
                "project_norm",
                "objectnummer",
                "objectnummer_norm",
                "verschiltype",
                "bron",
                "ernst",
                "object_wegnummer_vermoed",
                "geselecteerde_weg",
                "metrering",
                "melding",
            ]
        )

    sort_order = {"waarschuwing": 0, "aandachtspunt": 1, "info": 2, "ok": 3}
    result = pd.DataFrame(records)
    result["_sort_ernst"] = result["ernst"].map(sort_order).fillna(9)
    result = result.sort_values(["_sort_ernst", "onderhoudsproject", "verschiltype", "objectnummer"]).drop(columns=["_sort_ernst"]).reset_index(drop=True)
    return result


def _numeric_score(series: pd.Series) -> int:
    """Tel hoeveel waarden in een kolom numeriek leesbaar zijn."""
    if series is None or series.empty:
        return 0
    numeric = pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return int(numeric.notna().sum())


def _best_numeric_column_near(df: pd.DataFrame, preferred_column: str | None, stop_column: str | None = None) -> str | None:
    """
    Kies de meest waarschijnlijke numerieke kolom rond een bekende kop.

    Sommige iASSET-overzichten hebben een samengestelde hoeveelheidkop: de
    eenheidskolom heeft dan de herkenbare kop, terwijl de echte hoeveelheid in
    de eerstvolgende naamloze kolom staat. Daarom zoeken we heel lokaal naar de
    beste numerieke kandidaat.
    """
    if df is None or df.empty or not preferred_column or preferred_column not in df.columns:
        return preferred_column

    columns = list(df.columns)
    start_index = columns.index(preferred_column)
    stop_index = columns.index(stop_column) if stop_column in columns else min(len(columns), start_index + 4)

    candidate_columns = columns[start_index:stop_index]
    if not candidate_columns:
        return preferred_column

    scored = [
        (_numeric_score(df[column]), -offset, column)
        for offset, column in enumerate(candidate_columns)
    ]
    scored.sort(reverse=True)

    if scored and scored[0][0] > 0:
        return scored[0][2]

    return preferred_column


def _safe_sum(series: pd.Series) -> float:
    """Tel een numerieke kolom veilig op, ook bij komma-decimalen."""
    if series.empty:
        return 0.0

    numeric = pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return float(numeric.fillna(0).sum())


def summarize_passport_projects(passport_df: pd.DataFrame, selected_road: str | None = None) -> pd.DataFrame:
    """Groepeer paspoortobjecten per onderhoudsproject."""
    if passport_df is None or passport_df.empty:
        return pd.DataFrame()

    if "Onderhoudsproject" not in passport_df.columns:
        return pd.DataFrame()

    working = _prepare_passport_project_rows(passport_df, selected_road=selected_road)

    if working.empty:
        return pd.DataFrame()

    object_column = _first_existing_column(working, ["nummer", "bron_id", "objectnummer", "sys_id"])
    subthema_series = working["subthema"].apply(normalize_text) if "subthema" in working.columns else pd.Series("", index=working.index)
    working["_is_primary"] = subthema_series.isin({normalize_text(value) for value in BACKBONE_TYPES})
    working["_is_exempt"] = working.apply(is_maintenance_project_exempt, axis=1)

    records: list[dict[str, Any]] = []

    for project_norm, group in working.groupby("project_norm", dropna=False):
        project_names = [clean_display_value(value) for value in group["Onderhoudsproject"] if clean_display_value(value)]
        project_name = project_names[0] if project_names else project_norm
        object_values = group[object_column] if object_column else group.index.to_series()

        record = {
            "onderhoudsproject": project_name,
            "project_norm": project_norm,
            "paspoort_objecten": int(len(group)),
            "paspoort_unieke_objecten": int(object_values.astype(str).nunique()),
            "paspoort_primaire_objecten": int(group["_is_primary"].sum()),
            "paspoort_secundaire_objecten": int((~group["_is_primary"]).sum()),
            "paspoort_uitzondering_objecten": int(group["_is_exempt"].sum()),
            "paspoort_subthema_samenvatting": _join_preview(group["subthema"]) if "subthema" in group.columns else "",
            "paspoort_objectvoorbeeld": _join_preview(object_values, max_items=8),
            "paspoort_naamvarianten": _join_preview(project_names),
            "paspoort_ongeldige_metrering_aantal": int((~group["_hm_valid_control"]).sum()),
            "paspoort_ongeldige_metrering_objecten": _join_preview(
                object_values[~group["_hm_valid_control"]],
                max_items=8,
            ),
        }

        hm_values = pd.to_numeric(group.loc[group["_hm_valid_control"], "_hm_sort_control"], errors="coerce").dropna()
        if not hm_values.empty:
            record["paspoort_hm_min"] = float(hm_values.min())
            record["paspoort_hm_max"] = float(hm_values.max())
        else:
            record["paspoort_hm_min"] = ""
            record["paspoort_hm_max"] = ""

        if "Wegnummer" in group.columns:
            record["paspoort_wegen"] = _join_preview(group["Wegnummer"])

        records.append(record)

    return pd.DataFrame(records).sort_values(["onderhoudsproject"]).reset_index(drop=True)


def summarize_maintenance_projects(maintenance_df: pd.DataFrame, selected_road: str | None = None) -> pd.DataFrame:
    """Groepeer onderhoudsexportregels per onderhoudsproject."""
    if maintenance_df is None or maintenance_df.empty:
        return pd.DataFrame()

    working = _prepare_maintenance_project_rows(maintenance_df, selected_road=selected_road)

    if working.empty:
        return pd.DataFrame()

    project_column = _first_existing_column(working, ["Onderhoudsproject", "project"])
    object_column = _first_existing_column(working, ["objectnummer", "nummer", "bron_id"])
    measure_column = _first_existing_column(working, ["maatregel", "Maatregel Omschrijving"])
    amount_column = _first_existing_column(working, ["hoeveelheid", "Hoeveelh."])
    price_column = _first_existing_column(working, ["totaalprijs", "Totaalprijs"])
    amount_value_column = _best_numeric_column_near(working, amount_column, stop_column=price_column)

    records: list[dict[str, Any]] = []

    for project_norm, group in working.groupby("project_norm", dropna=False):
        project_names = [clean_display_value(value) for value in group[project_column] if clean_display_value(value)]
        project_name = project_names[0] if project_names else project_norm

        record = {
            "onderhoudsproject": project_name,
            "project_norm": project_norm,
            "onderhoud_regels": int(len(group)),
            "onderhoud_naamvarianten": _join_preview(project_names),
            "onderhoud_bronbestanden": _join_preview(group["bronbestand_onderhoud"]) if "bronbestand_onderhoud" in group.columns else "",
        }

        if object_column:
            object_values = group[object_column]
            record["onderhoud_unieke_objecten"] = int(object_values.astype(str).replace("", pd.NA).dropna().nunique())
            record["onderhoud_objectvoorbeeld"] = _join_preview(object_values, max_items=8)
        else:
            record["onderhoud_unieke_objecten"] = 0
            record["onderhoud_objectvoorbeeld"] = ""

        if measure_column:
            record["onderhoud_unieke_maatregelen"] = int(group[measure_column].astype(str).replace("", pd.NA).dropna().nunique())
            record["onderhoud_maatregel_samenvatting"] = _join_preview(group[measure_column], max_items=6)
        else:
            record["onderhoud_unieke_maatregelen"] = 0
            record["onderhoud_maatregel_samenvatting"] = ""

        record["onderhoud_hoeveelheid_totaal"] = _safe_sum(group[amount_value_column]) if amount_value_column else 0.0
        record["onderhoud_hoeveelheid_bronkolom"] = amount_value_column or ""
        record["onderhoud_totaalprijs"] = _safe_sum(group[price_column]) if price_column else 0.0

        records.append(record)

    return pd.DataFrame(records).sort_values(["onderhoudsproject"]).reset_index(drop=True)


def _difference_counts_for_project(object_differences: pd.DataFrame, project_key: str) -> dict[str, int]:
    """Tel objectverschillen voor één onderhoudsproject."""
    if object_differences is None or object_differences.empty:
        return {
            "objectverschillen_aantal": 0,
            "alleen_in_paspoort": 0,
            "alleen_in_onderhoud": 0,
            "onderhoud_object_wegnummer_verdacht": 0,
            "ongeldige_metrering_paspoort": 0,
        }

    project_diffs = object_differences[object_differences["project_norm"].astype(str) == str(project_key)]
    if project_diffs.empty:
        return {
            "objectverschillen_aantal": 0,
            "alleen_in_paspoort": 0,
            "alleen_in_onderhoud": 0,
            "onderhoud_object_wegnummer_verdacht": 0,
            "ongeldige_metrering_paspoort": 0,
        }

    return {
        "objectverschillen_aantal": int(project_diffs["verschiltype"].isin(["ALLEEN_IN_PASPOORT", "ALLEEN_IN_ONDERHOUD"]).sum()),
        "alleen_in_paspoort": int((project_diffs["verschiltype"] == "ALLEEN_IN_PASPOORT").sum()),
        "alleen_in_onderhoud": int((project_diffs["verschiltype"] == "ALLEEN_IN_ONDERHOUD").sum()),
        "onderhoud_object_wegnummer_verdacht": int((project_diffs["verschiltype"] == "OBJECT_WEGNUMMER_VERDACHT").sum()),
        "ongeldige_metrering_paspoort": int((project_diffs["verschiltype"] == "ONGELDIGE_METRERING_PASPOORT").sum()),
    }


def compare_passport_and_maintenance(
    passport_projects: pd.DataFrame,
    maintenance_projects: pd.DataFrame,
    object_differences: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Leg paspoortprojecten, onderhoudsprojecten en objectverschillen naast elkaar."""
    passport_by_key = {
        str(row["project_norm"]): row.to_dict()
        for _, row in passport_projects.iterrows()
    } if passport_projects is not None and not passport_projects.empty else {}

    maintenance_by_key = {
        str(row["project_norm"]): row.to_dict()
        for _, row in maintenance_projects.iterrows()
    } if maintenance_projects is not None and not maintenance_projects.empty else {}

    keys = sorted(set(passport_by_key) | set(maintenance_by_key))
    records: list[dict[str, Any]] = []

    for key in keys:
        passport_record = passport_by_key.get(key, {})
        maintenance_record = maintenance_by_key.get(key, {})
        in_passport = bool(passport_record)
        in_maintenance = bool(maintenance_record)
        diff_counts = _difference_counts_for_project(object_differences, key)

        project_name = (
            passport_record.get("onderhoudsproject")
            or maintenance_record.get("onderhoudsproject")
            or key
        )

        if in_passport and in_maintenance:
            if diff_counts["onderhoud_object_wegnummer_verdacht"] > 0:
                status = "OBJECT_WEGNUMMER_VERDACHT"
                severity = "waarschuwing"
                message = (
                    "Projectnaam komt in beide exports voor, maar de onderhoudsexport bevat "
                    "objectnummers die bij een ander wegnummer lijken te horen."
                )
            elif diff_counts["objectverschillen_aantal"] > 0:
                status = "OBJECTVERSCHIL"
                severity = "waarschuwing"
                message = (
                    "Projectnaam komt in beide exports voor, maar de objectsets zijn niet gelijk."
                )
            elif diff_counts["ongeldige_metrering_paspoort"] > 0:
                status = "HM_BEREIK_VERDACHT"
                severity = "aandachtspunt"
                message = (
                    "Projectnaam komt in beide exports voor, maar één of meer paspoortobjecten "
                    "hebben een ongeldige metrering. Deze zijn genegeerd in hm_min/hm_max."
                )
            else:
                status = "OK_VOLLEDIG"
                severity = "ok"
                message = "Projectnaam én objectset komen overeen tussen paspoortexport en onderhoudsexport."
        elif in_passport:
            status = "ONTBREEKT_IN_ONDERHOUD"
            severity = "waarschuwing"
            message = "Project staat bij objecten in de paspoortexport, maar ontbreekt in de onderhoudsexport."
        else:
            status = "GEEN_PASPOORTOBJECTEN"
            severity = "waarschuwing"
            message = "Project staat in de onderhoudsexport, maar er zijn geen paspoortobjecten met deze projectnaam."

        record = {
            "status": status,
            "ernst": severity,
            "onderhoudsproject": project_name,
            "project_norm": key,
            "controle_bericht": message,
            "in_paspoortexport": in_passport,
            "in_onderhoudsexport": in_maintenance,
            "projectnaam_in_beide_exports": bool(in_passport and in_maintenance),
            **diff_counts,
        }
        record.update({column: passport_record.get(column, "") for column in passport_projects.columns if column not in {"project_norm", "onderhoudsproject"}} if passport_projects is not None and not passport_projects.empty else {})
        record.update({column: maintenance_record.get(column, "") for column in maintenance_projects.columns if column not in {"project_norm", "onderhoudsproject"}} if maintenance_projects is not None and not maintenance_projects.empty else {})

        records.append(record)

    if not records:
        return pd.DataFrame()

    sort_order = {"waarschuwing": 0, "aandachtspunt": 1, "info": 2, "ok": 3}
    result = pd.DataFrame(records)
    result["_sort_ernst"] = result["ernst"].map(sort_order).fillna(9)
    result = result.sort_values(["_sort_ernst", "onderhoudsproject"]).drop(columns=["_sort_ernst"]).reset_index(drop=True)
    return result


def _objects_for_project_from_diffs(
    object_differences: pd.DataFrame | None,
    project_key: str,
    difference_types: set[str] | None = None,
) -> list[str]:
    """Geef betrokken objectnummers voor één project terug.

    Deze helper houdt de actielijst compact: de databeheerder ziet direct welke
    objecten nagekeken moeten worden, zonder eerst door de technische
    objectverschillenexport te hoeven filteren.
    """
    if object_differences is None or object_differences.empty:
        return []

    project_diffs = object_differences[object_differences["project_norm"].astype(str) == str(project_key)]
    if difference_types:
        project_diffs = project_diffs[project_diffs["verschiltype"].isin(difference_types)]

    objects: list[str] = []
    seen: set[str] = set()
    for value in project_diffs.get("objectnummer", pd.Series(dtype=str)):
        text = clean_display_value(value)
        if not text or text in seen:
            continue
        seen.add(text)
        objects.append(text)

    return objects


def _preview_objects_for_action_list(objects: Iterable[Any], max_items: int = 12) -> str:
    """Maak een objectlijst die lang genoeg is voor controlewerk, maar niet eindeloos."""
    return _join_preview(objects, max_items=max_items)


def _safe_int_value(value: Any, default: int = 0) -> int:
    """Zet een mogelijke lege/NaN-waarde veilig om naar int."""
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    try:
        return int(numeric)
    except (TypeError, ValueError, OverflowError):
        return default


def _practical_category_for_status(status: str, row: dict[str, Any], involved_objects: list[str]) -> str:
    """
    Vertaal een technische status naar een praktische afhandelcategorie.

    Deze categorie is bewust iets breder dan de technische status. De tool kan
    signaleren dat iets verdacht is, maar de databeheerder bepaalt uiteindelijk
    of het bijvoorbeeld een echte fout, een grensgeval of een verklaarbare
    uitzondering is.
    """
    if status == "OBJECT_WEGNUMMER_VERDACHT":
        return "wegnummer_objectpaspoort_of_grensgeval_controleren"
    if status == "ONTBREEKT_IN_ONDERHOUD":
        return "oude_of_ontbrekende_projectnaam_controleren"
    if status == "OBJECTVERSCHIL":
        return "objectset_of_projecttype_controleren"
    if status == "HM_BEREIK_VERDACHT":
        return "metrering_paspoort_corrigeren"
    if status == "GEEN_PASPOORTOBJECTEN":
        return "onderhoudsproject_zonder_paspoortobjecten_controleren"
    return "controlepunt_handmatig_beoordelen"


def _action_text_for_status(status: str, row: dict[str, Any], involved_objects: list[str]) -> tuple[str, str, str, str]:
    """Vertaal een technische Fase-4-status naar begrijpelijke controle-instructies."""
    project_name = clean_display_value(row.get("onderhoudsproject", "")) or "dit onderhoudsproject"
    paspoort_count = _safe_int_value(row.get("paspoort_unieke_objecten", 0))
    onderhoud_count = _safe_int_value(row.get("onderhoud_unieke_objecten", 0))
    only_passport = _safe_int_value(row.get("alleen_in_paspoort", 0))
    only_maintenance = _safe_int_value(row.get("alleen_in_onderhoud", 0))
    wrong_road = _safe_int_value(row.get("onderhoud_object_wegnummer_verdacht", 0))
    invalid_hm = _safe_int_value(row.get("ongeldige_metrering_paspoort", 0))

    if status == "ONTBREEKT_IN_ONDERHOUD":
        category = "Project ontbreekt in onderhoudsexport"
        explanation = (
            f"{project_name} staat bij {paspoort_count} paspoortobject(en), "
            "maar komt niet voor in de onderhoudsexport."
        )
        cause = (
            "Het onderhoudsproject is mogelijk nog niet aangemaakt, niet mee-geëxporteerd, "
            "of de projectnaam wijkt in iASSET net anders af."
        )
        action = (
            "Zoek het onderhoudsproject exact op in iASSET Onderhoud. Controleer daarna of "
            "de projectnaam gelijk gespeld is en of het project in de onderhoudsexportfilter zit."
        )
    elif status == "GEEN_PASPOORTOBJECTEN":
        category = "Onderhoudsproject zonder paspoortobjecten"
        explanation = (
            f"{project_name} staat in de onderhoudsexport met {onderhoud_count} uniek(e) object(en), "
            "maar er zijn geen paspoortobjecten met deze projectnaam."
        )
        cause = (
            "Het project kan verouderd zijn, verkeerd gespeld zijn, of objecten bevatten die niet "
            "in de gebruikte paspoortexport/selectie zitten."
        )
        action = (
            "Controleer of het onderhoudsproject nog actueel is. Zoek de betrokken objectnummers "
            "in iASSET en vergelijk de projectnaam met de paspoortexport."
        )
    elif status == "OBJECT_WEGNUMMER_VERDACHT":
        category = "Objectnummer lijkt bij andere N-weg te horen"
        explanation = (
            f"De onderhoudsexport voor {project_name} bevat {wrong_road} objectnummer(s) "
            "die bij een ander wegnummer lijken te horen."
        )
        cause = (
            "Het object kan foutief aan dit onderhoudsproject hangen, de export kan objecten van "
            "meerdere wegen bevatten, of het objectnummer is historisch/administratief afwijkend."
        )
        action = (
            "Controleer de betrokken objectnummers in iASSET. Bevestig of ze echt bij deze weg en "
            "dit onderhoudsproject horen; corrigeer anders de koppeling of de exportselectie."
        )
    elif status == "OBJECTVERSCHIL":
        category = "Objectsets verschillen"
        explanation = (
            f"{project_name} bestaat in beide exports, maar de objectsets verschillen: "
            f"{only_passport} alleen in paspoort en {only_maintenance} alleen in onderhoud."
        )
        cause = (
            "De paspoortexport en onderhoudsexport zijn mogelijk niet op hetzelfde moment gemaakt, "
            "of één of meer objecten zijn verkeerd gekoppeld."
        )
        action = (
            "Controleer de objectverschillenlijst. Bepaal per object of de paspoortkoppeling of de "
            "onderhoudsexport leidend is en werk daarna iASSET of de exportselectie bij."
        )
    elif status == "HM_BEREIK_VERDACHT":
        category = "Ongeldige metrering in paspoort"
        explanation = (
            f"{project_name} heeft {invalid_hm} paspoortobject(en) met ongeldige metrering. "
            "Die waarden zijn niet gebruikt voor hm_min/hm_max."
        )
        cause = (
            "De metrering bevat waarschijnlijk een typfout of een niet-parseerbare waarde, "
            "bijvoorbeeld een dubbele komma."
        )
        action = (
            "Controleer en corrigeer de metrering van de genoemde objecten in iASSET. Draai daarna "
            "de Fase-4-controle opnieuw om te zien of het hm-bereik weer betrouwbaar is."
        )
    else:
        category = "Controlepunt"
        explanation = clean_display_value(row.get("controle_bericht", "")) or "Controleer dit onderhoudsproject."
        cause = "De technische controle heeft een afwijking gevonden die menselijke beoordeling vraagt."
        action = "Open het onderhoudsproject en de betrokken objecten in iASSET en beoordeel de koppeling."

    return category, explanation, cause, action


def build_action_list(
    comparison: pd.DataFrame,
    object_differences: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Maak een werkbare actielijst uit de Fase-4-controle.

    De vergelijkingstabellen zijn bewust volledig en technisch. Deze actielijst
    vertaalt dezelfde signalen naar controlewerk voor de databeheerder:
    wat controleren, waarom, welke objecten en welke actie ligt voor de hand?
    """
    columns = [
        "onderhoudsproject",
        "status",
        "ernst",
        "controlecategorie",
        "praktische_categorie",
        "aantal_objecten",
        "betrokken_objecten",
        "uitleg",
        "mogelijke_oorzaak",
        "voorgestelde_actie",
        "beoordeling_databeheerder",
        "afhandelstatus",
        "actiehouder",
        "opmerking_afhandeling",
        "project_norm",
    ]

    if comparison is None or comparison.empty:
        return pd.DataFrame(columns=columns)

    records: list[dict[str, Any]] = []
    diff_type_map = {
        "OBJECT_WEGNUMMER_VERDACHT": {"OBJECT_WEGNUMMER_VERDACHT"},
        "OBJECTVERSCHIL": {"ALLEEN_IN_PASPOORT", "ALLEEN_IN_ONDERHOUD"},
        "HM_BEREIK_VERDACHT": {"ONGELDIGE_METRERING_PASPOORT"},
        "ONTBREEKT_IN_ONDERHOUD": {"ALLEEN_IN_PASPOORT"},
        "GEEN_PASPOORTOBJECTEN": {"ALLEEN_IN_ONDERHOUD", "OBJECT_WEGNUMMER_VERDACHT"},
    }

    for _, row_series in comparison.iterrows():
        row = row_series.to_dict()
        status = clean_display_value(row.get("status", ""))
        if status in {"OK", "OK_VOLLEDIG"}:
            continue

        project_key = clean_display_value(row.get("project_norm", ""))
        involved_types = diff_type_map.get(status)
        involved_objects = _objects_for_project_from_diffs(object_differences, project_key, involved_types)

        if not involved_objects and status == "ONTBREEKT_IN_ONDERHOUD":
            involved_objects = [
                value.strip()
                for value in clean_display_value(row.get("paspoort_objectvoorbeeld", "")).split(",")
                if value.strip()
            ]
        elif not involved_objects and status == "GEEN_PASPOORTOBJECTEN":
            involved_objects = [
                value.strip()
                for value in clean_display_value(row.get("onderhoud_objectvoorbeeld", "")).split(",")
                if value.strip()
            ]

        category, explanation, cause, action = _action_text_for_status(status, row, involved_objects)
        practical_category = _practical_category_for_status(status, row, involved_objects)

        records.append(
            {
                "onderhoudsproject": clean_display_value(row.get("onderhoudsproject", "")),
                "status": status,
                "ernst": clean_display_value(row.get("ernst", "")),
                "controlecategorie": category,
                "praktische_categorie": practical_category,
                "aantal_objecten": len(involved_objects),
                "betrokken_objecten": _preview_objects_for_action_list(involved_objects),
                "uitleg": explanation,
                "mogelijke_oorzaak": cause,
                "voorgestelde_actie": action,
                "beoordeling_databeheerder": "",
                "afhandelstatus": "nieuw",
                "actiehouder": "",
                "opmerking_afhandeling": "",
                "project_norm": project_key,
            }
        )

    if not records:
        return pd.DataFrame(columns=columns)

    sort_order = {"waarschuwing": 0, "aandachtspunt": 1, "info": 2, "ok": 3}
    result = pd.DataFrame(records)
    result["_sort_ernst"] = result["ernst"].map(sort_order).fillna(9)
    result = result.sort_values(["_sort_ernst", "onderhoudsproject", "controlecategorie"]).drop(columns=["_sort_ernst"]).reset_index(drop=True)
    return result


def build_maintenance_control(
    passport_df: pd.DataFrame,
    maintenance_df: pd.DataFrame,
    selected_road: str | None = None,
) -> MaintenanceControlResult:
    """
    Bouw de volledige Fase-4-controle.

    Deze functie is bewust pandas-only en kent Streamlit niet. Daardoor kunnen we
    de controle later ook als commandline- of batchcontrole gebruiken.
    """
    warnings: list[str] = []

    passport_projects = summarize_passport_projects(passport_df, selected_road=selected_road)
    maintenance_projects = summarize_maintenance_projects(maintenance_df, selected_road=selected_road)

    if passport_projects.empty:
        warnings.append("Geen onderhoudsprojecten gevonden in de paspoortexport voor deze selectie.")
    if maintenance_projects.empty:
        warnings.append("Geen onderhoudsprojecten gevonden in de onderhoudsexport voor deze selectie.")

    object_differences = build_object_differences(passport_df, maintenance_df, selected_road=selected_road)
    comparison = compare_passport_and_maintenance(passport_projects, maintenance_projects, object_differences)
    action_list = build_action_list(comparison, object_differences)

    if comparison.empty:
        summary = {
            "projecten_totaal": 0,
            "projecten_ok": 0,
            "ok_volledig": 0,
            "ok_projectnaam": 0,
            "objectverschillen": 0,
            "hm_bereik_verdacht": 0,
            "object_wegnummer_verdacht": 0,
            "ontbreekt_in_onderhoud": 0,
            "geen_paspoortobjecten": 0,
            "waarschuwingen": 0,
            "aandachtspunten": 0,
            "objectverschillen_regels": 0,
            "acties": 0,
        }
    else:
        summary = {
            "projecten_totaal": int(len(comparison)),
            "projecten_ok": int((comparison["status"] == "OK_VOLLEDIG").sum()),
            "ok_volledig": int((comparison["status"] == "OK_VOLLEDIG").sum()),
            "ok_projectnaam": int(comparison["projectnaam_in_beide_exports"].sum()),
            "objectverschillen": int((comparison["status"] == "OBJECTVERSCHIL").sum()),
            "hm_bereik_verdacht": int((comparison["status"] == "HM_BEREIK_VERDACHT").sum()),
            "object_wegnummer_verdacht": int((comparison["status"] == "OBJECT_WEGNUMMER_VERDACHT").sum()),
            "ontbreekt_in_onderhoud": int((comparison["status"] == "ONTBREEKT_IN_ONDERHOUD").sum()),
            "geen_paspoortobjecten": int((comparison["status"] == "GEEN_PASPOORTOBJECTEN").sum()),
            "waarschuwingen": int((comparison["ernst"] == "waarschuwing").sum()),
            "aandachtspunten": int((comparison["ernst"] == "aandachtspunt").sum()),
            "objectverschillen_regels": int(len(object_differences)),
            "acties": int(len(action_list)),
        }

    return MaintenanceControlResult(
        summary=summary,
        comparison=comparison,
        passport_projects=passport_projects,
        maintenance_projects=maintenance_projects,
        object_differences=object_differences,
        action_list=action_list,
        warnings=warnings,
    )
