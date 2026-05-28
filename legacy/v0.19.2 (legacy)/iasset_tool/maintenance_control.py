"""
Controle tussen paspoortexport en onderhoudsexport.

Deze module bevat de onderhoudscontrole: objecten uit de paspoortexport
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
import copy
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

# Kolommen die door de databeheerder mogen worden ingevuld en die in v0.16.4
# opnieuw kunnen worden meegenomen bij een nieuwe controle.
ACTION_FOLLOW_UP_COLUMNS: tuple[str, ...] = (
    "beoordeling_databeheerder",
    "afhandelstatus",
    "actiehouder",
    "opmerking_afhandeling",
)

# Standaardwaarden voor de werkvoorraad in de Streamlit-app. Vrije tekst blijft
# toegestaan, maar deze waarden geven de databeheerder een gedeelde taal.
ACTION_FOLLOW_UP_STATUS_OPTIONS: tuple[str, ...] = (
    "nieuw",
    "in onderzoek",
    "te corrigeren in paspoort",
    "te corrigeren in onderhoud",
    "verklaarbare uitzondering",
    "afgehandeld",
)

ACTION_WORK_QUEUE_DISPLAY_COLUMNS: tuple[str, ...] = (
    "wegnummer",
    "onderhoudsproject",
    "ernst",
    "status",
    "praktische_categorie",
    "duiding",
    "duiding_groep",
    "duiding_uitleg",
    "voortgang_status",
    "voortgang_uitleg",
    "aantal_objecten",
    "betrokken_objecten",
    "mogelijke_onderhoudsmatch",
    "onderhoudsmatch_type",
    "onderhoudsmatch_uitleg",
    "beoordeling_databeheerder",
    "afhandelstatus",
    "actiehouder",
    "opmerking_afhandeling",
    "controlecategorie",
    "uitleg",
    "mogelijke_oorzaak",
    "voorgestelde_actie",
    "project_norm",
)


MUTATION_SUGGESTION_COLUMNS: tuple[str, ...] = (
    "wegnummer",
    "onderhoudsproject",
    "voorstelstatus",
    "voorsteltype",
    "ernst",
    "duiding",
    "duiding_groep",
    "duiding_uitleg",
    "bron_export",
    "objectnummer",
    "veld",
    "huidige_waarde",
    "voorgestelde_waarde",
    "zekerheid",
    "toelichting",
    "voorgestelde_controle",
    "alleen_na_controle",
    "menselijke_controle_verplicht",
    "automatisch_doorvoeren",
    "veiligheidsmelding",
    "project_norm",
)

ACTION_MATCH_COLUMNS: tuple[str, ...] = (
    "project_norm",
    "status",
    "controlecategorie",
    "praktische_categorie",
)

ACTION_PROGRESS_COLUMNS: tuple[str, ...] = (
    "voortgang_status",
    "voortgang_uitleg",
)

ACTION_LIST_COLUMN_ALIASES: dict[str, str] = {
    "beoordeling databeheerder": "beoordeling_databeheerder",
    "beoordeling_databeheerder": "beoordeling_databeheerder",
    "afhandelstatus": "afhandelstatus",
    "afhandel status": "afhandelstatus",
    "actiehouder": "actiehouder",
    "actie houder": "actiehouder",
    "opmerking afhandeling": "opmerking_afhandeling",
    "opmerking_afhandeling": "opmerking_afhandeling",
    "praktische categorie": "praktische_categorie",
    "praktische_categorie": "praktische_categorie",
    "duiding": "duiding",
    "duiding groep": "duiding_groep",
    "duiding_groep": "duiding_groep",
    "duiding uitleg": "duiding_uitleg",
    "duiding_uitleg": "duiding_uitleg",
    "controle categorie": "controlecategorie",
    "controlecategorie": "controlecategorie",
    "project norm": "project_norm",
    "project_norm": "project_norm",
    "wegnummer": "wegnummer",
    "weg": "wegnummer",
    "voortgang status": "voortgang_status",
    "voortgang_status": "voortgang_status",
    "voortgang uitleg": "voortgang_uitleg",
    "voortgang_uitleg": "voortgang_uitleg",
}


def _canonical_action_list_column(value: Any) -> str:
    """Vertaal handmatig aangepaste actielijst-kolommen naar de appnaam."""
    text = clean_display_value(value).strip()
    key = _normalize_header_key(text)
    return ACTION_LIST_COLUMN_ALIASES.get(key, text)




@dataclass
class MaintenanceReadResult:
    """Resultaat van het veilig inlezen van onderhoudsexportbestanden."""

    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)


@dataclass
class MaintenanceControlResult:
    """Alle tabellen die de onderhoudscontrole oplevert."""

    summary: dict[str, int] = field(default_factory=dict)
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    passport_projects: pd.DataFrame = field(default_factory=pd.DataFrame)
    maintenance_projects: pd.DataFrame = field(default_factory=pd.DataFrame)
    object_differences: pd.DataFrame = field(default_factory=pd.DataFrame)
    action_list: pd.DataFrame = field(default_factory=pd.DataFrame)
    mutation_suggestions: pd.DataFrame = field(default_factory=pd.DataFrame)
    resolved_actions: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProjectNameParts:
    """Genormaliseerde onderdelen uit een onderhoudsprojectnaam."""

    road: str = ""
    category: str = ""
    hm_start: float | None = None
    hm_end: float | None = None


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


def read_action_list_safely(input_file: FileInput | Any) -> tuple[pd.DataFrame, list[str]]:
    """
    Lees een eerder ingevulde Onderhoudscontrole-actielijst veilig in.

    Waarom apart van de onderhoudsexport?
    De actielijst is geen iASSET-bronbestand, maar een werkdocument. We lezen
    daarom alleen de tabel en proberen geen onderhoudsproject-kopregels te
    interpreteren.
    """
    name, path, content = _resolve_input_file(input_file)
    warnings: list[str] = []
    suffix = Path(name).suffix.lower()

    if path is not None and not path.exists():
        return pd.DataFrame(), [f"Eerdere actielijst niet gevonden: {name}"]
    if content is not None and len(content) == 0:
        return pd.DataFrame(), [f"Eerdere actielijst {Path(name).name} is leeg en is overgeslagen."]

    try:
        if suffix in {".xlsx", ".xls", ".xlsm"}:
            df = pd.read_excel(_open_for_pandas(path, content), dtype=str, keep_default_na=False)
        else:
            # Actielijsten worden door de app als puntkomma-CSV geëxporteerd.
            # We proberen eerst die vorm en vallen daarna terug op komma-CSV.
            last_exc: Exception | None = None
            df = pd.DataFrame()
            for encoding in ("utf-8-sig", "utf-8", "latin1"):
                for separator in (";", ",", "\t"):
                    try:
                        df_candidate = pd.read_csv(
                            _open_for_pandas(path, content),
                            sep=separator,
                            dtype=str,
                            keep_default_na=False,
                            encoding=encoding,
                        )
                    except Exception as exc:
                        last_exc = exc
                        continue

                    if len(df_candidate.columns) > 1:
                        df = df_candidate
                        break
                if not df.empty or len(df.columns) > 1:
                    break

            if df.empty and len(df.columns) <= 1 and last_exc is not None:
                raise last_exc
    except Exception as exc:
        return pd.DataFrame(), [f"Kon eerdere actielijst {Path(name).name} niet lezen: {exc}"]

    if df is None or df.empty:
        return pd.DataFrame(), [f"Eerdere actielijst {Path(name).name} bevat geen regels."]

    df = df.copy()
    df.columns = [_canonical_action_list_column(column) for column in df.columns]
    df = _drop_fully_empty_rows(df)

    required = {"onderhoudsproject", "status"}
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        warnings.append(
            "Eerdere actielijst mist herkenbare kolommen: "
            + ", ".join(missing_required)
            + ". Beoordelingen worden niet overgenomen."
        )
        return pd.DataFrame(), warnings

    missing_follow_up = [column for column in ACTION_FOLLOW_UP_COLUMNS if column not in df.columns]
    if missing_follow_up:
        warnings.append(
            "Eerdere actielijst mist opvolgkolommen: "
            + ", ".join(missing_follow_up)
            + ". Alleen beschikbare opvolgvelden worden meegenomen."
        )

    warnings.append(f"Eerdere actielijst {Path(name).name}: {len(df)} regel(s) gelezen.")
    return df, warnings


def read_action_lists_safely(input_files: Sequence[FileInput | Any]) -> tuple[pd.DataFrame, list[str]]:
    """Lees één of meer eerder ingevulde actielijsten samen in."""
    warnings: list[str] = []
    frames: list[pd.DataFrame] = []

    for input_file in input_files:
        df_part, file_warnings = read_action_list_safely(input_file)
        warnings.extend(file_warnings)
        if df_part is not None and not df_part.empty:
            frames.append(df_part)

    if not frames:
        return pd.DataFrame(), warnings

    return pd.concat(frames, ignore_index=True), warnings


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



def parse_project_name_parts(value: Any) -> ProjectNameParts:
    """
    Haal wegnummer, projectcategorie en hm-bereik uit een onderhoudsprojectnaam.

    Voorbeelden die we ondersteunen:
    - N354-HRB-11.5-12.8
    - N354-PWR-21,5-21,6
    - N353-HRBR-05.6-07.8

    Als de naam afwijkt, geven we een leeg object terug. De controle mag nooit
    crashen op handmatig getypte projectnamen.
    """
    text = normalize_project_name(value)
    if not text:
        return ProjectNameParts()

    match = re.search(
        r"\b(N\d{3,4})-([A-Z0-9]+)-(\d+(?:[.,]\d+)?)-(\d+(?:[.,]\d+)?)\b",
        text,
    )
    if not match:
        return ProjectNameParts(road=_extract_road_from_text(text))

    start = parse_hm_sort(match.group(3), fallback=float("nan"))
    end = parse_hm_sort(match.group(4), fallback=float("nan"))
    if pd.isna(start) or pd.isna(end):
        return ProjectNameParts(road=match.group(1), category=match.group(2))

    hm_start, hm_end = sorted((float(start), float(end)))
    return ProjectNameParts(
        road=match.group(1),
        category=match.group(2),
        hm_start=hm_start,
        hm_end=hm_end,
    )


def _project_category_family(category: Any) -> str:
    """
    Groepeer projectcategorieën voor voorzichtige matchvoorstellen.

    HRBR/HRBL horen bij de HRB-familie, PWR/PWL bij PW, enzovoort. Exacte
    categorie krijgt in de score nog steeds voorrang; deze familie is alleen een
    terugval om nuttige suggesties niet te missen.
    """
    text = clean_display_value(category).upper()
    if text.startswith("HRB"):
        return "HRB"
    if text.startswith("PWR") or text.startswith("PWL") or text == "PW":
        return "PW"
    if text.startswith("FPR") or text.startswith("FPL") or text == "FP":
        return "FP"
    if text.startswith("LBP"):
        return "LBP"
    if text.startswith("BB"):
        return "BB"
    return re.sub(r"[LR]$", "", text)


def _hm_overlap_and_gap(a: ProjectNameParts, b: ProjectNameParts) -> tuple[float, float]:
    """Bereken hm-overlap en hm-afstand tussen twee projectnamen."""
    if a.hm_start is None or a.hm_end is None or b.hm_start is None or b.hm_end is None:
        return 0.0, float("inf")

    overlap = max(0.0, min(a.hm_end, b.hm_end) - max(a.hm_start, b.hm_start))
    if overlap > 0:
        return overlap, 0.0

    gap = max(0.0, max(a.hm_start, b.hm_start) - min(a.hm_end, b.hm_end))
    return 0.0, gap


def _suggest_maintenance_project_match(
    missing_project_name: Any,
    maintenance_projects: pd.DataFrame | None,
    *,
    max_suggestions: int = 3,
) -> dict[str, Any]:
    """
    Zoek een mogelijke bestaande onderhoudsprojectnaam voor een ontbrekende paspoortnaam.

    Dit is géén automatische correctie. Het is alleen een leesbare hint voor de
    databeheerder wanneer een project in de paspoortexport ontbreekt in de
    onderhoudsexport. We vergelijken bewust conservatief op wegnummer,
    projectcategorie en hm-bereik.
    """
    empty = {
        "mogelijke_onderhoudsmatch": "",
        "onderhoudsmatch_type": "",
        "onderhoudsmatch_score": 0,
        "onderhoudsmatch_uitleg": "",
    }

    parts = parse_project_name_parts(missing_project_name)
    if not parts.road or maintenance_projects is None or maintenance_projects.empty:
        return empty

    candidates: list[dict[str, Any]] = []

    for _, candidate_row in maintenance_projects.iterrows():
        candidate_name = clean_display_value(candidate_row.get("onderhoudsproject", ""))
        candidate_parts = parse_project_name_parts(candidate_name)
        if not candidate_parts.road or candidate_parts.road != parts.road:
            continue

        exact_category = bool(parts.category and candidate_parts.category and parts.category == candidate_parts.category)
        same_family = bool(
            parts.category
            and candidate_parts.category
            and _project_category_family(parts.category) == _project_category_family(candidate_parts.category)
        )
        if parts.category and candidate_parts.category and not (exact_category or same_family):
            continue

        overlap, gap = _hm_overlap_and_gap(parts, candidate_parts)
        score = 40  # zelfde weg
        match_type = "zelfde_weg"

        if exact_category:
            score += 30
            match_type = "zelfde_categorie"
        elif same_family:
            score += 20
            match_type = "zelfde_categoriefamilie"

        if overlap > 0:
            score += 30
            match_type = "hm_overlap_zelfde_categorie" if exact_category else "hm_overlap_zelfde_categoriefamilie"
        elif gap <= 0.2:
            score += 18
            match_type = "hm_aangrenzend_zelfde_categorie" if exact_category else "hm_aangrenzend_zelfde_categoriefamilie"
        elif gap <= 1.0:
            score += 8
            match_type = "hm_dichtbij_zelfde_categorie" if exact_category else "hm_dichtbij_zelfde_categoriefamilie"

        if score < 70:
            continue

        candidates.append(
            {
                "project": candidate_name,
                "score": int(score),
                "type": match_type,
                "overlap": overlap,
                "gap": gap,
            }
        )

    if not candidates:
        return empty

    candidates.sort(key=lambda item: (item["score"], item["overlap"], -item["gap"], item["project"]), reverse=True)
    selected = candidates[:max_suggestions]
    match_text = "; ".join(f"{item['project']} ({item['type']}, score {item['score']})" for item in selected)
    best = selected[0]

    if best["overlap"] > 0:
        uitleg = (
            f"Mogelijke match op basis van hetzelfde wegnummer, vergelijkbare categorie en "
            f"overlappend hm-bereik: {best['project']}."
        )
    elif best["gap"] != float("inf"):
        uitleg = (
            f"Mogelijke match op basis van hetzelfde wegnummer, vergelijkbare categorie en "
            f"nabij hm-bereik: {best['project']}."
        )
    else:
        uitleg = f"Mogelijke match op basis van hetzelfde wegnummer en vergelijkbare categorie: {best['project']}."

    return {
        "mogelijke_onderhoudsmatch": match_text,
        "onderhoudsmatch_type": best["type"],
        "onderhoudsmatch_score": best["score"],
        "onderhoudsmatch_uitleg": uitleg,
    }


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


def _first_project_from_match_text(value: Any) -> str:
    """
    Haal de eerste projectnaam uit een mogelijke-matchtekst.

    ``mogelijke_onderhoudsmatch`` kan meerdere kandidaten bevatten, inclusief
    score-uitleg tussen haakjes. Voor mutatievoorstellen willen we alleen de
    projectnaam als voorgestelde waarde tonen.
    """
    text = clean_display_value(value)
    if not text:
        return ""

    first = text.split(";")[0].strip()
    first = re.sub(r"\s+\([^)]*\)\s*$", "", first).strip()
    return first






def normalize_object_number(value: Any) -> str:
    """
    Normaliseer objectnummers voor onderhoudscontrole-vergelijking.

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


def _road_from_project_row(row: dict[str, Any] | pd.Series | None, project_name: Any = "") -> str:
    """
    Bepaal het wegnummer voor een onderhoudscontrole-regel.

    Bij een losse wegcontrole komt het wegnummer vaak uit de selectie in de app.
    Bij een netwerkbrede controle moet de tool het wegnummer zelf herleiden uit
    de projectnaam of uit de paspoort-samenvatting. Dit voorkomt dat een
    areaalbrede export minder streng wordt dan een losse wegcontrole.
    """
    if row is not None:
        getter = row.get if hasattr(row, "get") else lambda key, default="": default
        for column in ("wegnummer", "paspoort_wegen", "geselecteerde_weg"):
            value = clean_display_value(getter(column, ""))
            road = _extract_road_from_text(value)
            if road:
                return road

    road = _extract_road_from_text(project_name)
    return road


def _expected_road_for_project(project_name: Any, selected_road: str | None = None) -> str:
    """Bepaal tegen welk wegnummer objectnummers gecontroleerd moeten worden."""
    selected = clean_display_value(selected_road).upper() if selected_road else ""
    if selected:
        return selected
    return _extract_road_from_text(project_name)


def _hm_value_for_control(row: pd.Series) -> float:
    """
    Bepaal een betrouwbare hectometerwaarde voor onderhoudscontrole-samenvattingen.

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

    Deze tabel is de verdiepende onderhoudscontrole: projectnamen kunnen in beide
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
        expected_road = _expected_road_for_project(project_name, selected_road=selected_road)

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
                    "geselecteerde_weg": expected_road,
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
                    "geselecteerde_weg": expected_road,
                    "metrering": "",
                    "melding": "Object staat in de onderhoudsexport, maar niet in de paspoortexport.",
                }
            )

        for object_norm in sorted(maintenance_objects):
            object_road = _extract_road_from_text(object_norm)
            if expected_road and object_road and object_road != expected_road:
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
                        "geselecteerde_weg": expected_road,
                        "metrering": "",
                        "melding": (
                            f"Objectnummer lijkt bij {object_road} te horen, "
                            f"maar het onderhoudsproject hoort bij {expected_road}."
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
                    "geselecteerde_weg": expected_road,
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
            "wegnummer": _road_from_project_row({"paspoort_wegen": _join_preview(group["Wegnummer"]) if "Wegnummer" in group.columns else ""}, project_name),
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
            "wegnummer": _road_from_project_row({}, project_name),
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

        match_suggestion = (
            _suggest_maintenance_project_match(project_name, maintenance_projects)
            if status == "ONTBREEKT_IN_ONDERHOUD"
            else {
                "mogelijke_onderhoudsmatch": "",
                "onderhoudsmatch_type": "",
                "onderhoudsmatch_score": 0,
                "onderhoudsmatch_uitleg": "",
            }
        )

        record = {
            "status": status,
            "ernst": severity,
            "wegnummer": _road_from_project_row(passport_record or maintenance_record, project_name),
            "onderhoudsproject": project_name,
            "project_norm": key,
            "controle_bericht": message,
            "in_paspoortexport": in_passport,
            "in_onderhoudsexport": in_maintenance,
            "projectnaam_in_beide_exports": bool(in_passport and in_maintenance),
            **diff_counts,
            **match_suggestion,
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


def _normalize_action_match_value(value: Any) -> str:
    """Normaliseer één waarde voor het herkennen van dezelfde actieregel."""
    return re.sub(r"\s+", " ", clean_display_value(value)).strip().upper()


def _action_match_key(
    row: dict[str, Any] | pd.Series,
    *,
    include_practical_category: bool = True,
) -> tuple[str, str, str, str]:
    """
    Maak een stabiele sleutel voor een onderhoudscontrole-actieregel.

    We matchen bewust niet op vrije tekst zoals uitleg of voorgestelde actie,
    want die teksten kunnen tussen versies verbeteren. De combinatie project,
    status en categorie is stabiel genoeg voor de huidige actielijst waarin
    maximaal één actie per project/status staat.
    """
    if isinstance(row, pd.Series):
        row_dict = row.to_dict()
    else:
        row_dict = dict(row)

    project_key = clean_display_value(row_dict.get("project_norm", ""))
    if not project_key:
        project_key = normalize_project_name(row_dict.get("onderhoudsproject", ""))

    practical_category = (
        _normalize_action_match_value(row_dict.get("praktische_categorie", ""))
        if include_practical_category
        else ""
    )

    return (
        _normalize_action_match_value(project_key),
        _normalize_action_match_value(row_dict.get("status", "")),
        _normalize_action_match_value(row_dict.get("controlecategorie", "")),
        practical_category,
    )


def _action_match_candidate_keys(row: dict[str, Any] | pd.Series) -> list[tuple[str, str, str, str]]:
    """
    Geef strikte én terugval-matchsleutels.

    Hiermee kunnen actielijsten uit v0.16.2 of handmatig aangepaste Excel/CSV's
    zonder ``praktische_categorie`` toch nog gekoppeld worden aan v0.16.4.
    """
    strict = _action_match_key(row, include_practical_category=True)
    relaxed = _action_match_key(row, include_practical_category=False)
    return [strict] if strict == relaxed else [strict, relaxed]


def merge_previous_action_follow_up(
    action_list: pd.DataFrame,
    previous_action_list: pd.DataFrame | None,
) -> tuple[pd.DataFrame, int]:
    """
    Neem eerdere beoordeling/afhandeling over in een nieuwe actielijst.

    De technische controles blijven leidend. We kopiëren alleen de
    databeheerdervelden wanneer dezelfde actieregel opnieuw gevonden wordt.
    Nieuwe of gewijzigde meldingen blijven dus gewoon op ``afhandelstatus =
    nieuw`` staan.
    """
    if action_list is None:
        return pd.DataFrame(), 0

    result = action_list.copy()
    for column in ACTION_FOLLOW_UP_COLUMNS:
        if column not in result.columns:
            result[column] = ""

    if previous_action_list is None or previous_action_list.empty or result.empty:
        return result, 0

    previous_lookup: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for _, previous_row in previous_action_list.iterrows():
        key = _action_match_key(previous_row)
        if not any(key):
            continue

        follow_up_values: dict[str, str] = {}
        for column in ACTION_FOLLOW_UP_COLUMNS:
            if column in previous_action_list.columns:
                value = clean_display_value(previous_row.get(column, ""))
                if value:
                    follow_up_values[column] = value

        if follow_up_values:
            for candidate_key in _action_match_candidate_keys(previous_row):
                previous_lookup.setdefault(candidate_key, follow_up_values)

    copied = 0
    for index, row in result.iterrows():
        values = None
        for candidate_key in _action_match_candidate_keys(row):
            values = previous_lookup.get(candidate_key)
            if values:
                break
        if not values:
            continue

        copied += 1
        for column, value in values.items():
            result.at[index, column] = value

    return result, copied



def apply_action_progress_tracking(
    action_list: pd.DataFrame,
    previous_action_list: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Markeer voortgang ten opzichte van een vorige onderhoudscontrole-actielijst.

    Dit is bedoeld voor periodiek beheer. Een controlepunt kan:
    - nieuw zijn: staat nu wel in de actielijst, maar zat niet in de vorige;
    - bestaand zijn: stond ook in de vorige actielijst;
    - opgelost/niet meer gevonden zijn: stond in de vorige actielijst, maar komt
      nu niet meer terug.

    De functie voert geen correcties uit. Hij geeft alleen voortgangsinformatie
    terug voor dashboard, export en overleg.
    """
    current = ensure_action_work_queue_columns(action_list)
    if current.empty:
        current = ensure_action_work_queue_columns(action_list)

    for column in ACTION_PROGRESS_COLUMNS:
        if column not in current.columns:
            current[column] = ""

    previous = ensure_action_work_queue_columns(previous_action_list)
    if previous_action_list is None or previous.empty:
        if not current.empty:
            current["voortgang_status"] = "nieuw_controlepunt"
            current["voortgang_uitleg"] = (
                "Geen vorige actielijst gebruikt; dit controlepunt staat als nieuw in deze controle."
            )
        counts = {
            "controlepunten_nieuw": int(len(current)),
            "controlepunten_bestaand": 0,
            "controlepunten_opgelost": 0,
        }
        return current, pd.DataFrame(columns=list(current.columns)), counts

    previous_keys: dict[tuple[str, str, str, str], pd.Series] = {}
    for _, previous_row in previous.iterrows():
        for candidate_key in _action_match_candidate_keys(previous_row):
            if any(candidate_key):
                previous_keys.setdefault(candidate_key, previous_row)

    current_candidate_keys: set[tuple[str, str, str, str]] = set()
    new_count = 0
    existing_count = 0

    for index, row in current.iterrows():
        matched = False
        for candidate_key in _action_match_candidate_keys(row):
            if any(candidate_key):
                current_candidate_keys.add(candidate_key)
            if candidate_key in previous_keys:
                matched = True

        if matched:
            existing_count += 1
            current.at[index, "voortgang_status"] = "bestaand_controlepunt"
            current.at[index, "voortgang_uitleg"] = (
                "Dit controlepunt stond ook in de vorige actielijst. Eerdere beoordeling en afhandelstatus zijn waar mogelijk overgenomen."
            )
        else:
            new_count += 1
            current.at[index, "voortgang_status"] = "nieuw_controlepunt"
            current.at[index, "voortgang_uitleg"] = (
                "Dit controlepunt stond niet in de vorige actielijst en vraagt om nieuwe beoordeling."
            )

    resolved_records: list[dict[str, Any]] = []
    seen_resolved: set[tuple[str, str, str, str]] = set()
    for _, previous_row in previous.iterrows():
        candidate_keys = [key for key in _action_match_candidate_keys(previous_row) if any(key)]
        if not candidate_keys:
            continue

        if any(key in current_candidate_keys for key in candidate_keys):
            continue

        strict_key = candidate_keys[0]
        if strict_key in seen_resolved:
            continue
        seen_resolved.add(strict_key)

        record = previous_row.to_dict()
        record["voortgang_status"] = "opgelost_of_niet_meer_gevonden"
        record["voortgang_uitleg"] = (
            "Dit controlepunt stond in de vorige actielijst, maar komt niet terug in de nieuwe controle. "
            "Controleer of het daadwerkelijk is opgelost of dat het buiten de nieuwe exportselectie valt."
        )
        resolved_records.append(record)

    resolved = ensure_action_work_queue_columns(pd.DataFrame(resolved_records)) if resolved_records else pd.DataFrame(columns=list(current.columns))
    counts = {
        "controlepunten_nieuw": int(new_count),
        "controlepunten_bestaand": int(existing_count),
        "controlepunten_opgelost": int(len(resolved)),
    }
    return current, resolved, counts


def ensure_action_work_queue_columns(action_list: pd.DataFrame | None) -> pd.DataFrame:
    """
    Maak een actielijst geschikt voor de werkvoorraadweergave.

    Waarom deze stap?
    Oude of handmatig aangepaste actielijsten kunnen opvolgkolommen missen of
    net anders noemen. De app moet dan nog steeds een stabiele tabel kunnen
    tonen en exporteren.
    """
    if action_list is None or action_list.empty:
        return pd.DataFrame(columns=list(ACTION_WORK_QUEUE_DISPLAY_COLUMNS))

    result = action_list.copy()

    for column in ACTION_FOLLOW_UP_COLUMNS:
        if column not in result.columns:
            result[column] = ""

    if "afhandelstatus" in result.columns:
        result["afhandelstatus"] = result["afhandelstatus"].apply(
            lambda value: clean_display_value(value) or "nieuw"
        )

    ordered_columns = [column for column in ACTION_WORK_QUEUE_DISPLAY_COLUMNS if column in result.columns]
    remaining_columns = [column for column in result.columns if column not in ordered_columns]
    return result[ordered_columns + remaining_columns]


def action_work_queue_summary(action_list: pd.DataFrame | None) -> dict[str, int]:
    """
    Tel de werkvoorraad voor het onderhoudscontrole-dashboard.

    De inhoudelijke onderhoudscontrole blijft in `build_maintenance_control`.
    Deze samenvatting is puur bedoeld voor de gebruikersinterface: hoeveel werk
    staat er open en waar zit het zwaartepunt?
    """
    result = {
        "controlepunten": 0,
        "waarschuwingen": 0,
        "aandachtspunten": 0,
        "nieuw": 0,
        "in_onderzoek": 0,
        "te_corrigeren": 0,
        "verklaarbare_uitzondering": 0,
        "afgehandeld": 0,
        "open": 0,
    }

    if action_list is None or action_list.empty:
        return result

    queue = ensure_action_work_queue_columns(action_list)
    result["controlepunten"] = int(len(queue))

    if "ernst" in queue.columns:
        severity = queue["ernst"].fillna("").astype(str).str.strip().str.lower()
        result["waarschuwingen"] = int((severity == "waarschuwing").sum())
        result["aandachtspunten"] = int((severity == "aandachtspunt").sum())

    status = queue.get("afhandelstatus", pd.Series(dtype=str)).fillna("").astype(str).str.strip().str.lower()
    result["nieuw"] = int((status == "nieuw").sum())
    result["in_onderzoek"] = int((status == "in onderzoek").sum())
    result["te_corrigeren"] = int(status.str.contains("te corrigeren", regex=False).sum())
    result["verklaarbare_uitzondering"] = int((status == "verklaarbare uitzondering").sum())
    result["afgehandeld"] = int((status == "afgehandeld").sum())
    result["open"] = int(result["controlepunten"] - result["afgehandeld"])

    return result


def filter_action_work_queue(
    action_list: pd.DataFrame | None,
    *,
    ernst: str | None = None,
    status: str | None = None,
    praktische_categorie: str | None = None,
    duiding: str | None = None,
    duiding_groep: str | None = None,
    voortgang_status: str | None = None,
    afhandelstatus: str | None = None,
    actiehouder: str | None = None,
    zoektekst: str | None = None,
    wegnummer: str | None = None,
) -> pd.DataFrame:
    """
    Filter de onderhoudscontrole-werkvoorraad op veelgebruikte gebruikersfilters.

    De functie is bewust pandas-only, zodat dezelfde filterlogica in Streamlit
    én in tests gebruikt kan worden.
    """
    queue = ensure_action_work_queue_columns(action_list)
    if queue.empty:
        return queue

    result = queue.copy()

    filter_map = {
        "wegnummer": wegnummer,
        "ernst": ernst,
        "status": status,
        "praktische_categorie": praktische_categorie,
        "duiding": duiding,
        "duiding_groep": duiding_groep,
        "voortgang_status": voortgang_status,
        "afhandelstatus": afhandelstatus,
        "actiehouder": actiehouder,
    }

    for column, value in filter_map.items():
        cleaned = clean_display_value(value).strip()
        if not cleaned or cleaned.lower().startswith("alle "):
            continue
        if column not in result.columns:
            continue

        result = result[result[column].fillna("").astype(str) == cleaned]

    search = clean_display_value(zoektekst).strip().lower()
    if search:
        search_columns = [
            column
            for column in (
                "wegnummer",
                "onderhoudsproject",
                "status",
                "praktische_categorie",
                "duiding",
                "duiding_groep",
                "duiding_uitleg",
                "voortgang_status",
                "voortgang_uitleg",
                "betrokken_objecten",
                "mogelijke_onderhoudsmatch",
                "onderhoudsmatch_uitleg",
                "uitleg",
                "mogelijke_oorzaak",
                "voorgestelde_actie",
                "beoordeling_databeheerder",
                "actiehouder",
                "opmerking_afhandeling",
            )
            if column in result.columns
        ]

        if search_columns:
            haystack = result[search_columns].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
            result = result[haystack.str.contains(re.escape(search), regex=True)]

    return result.reset_index(drop=True)


def merge_action_work_queue_edits(
    action_list: pd.DataFrame | None,
    edited_action_list: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Verwerk bewerkingen uit een gefilterde werkvoorraadtabel terug in de hele actielijst.

    In de app bewerkt de gebruiker vaak een gefilterde selectie. Deze functie
    kopieert alleen de opvolgvelden terug naar de volledige actielijst en laat
    technische controlekolommen ongemoeid.
    """
    base = ensure_action_work_queue_columns(action_list)
    if base.empty or edited_action_list is None or edited_action_list.empty:
        return base

    edited = ensure_action_work_queue_columns(edited_action_list)

    edited_lookup: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for _, edited_row in edited.iterrows():
        values = {
            column: clean_display_value(edited_row.get(column, ""))
            for column in ACTION_FOLLOW_UP_COLUMNS
            if column in edited.columns
        }
        for candidate_key in _action_match_candidate_keys(edited_row):
            if any(candidate_key):
                edited_lookup[candidate_key] = values

    result = base.copy()
    for index, row in result.iterrows():
        values = None
        for candidate_key in _action_match_candidate_keys(row):
            values = edited_lookup.get(candidate_key)
            if values is not None:
                break

        if values is None:
            continue

        for column, value in values.items():
            result.at[index, column] = value or ("nieuw" if column == "afhandelstatus" else "")

    return result


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


def _duiding_for_action(status: str, row: dict[str, Any], involved_objects: list[str]) -> str:
    """
    Geef een menselijke duiding naast de technische status.

    Deze kolom is geen besluit en geen automatische classificatie voor mutaties.
    Hij helpt de databeheerder om controlepunten sneller te ordenen in:
    waarschijnlijke fouten, twijfelgevallen/grensgevallen en export- of
    naamkwesties. De uiteindelijke beoordeling blijft mensenwerk.
    """
    if status == "HM_BEREIK_VERDACHT":
        return "waarschijnlijke_fout_in_paspoort"
    if status == "OBJECT_WEGNUMMER_VERDACHT":
        return "fout_of_grensgeval_controleren"
    if status == "ONTBREEKT_IN_ONDERHOUD":
        match_text = clean_display_value(row.get("mogelijke_onderhoudsmatch", ""))
        if match_text:
            return "mogelijke_oude_projectnaam"
        return "ontbrekend_project_of_exportfilter_controleren"
    if status == "OBJECTVERSCHIL":
        return "koppeling_of_exportmoment_controleren"
    if status == "GEEN_PASPOORTOBJECTEN":
        return "mogelijk_verweesd_project_of_exportselectie"
    return "handmatig_beoordelen"


def _duiding_for_suggestion_type(voorsteltype: str) -> str:
    """Duid veilige mutatievoorstellen zonder de gebruiker naar één oplossing te duwen."""
    if voorsteltype == "METRERING_PASPOORT_CORRIGEREN":
        return "waarschijnlijke_fout_in_paspoort"
    if voorsteltype == "WEGNUMMER_OBJECT_CONTROLEREN":
        return "fout_of_grensgeval_controleren"
    if voorsteltype in {
        "PROJECTNAAM_PASPOORT_CONTROLEREN",
        "ONDERHOUDSPROJECT_AANMAKEN_OF_EXPORTFILTER_CONTROLEREN",
        "ONDERHOUDSPROJECT_ZONDER_PASPOORTOBJECTEN_CONTROLEREN",
    }:
        return "projectnaam_of_exportfilter_controleren"
    if voorsteltype in {
        "OBJECTKOPPELING_ONDERHOUD_CONTROLEREN",
        "OBJECTKOPPELING_PASPOORT_OF_ONDERHOUD_CONTROLEREN",
    }:
        return "objectkoppeling_controleren"
    return "handmatig_beoordelen"


def _duiding_metadata(duiding: Any) -> tuple[str, str]:
    """
    Vertaal de technische duidingscode naar een bredere groep en een korte uitleg.

    De codekolom blijft bewust bestaan voor stabiele exports. De extra groep en
    uitleg zijn bedoeld voor normaal dagelijks gebruik: snel zien of iets
    waarschijnlijk een fout, een twijfelgeval of vooral een export-/naamkwestie is.
    """
    value = clean_display_value(duiding).strip()

    mapping: dict[str, tuple[str, str]] = {
        "waarschijnlijke_fout_in_paspoort": (
            "waarschijnlijke_fout",
            "Waarschijnlijk staat er een fout in het objectpaspoort. Controleer en corrigeer alleen na inhoudelijke bevestiging.",
        ),
        "fout_of_grensgeval_controleren": (
            "twijfel_of_grensgeval",
            "Het objectnummer of de koppeling lijkt verdacht, maar kan ook een grensgeval of administratieve uitzondering zijn.",
        ),
        "mogelijke_oude_projectnaam": (
            "oude_projectnaam_of_migratie",
            "Het project ontbreekt in de onderhoudsexport, maar lijkt mogelijk op een bestaand onderhoudsproject.",
        ),
        "ontbrekend_project_of_exportfilter_controleren": (
            "export_of_naamkwestie",
            "Het project ontbreekt in de onderhoudsexport. Controleer of het project bestaat en of de exportselectie volledig is.",
        ),
        "koppeling_of_exportmoment_controleren": (
            "objectkoppeling_controleren",
            "De objectsets verschillen. Controleer of dit een koppelfout is of een verschil tussen exportmomenten.",
        ),
        "mogelijk_verweesd_project_of_exportselectie": (
            "verweesd_project_of_exportselectie",
            "Het project staat in de onderhoudsexport, maar heeft geen bijpassende paspoortobjecten in deze selectie.",
        ),
        "projectnaam_of_exportfilter_controleren": (
            "export_of_naamkwestie",
            "Controleer projectnaam, onderhoudsproject en exportfilter voordat je iets wijzigt.",
        ),
        "objectkoppeling_controleren": (
            "objectkoppeling_controleren",
            "Controleer of het object aan het juiste onderhoudsproject en de juiste objectcategorie hangt.",
        ),
        "handmatig_beoordelen": (
            "handmatig_beoordelen",
            "De melding vraagt om inhoudelijke beoordeling door de databeheerder.",
        ),
    }

    return mapping.get(value, ("handmatig_beoordelen", "Controleer deze melding handmatig; de tool doet geen automatische aanpassing."))


def _action_text_for_status(status: str, row: dict[str, Any], involved_objects: list[str]) -> tuple[str, str, str, str]:
    """Vertaal een technische onderhoudscontrole-status naar begrijpelijke controle-instructies."""
    project_name = clean_display_value(row.get("onderhoudsproject", "")) or "dit onderhoudsproject"
    paspoort_count = _safe_int_value(row.get("paspoort_unieke_objecten", 0))
    onderhoud_count = _safe_int_value(row.get("onderhoud_unieke_objecten", 0))
    only_passport = _safe_int_value(row.get("alleen_in_paspoort", 0))
    only_maintenance = _safe_int_value(row.get("alleen_in_onderhoud", 0))
    wrong_road = _safe_int_value(row.get("onderhoud_object_wegnummer_verdacht", 0))
    invalid_hm = _safe_int_value(row.get("ongeldige_metrering_paspoort", 0))

    if status == "ONTBREEKT_IN_ONDERHOUD":
        category = "Project ontbreekt in onderhoudsexport"
        match_text = clean_display_value(row.get("mogelijke_onderhoudsmatch", ""))
        match_hint = f" Mogelijke onderhoudsmatch: {match_text}." if match_text else ""
        explanation = (
            f"{project_name} staat bij {paspoort_count} paspoortobject(en), "
            "maar komt niet voor in de onderhoudsexport."
            f"{match_hint}"
        )
        cause = (
            "Het onderhoudsproject is mogelijk nog niet aangemaakt, niet mee-geëxporteerd, "
            "of de projectnaam wijkt in iASSET net anders af. Als er een mogelijke match is, "
            "kan het ook gaan om een oude projectnaam die nog bij objecten staat."
        )
        action = (
            "Zoek het onderhoudsproject exact op in iASSET Onderhoud. Controleer daarna of "
            "de projectnaam gelijk gespeld is en of het project in de onderhoudsexportfilter zit. "
            "Bekijk eventuele mogelijke onderhoudsmatch als hint, niet als automatische correctie."
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
            "de onderhoudscontrole opnieuw om te zien of het hm-bereik weer betrouwbaar is."
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
    Maak een werkbare actielijst uit de onderhoudscontrole.

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
        "duiding",
        "duiding_groep",
        "duiding_uitleg",
        "aantal_objecten",
        "betrokken_objecten",
        "mogelijke_onderhoudsmatch",
        "onderhoudsmatch_type",
        "onderhoudsmatch_score",
        "onderhoudsmatch_uitleg",
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
        duiding = _duiding_for_action(status, row, involved_objects)
        duiding_groep, duiding_uitleg = _duiding_metadata(duiding)

        records.append(
            {
                "wegnummer": clean_display_value(row.get("wegnummer", "")),
                "onderhoudsproject": clean_display_value(row.get("onderhoudsproject", "")),
                "status": status,
                "ernst": clean_display_value(row.get("ernst", "")),
                "controlecategorie": category,
                "praktische_categorie": practical_category,
                "duiding": duiding,
                "duiding_groep": duiding_groep,
                "duiding_uitleg": duiding_uitleg,
                "aantal_objecten": len(involved_objects),
                "betrokken_objecten": _preview_objects_for_action_list(involved_objects),
                "mogelijke_onderhoudsmatch": clean_display_value(row.get("mogelijke_onderhoudsmatch", "")),
                "onderhoudsmatch_type": clean_display_value(row.get("onderhoudsmatch_type", "")),
                "onderhoudsmatch_score": _safe_int_value(row.get("onderhoudsmatch_score", 0)),
                "onderhoudsmatch_uitleg": clean_display_value(row.get("onderhoudsmatch_uitleg", "")),
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



def _empty_mutation_suggestions() -> pd.DataFrame:
    """Geef een lege mutatievoorsteltabel met vaste kolommen terug."""
    return pd.DataFrame(columns=list(MUTATION_SUGGESTION_COLUMNS))


def _append_mutation_suggestion(
    records: list[dict[str, Any]],
    *,
    onderhoudsproject: Any,
    project_norm: Any,
    voorsteltype: str,
    ernst: str,
    bron_export: str,
    objectnummer: Any = "",
    veld: str = "",
    huidige_waarde: Any = "",
    voorgestelde_waarde: Any = "",
    zekerheid: str = "controle_nodig",
    toelichting: str = "",
    voorgestelde_controle: str = "",
    alleen_na_controle: bool = True,
    voorstelstatus: str = "concept_voorstel",
    duiding: str | None = None,
) -> None:
    """
    Voeg één veilige voorstelregel toe.

    Dit is nadrukkelijk géén automatische mutatie. De tabel helpt de
    databeheerder om de juiste correctie in iASSET of de exportselectie te
    bepalen. Iedere regel krijgt daarom vaste veiligheidskolommen die duidelijk
    maken dat verwerking altijd handmatig en na menselijke controle gebeurt.
    """
    duiding_value = clean_display_value(duiding) or _duiding_for_suggestion_type(voorsteltype)
    duiding_groep, duiding_uitleg = _duiding_metadata(duiding_value)

    records.append(
        {
            "wegnummer": _road_from_project_row({}, onderhoudsproject),
            "onderhoudsproject": clean_display_value(onderhoudsproject),
            "voorstelstatus": clean_display_value(voorstelstatus) or "concept_voorstel",
            "voorsteltype": voorsteltype,
            "ernst": ernst,
            "duiding": duiding_value,
            "duiding_groep": duiding_groep,
            "duiding_uitleg": duiding_uitleg,
            "bron_export": bron_export,
            "objectnummer": clean_display_value(objectnummer),
            "veld": veld,
            "huidige_waarde": clean_display_value(huidige_waarde),
            "voorgestelde_waarde": clean_display_value(voorgestelde_waarde),
            "zekerheid": zekerheid,
            "toelichting": toelichting,
            "voorgestelde_controle": voorgestelde_controle,
            "alleen_na_controle": True,
            "menselijke_controle_verplicht": True,
            "automatisch_doorvoeren": "nee",
            "veiligheidsmelding": (
                "Niet automatisch doorvoeren; controleer de voorstelregel en verwerk een eventuele "
                "wijziging handmatig in iASSET of de exportselectie."
            ),
            "project_norm": clean_display_value(project_norm),
        }
    )


def build_mutation_suggestions(
    comparison: pd.DataFrame,
    object_differences: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Vertaal onderhoudscontrolepunten naar veilige mutatievoorstellen.

    De voorstellen zijn bedoeld als werklijst voor iASSET-correcties. De tool
    schrijft niets terug naar iASSET en zet geen waarden automatisch om. Een
    voorstelregel zegt alleen: "controleer dit veld en overweeg deze actie".

    Veiligheidsregel:
    alle voorstelregels blijven conceptvoorstellen met menselijke controle
    verplicht. Deze functie bevat geen codepad dat wijzigingen doorvoert.
    """
    if comparison is None or comparison.empty:
        return _empty_mutation_suggestions()

    records: list[dict[str, Any]] = []

    diffs = object_differences if object_differences is not None else pd.DataFrame()
    if diffs is None or diffs.empty:
        diffs = pd.DataFrame(columns=[
            "onderhoudsproject",
            "project_norm",
            "objectnummer",
            "verschiltype",
            "bron",
            "metrering",
            "melding",
        ])

    for _, row_series in comparison.iterrows():
        row = row_series.to_dict()
        status = clean_display_value(row.get("status", ""))
        if status in {"", "OK", "OK_VOLLEDIG"}:
            continue

        project_name = clean_display_value(row.get("onderhoudsproject", ""))
        project_key = clean_display_value(row.get("project_norm", ""))
        project_diffs = diffs[diffs.get("project_norm", pd.Series(dtype=str)).astype(str) == project_key]

        if status == "ONTBREEKT_IN_ONDERHOUD":
            suggested_project = _first_project_from_match_text(row.get("mogelijke_onderhoudsmatch", ""))
            if suggested_project:
                _append_mutation_suggestion(
                    records,
                    onderhoudsproject=project_name,
                    project_norm=project_key,
                    voorsteltype="PROJECTNAAM_PASPOORT_CONTROLEREN",
                    ernst="waarschuwing",
                    bron_export="paspoortexport",
                    veld="Onderhoudsproject",
                    huidige_waarde=project_name,
                    voorgestelde_waarde=suggested_project,
                    zekerheid="match_hint",
                    toelichting=(
                        "Dit project staat bij paspoortobjecten, maar ontbreekt in de onderhoudsexport. "
                        "Er is wel een mogelijke bestaande onderhoudsmatch gevonden."
                    ),
                    voorgestelde_controle=(
                        "Controleer of de paspoortobjecten aan de voorgestelde onderhoudsmatch moeten hangen. "
                        "Pas de projectnaam alleen aan als dit inhoudelijk klopt."
                    ),
                )
            else:
                _append_mutation_suggestion(
                    records,
                    onderhoudsproject=project_name,
                    project_norm=project_key,
                    voorsteltype="ONDERHOUDSPROJECT_AANMAKEN_OF_EXPORTFILTER_CONTROLEREN",
                    ernst="waarschuwing",
                    bron_export="onderhoudsexport",
                    veld="Onderhoudsproject",
                    huidige_waarde="",
                    voorgestelde_waarde=project_name,
                    zekerheid="controle_nodig",
                    toelichting=(
                        "Dit project staat bij paspoortobjecten, maar ontbreekt in de onderhoudsexport."
                    ),
                    voorgestelde_controle=(
                        "Zoek het project exact op in iASSET Onderhoud. Controleer of het project moet worden "
                        "aangemaakt, hernoemd, of dat de onderhoudsexport/selectie onvolledig was."
                    ),
                )

        elif status == "GEEN_PASPOORTOBJECTEN":
            _append_mutation_suggestion(
                records,
                onderhoudsproject=project_name,
                project_norm=project_key,
                voorsteltype="ONDERHOUDSPROJECT_ZONDER_PASPOORTOBJECTEN_CONTROLEREN",
                ernst="waarschuwing",
                bron_export="onderhoudsexport",
                veld="Onderhoudsproject",
                huidige_waarde=project_name,
                voorgestelde_waarde="",
                zekerheid="controle_nodig",
                toelichting=(
                    "Dit project staat in de onderhoudsexport, maar heeft geen objecten in de paspoortexport."
                ),
                voorgestelde_controle=(
                    "Controleer of het onderhoudsproject verouderd is, of dat de paspoortexport niet dezelfde "
                    "objectselectie bevat. Verwijder of herstel het project alleen na inhoudelijke controle."
                ),
            )

        if not project_diffs.empty:
            for _, diff_series in project_diffs.iterrows():
                diff = diff_series.to_dict()
                diff_type = clean_display_value(diff.get("verschiltype", ""))
                objectnummer = clean_display_value(diff.get("objectnummer", ""))

                if diff_type == "ALLEEN_IN_PASPOORT":
                    _append_mutation_suggestion(
                        records,
                        onderhoudsproject=project_name,
                        project_norm=project_key,
                        voorsteltype="OBJECTKOPPELING_ONDERHOUD_CONTROLEREN",
                        ernst="waarschuwing",
                        bron_export="paspoortexport",
                        objectnummer=objectnummer,
                        veld="Onderhoudsproject",
                        huidige_waarde=project_name,
                        voorgestelde_waarde=clean_display_value(row.get("mogelijke_onderhoudsmatch", "")),
                        zekerheid="controle_nodig",
                        toelichting=(
                            "Dit object hangt in de paspoortexport aan dit project, maar komt niet voor in de "
                            "onderhoudsexport voor hetzelfde project."
                        ),
                        voorgestelde_controle=(
                            "Controleer of het object aan het juiste onderhoudsproject hangt en of het project "
                            "in de onderhoudsexport hoort terug te komen."
                        ),
                    )
                elif diff_type == "ALLEEN_IN_ONDERHOUD":
                    _append_mutation_suggestion(
                        records,
                        onderhoudsproject=project_name,
                        project_norm=project_key,
                        voorsteltype="OBJECTKOPPELING_PASPOORT_OF_ONDERHOUD_CONTROLEREN",
                        ernst="waarschuwing",
                        bron_export="onderhoudsexport",
                        objectnummer=objectnummer,
                        veld="Onderhoudsproject",
                        huidige_waarde="",
                        voorgestelde_waarde=project_name,
                        zekerheid="controle_nodig",
                        toelichting=(
                            "Dit object staat in de onderhoudsexport bij dit project, maar niet in de "
                            "paspoortexport voor hetzelfde project."
                        ),
                        voorgestelde_controle=(
                            "Zoek het object in iASSET. Controleer of het paspoortproject moet worden aangepast, "
                            "of dat de onderhoudsregel/exportselectie niet klopt."
                        ),
                    )
                elif diff_type == "OBJECT_WEGNUMMER_VERDACHT":
                    road_guess = clean_display_value(diff.get("object_wegnummer_vermoed", ""))
                    selected_road = clean_display_value(diff.get("geselecteerde_weg", ""))
                    _append_mutation_suggestion(
                        records,
                        onderhoudsproject=project_name,
                        project_norm=project_key,
                        voorsteltype="WEGNUMMER_OBJECT_CONTROLEREN",
                        ernst="waarschuwing",
                        bron_export=clean_display_value(diff.get("bron", "onderhoudsexport")),
                        objectnummer=objectnummer,
                        veld="Wegnummer",
                        huidige_waarde=road_guess,
                        voorgestelde_waarde=selected_road,
                        zekerheid="controle_nodig",
                        toelichting=(
                            f"Het objectnummer lijkt bij {road_guess or 'een andere weg'} te horen, "
                            f"terwijl de controle voor {selected_road or 'de geselecteerde weg'} draait."
                        ),
                        voorgestelde_controle=(
                            "Controleer of dit een fout wegnummer, een grensgeval of een exportselectieprobleem is. "
                            "Pas het objectpaspoort of de onderhoudskoppeling alleen aan na controle op kaart/iASSET."
                        ),
                    )
                elif diff_type == "ONGELDIGE_METRERING_PASPOORT":
                    _append_mutation_suggestion(
                        records,
                        onderhoudsproject=project_name,
                        project_norm=project_key,
                        voorsteltype="METRERING_PASPOORT_CORRIGEREN",
                        ernst="aandachtspunt",
                        bron_export="paspoortexport",
                        objectnummer=objectnummer,
                        veld="Metrering",
                        huidige_waarde=diff.get("metrering", ""),
                        voorgestelde_waarde="",
                        zekerheid="handmatig_bepalen",
                        toelichting=(
                            "De metrering is ongeldig en is daarom niet meegenomen in hm_min/hm_max."
                        ),
                        voorgestelde_controle=(
                            "Controleer de juiste metrering in iASSET/StreetSmart en vul een geldige waarde in."
                        ),
                    )

    if not records:
        return _empty_mutation_suggestions()

    result = pd.DataFrame(records)
    for column in MUTATION_SUGGESTION_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    sort_order = {"waarschuwing": 0, "aandachtspunt": 1, "info": 2, "ok": 3}
    result["_sort_ernst"] = result["ernst"].map(sort_order).fillna(9)
    result = (
        result[list(MUTATION_SUGGESTION_COLUMNS) + ["_sort_ernst"]]
        .sort_values(["_sort_ernst", "onderhoudsproject", "voorsteltype", "objectnummer"])
        .drop(columns=["_sort_ernst"])
        .reset_index(drop=True)
    )
    return result


def build_maintenance_control(
    passport_df: pd.DataFrame,
    maintenance_df: pd.DataFrame,
    selected_road: str | None = None,
    previous_action_list: pd.DataFrame | None = None,
) -> MaintenanceControlResult:
    """
    Bouw de volledige onderhoudscontrole.

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
    action_list, copied_follow_up = merge_previous_action_follow_up(action_list, previous_action_list)
    action_list, resolved_actions, progress_counts = apply_action_progress_tracking(action_list, previous_action_list)
    mutation_suggestions = build_mutation_suggestions(comparison, object_differences)

    if copied_follow_up:
        warnings.append(
            f"{copied_follow_up} eerdere beoordeling(en) overgenomen in de Onderhoudscontrole-actielijst."
        )

    if comparison.empty:
        summary = {
            "wegen_gecontroleerd": 0,
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
            "acties_met_overgenomen_beoordeling": 0,
            "acties_met_mogelijke_projectmatch": 0,
            "mutatievoorstellen": 0,
            "acties_waarschijnlijke_fout": 0,
            "acties_twijfel_of_grensgeval": 0,
            "acties_export_of_naamkwestie": 0,
            "acties_objectkoppeling_controleren": 0,
            "acties_verweesd_project_of_exportselectie": 0,
            "controlepunten_nieuw": 0,
            "controlepunten_bestaand": 0,
            "controlepunten_opgelost": 0,
        }
    else:
        summary = {
            "wegen_gecontroleerd": int(comparison.get("wegnummer", pd.Series(dtype=str)).fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique()),
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
            "acties_met_overgenomen_beoordeling": int(copied_follow_up),
            "acties_met_mogelijke_projectmatch": int(
                action_list.get("mogelijke_onderhoudsmatch", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("").sum()
            ),
            "mutatievoorstellen": int(len(mutation_suggestions)),
            "acties_waarschijnlijke_fout": int(
                (action_list.get("duiding_groep", pd.Series(dtype=str)).fillna("").astype(str) == "waarschijnlijke_fout").sum()
            ),
            "acties_twijfel_of_grensgeval": int(
                (action_list.get("duiding_groep", pd.Series(dtype=str)).fillna("").astype(str) == "twijfel_of_grensgeval").sum()
            ),
            "acties_export_of_naamkwestie": int(
                action_list.get("duiding_groep", pd.Series(dtype=str)).fillna("").astype(str).isin(
                    ["export_of_naamkwestie", "oude_projectnaam_of_migratie"]
                ).sum()
            ),
            "acties_objectkoppeling_controleren": int(
                (action_list.get("duiding_groep", pd.Series(dtype=str)).fillna("").astype(str) == "objectkoppeling_controleren").sum()
            ),
            "acties_verweesd_project_of_exportselectie": int(
                (action_list.get("duiding_groep", pd.Series(dtype=str)).fillna("").astype(str) == "verweesd_project_of_exportselectie").sum()
            ),
            "controlepunten_nieuw": int(progress_counts.get("controlepunten_nieuw", 0)),
            "controlepunten_bestaand": int(progress_counts.get("controlepunten_bestaand", 0)),
            "controlepunten_opgelost": int(progress_counts.get("controlepunten_opgelost", 0)),
        }

    return MaintenanceControlResult(
        summary=summary,
        comparison=comparison,
        passport_projects=passport_projects,
        maintenance_projects=maintenance_projects,
        object_differences=object_differences,
        action_list=action_list,
        mutation_suggestions=mutation_suggestions,
        resolved_actions=resolved_actions,
        warnings=warnings,
    )


def _safe_excel_sheet_name(name: str, used: set[str] | None = None) -> str:
    """
    Maak een geldige, unieke Excel-tabbladnaam.

    Excel staat maximaal 31 tekens toe en verbiedt enkele speciale tekens. Deze
    helper houdt de export robuust, ook als we later dynamische tabbladnamen
    toevoegen.
    """
    used = used if used is not None else set()
    cleaned = re.sub(r"[\[\]\:\*\?\/\\]", " ", clean_display_value(name)).strip()
    cleaned = re.sub(r"\s+", " ", cleaned) or "Blad"
    cleaned = cleaned[:31]

    candidate = cleaned
    counter = 2
    while candidate in used:
        suffix = f" {counter}"
        candidate = f"{cleaned[:31 - len(suffix)]}{suffix}"
        counter += 1

    used.add(candidate)
    return candidate


def _count_table(df: pd.DataFrame | None, column: str, label_column: str, count_column: str) -> pd.DataFrame:
    """Maak een kleine draaitabel voor het Excel-controlepakket."""
    if df is None or df.empty or column not in df.columns:
        return pd.DataFrame(columns=[label_column, count_column])

    series = df[column].fillna("").astype(str).str.strip()
    series = series.replace("", "(leeg)")
    result = (
        series.value_counts(dropna=False)
        .rename_axis(label_column)
        .reset_index(name=count_column)
        .sort_values([count_column, label_column], ascending=[False, True])
        .reset_index(drop=True)
    )
    return result


def _summary_dataframe(summary: dict[str, Any], scope_label: str = "") -> pd.DataFrame:
    """Zet de belangrijkste onderhoudscontrole-metrics om naar een leesbare tabel."""
    rows = [
        ("Controlebereik", scope_label),
        ("Wegen gecontroleerd", summary.get("wegen_gecontroleerd", 0)),
        ("Projecten totaal", summary.get("projecten_totaal", 0)),
        ("OK volledig", summary.get("ok_volledig", summary.get("projecten_ok", 0))),
        ("Projectnaam in beide exports", summary.get("ok_projectnaam", 0)),
        ("Objectverschillen", summary.get("objectverschillen", 0)),
        ("Verdacht objectwegnummer", summary.get("object_wegnummer_verdacht", 0)),
        ("Ontbreekt in onderhoud", summary.get("ontbreekt_in_onderhoud", 0)),
        ("Onderhoud zonder paspoortobjecten", summary.get("geen_paspoortobjecten", 0)),
        ("Verdacht hm-bereik", summary.get("hm_bereik_verdacht", 0)),
        ("Waarschuwingen", summary.get("waarschuwingen", 0)),
        ("Aandachtspunten", summary.get("aandachtspunten", 0)),
        ("Controlepunten in werkvoorraad", summary.get("acties", 0)),
        ("Nieuwe controlepunten", summary.get("controlepunten_nieuw", 0)),
        ("Bestaande controlepunten", summary.get("controlepunten_bestaand", 0)),
        ("Opgelost/niet meer gevonden", summary.get("controlepunten_opgelost", 0)),
        ("Mutatievoorstellen", summary.get("mutatievoorstellen", 0)),
        ("Eerdere beoordelingen overgenomen", summary.get("acties_met_overgenomen_beoordeling", 0)),
    ]
    return pd.DataFrame(rows, columns=["Onderdeel", "Waarde"])


def _write_dataframe_sheet(writer: pd.ExcelWriter, df: pd.DataFrame | None, sheet_name: str, used: set[str]) -> str:
    """Schrijf een dataframe naar een Excel-tabblad met nette basisopmaak."""
    safe_name = _safe_excel_sheet_name(sheet_name, used)
    table = pd.DataFrame() if df is None else df.copy()
    table.to_excel(writer, sheet_name=safe_name, index=False)
    return safe_name


def build_maintenance_control_workbook(
    result: MaintenanceControlResult,
    *,
    action_list: pd.DataFrame | None = None,
    scope_label: str = "",
) -> bytes:
    """
    Bouw een Excel-controlepakket voor de onderhoudscontrole.

    Het pakket is bedoeld voor overleg en archivering. Het is bewust een
    exportbestand: deze functie voert geen mutaties uit en bevat geen logica die
    iASSET of het maatregeltoetsdocument wijzigt.
    """
    if result is None:
        result = MaintenanceControlResult()

    package_action_list = ensure_action_work_queue_columns(
        action_list if action_list is not None else result.action_list
    )
    comparison = pd.DataFrame() if result.comparison is None else result.comparison.copy()
    mutation_suggestions = (
        _empty_mutation_suggestions()
        if result.mutation_suggestions is None or result.mutation_suggestions.empty
        else result.mutation_suggestions.copy()
    )
    object_differences = pd.DataFrame() if result.object_differences is None else result.object_differences.copy()
    passport_projects = pd.DataFrame() if result.passport_projects is None else result.passport_projects.copy()
    maintenance_projects = pd.DataFrame() if result.maintenance_projects is None else result.maintenance_projects.copy()
    resolved_actions = ensure_action_work_queue_columns(result.resolved_actions)

    summary = result.summary or {}
    action_summary = action_work_queue_summary(package_action_list)

    summary_tables = {
        "Kerncijfers": _summary_dataframe(summary, scope_label=scope_label),
        "Werkvoorraad": pd.DataFrame(
            [
                ("Controlepunten", action_summary.get("controlepunten", 0)),
                ("Open", action_summary.get("open", 0)),
                ("Nieuw sinds vorige controle", summary.get("controlepunten_nieuw", 0)),
                ("Bestaand sinds vorige controle", summary.get("controlepunten_bestaand", 0)),
                ("Opgelost/niet meer gevonden", summary.get("controlepunten_opgelost", 0)),
                ("Afhandelstatus nieuw", action_summary.get("nieuw", 0)),
                ("In onderzoek", action_summary.get("in_onderzoek", 0)),
                ("Te corrigeren", action_summary.get("te_corrigeren", 0)),
                ("Verklaarbare uitzondering", action_summary.get("verklaarbare_uitzondering", 0)),
                ("Afgehandeld", action_summary.get("afgehandeld", 0)),
            ],
            columns=["Onderdeel", "Waarde"],
        ),
        "Per weg": _count_table(package_action_list, "wegnummer", "wegnummer", "controlepunten"),
        "Per duidingsgroep": _count_table(package_action_list, "duiding_groep", "duiding_groep", "controlepunten"),
        "Per afhandelstatus": _count_table(package_action_list, "afhandelstatus", "afhandelstatus", "controlepunten"),
        "Per voortgang": _count_table(package_action_list, "voortgang_status", "voortgang_status", "controlepunten"),
        "Per actiehouder": _count_table(package_action_list, "actiehouder", "actiehouder", "controlepunten"),
    }

    # Eén samenvattingsblad met blokken onder elkaar is voor overleg prettiger
    # dan losse kleine tabbladen.
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        used_sheet_names: set[str] = set()
        summary_sheet = _safe_excel_sheet_name("Samenvatting", used_sheet_names)
        start_row = 0
        for title, table in summary_tables.items():
            pd.DataFrame([[title]], columns=["Onderhoudscontrole"]).to_excel(
                writer,
                sheet_name=summary_sheet,
                startrow=start_row,
                index=False,
            )
            start_row += 2
            table.to_excel(writer, sheet_name=summary_sheet, startrow=start_row, index=False)
            start_row += max(len(table), 1) + 3

        _write_dataframe_sheet(writer, package_action_list, "Werkvoorraad", used_sheet_names)
        _write_dataframe_sheet(writer, resolved_actions, "Opgelost", used_sheet_names)
        _write_dataframe_sheet(writer, mutation_suggestions, "Mutatievoorstellen", used_sheet_names)
        _write_dataframe_sheet(writer, comparison, "Resultaten", used_sheet_names)
        _write_dataframe_sheet(writer, object_differences, "Objectverschillen", used_sheet_names)
        _write_dataframe_sheet(writer, passport_projects, "Paspoortprojecten", used_sheet_names)
        _write_dataframe_sheet(writer, maintenance_projects, "Onderhoudsexport", used_sheet_names)

        workbook = writer.book
        for worksheet in workbook.worksheets:
            worksheet.freeze_panes = "A2"
            for column_cells in worksheet.columns:
                max_length = 0
                column_letter = column_cells[0].column_letter
                for cell in column_cells[:200]:
                    value = "" if cell.value is None else str(cell.value)
                    max_length = max(max_length, min(len(value), 80))
                worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 12), 50)

            for row in worksheet.iter_rows():
                for cell in row:
                    alignment = copy.copy(cell.alignment)
                    alignment.wrap_text = True
                    alignment.vertical = "top"
                    cell.alignment = alignment

    return buffer.getvalue()

