"""
Objectinspecteur en gecontroleerde paspoortmutaties.

Deze module bevat geen Streamlit-code. De UI gebruikt deze hulpfuncties om
objecten te zoeken, labels te tonen en te bepalen welke velden veilig in een
mutatieformulier kunnen worden aangeboden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd

from .changes import collect_changed_ids, get_export_profile_columns
from .utils import clean_display_value, normalize_text, safe_int


# Velden die nooit via de objectinspecteur worden gewijzigd.
#
# Geometrie en bron-id's blijven bewust buiten het formulier. De tool bereidt
# paspoortmutaties voor; geometrische mutaties horen vanuit de bronprocessen /
# BGT/iASSET te komen.
NON_EDITABLE_COLUMNS = {
    "geometry",
    "sys_id",
    "bron_id",
    "id",
    "gps coordinaten",
    "rds coordinaten",
    "Advies_Bron",
    "Advies_Onderhoudsproject",
    "validation_error",
    "Spoor_ID",
    "Is_Project_Grens",
}


# Velden die volgens het werkproces belangrijk zijn voor individuele
# verhardingsobjecten. De actuele export kan per bestand verschillen; daarom
# worden alleen aanwezige kolommen in het formulier getoond.
IMPORTANT_PASSPORT_FIELDS = [
    "Thema",
    "subthema",
    "nummer",
    "naam",
    "Gebruikersfunctie",
    "Type onderdeel",
    "verhardingssoort",
    "Soort verharding_N",
    "Soort deklaag specifiek",
    "Soort conservering",
    "Jaar aanleg",
    "Jaar deklaag",
    "Jaar conservering",
    "Jaar herstrating",
    "Besteknummer",
    "Onderhoudsproject",
]


IMPORTANT_LOCATION_FIELDS = [
    "Wegnummer",
    "Wegvak",
    "Wegvak V",
    "Wegvak G",
    "Wegvaknum",
    "Wegvaknum V",
    "Wegvaknum G",
    "Metrering",
    "Metrering V",
    "Metrering G",
    "Situering",
    "Situering V",
    "Bebouwde_kom",
]


@dataclass(frozen=True)
class ObjectSearchResult:
    """Compact zoekresultaat voor de objectinspecteur."""

    object_id: int
    label: str


def _available_columns(gdf: gpd.GeoDataFrame, columns: Iterable[str]) -> list[str]:
    """Geef kolommen terug die bestaan en niet leeg/ongeldig zijn."""
    if gdf is None or gdf.empty:
        return []

    result: list[str] = []
    for column in columns:
        if column in gdf.columns and column not in result:
            result.append(column)
    return result


def editable_fields_for_profile(
    gdf: gpd.GeoDataFrame,
    profile_name: str | None = None,
    *,
    include_location_fallback: bool = False,
) -> list[str]:
    """
    Bepaal welke velden in de objectinspecteur bewerkbaar zijn.

    We starten vanuit het gekozen iASSET-exportprofiel, omdat iASSET per import
    één vaste kolommenset voor alle objecten verwerkt. Identificatie- en
    geometriekolommen worden bewust uitgefilterd.

    Als een profiel in een bronexport maar weinig bewerkbare kolommen oplevert,
    kan de UI met ``include_location_fallback`` ook de bekende belangrijke
    paspoort- en liggingvelden aanbieden die in de dataset aanwezig zijn.
    """
    if gdf is None:
        return []

    profile_columns = get_export_profile_columns(profile_name)
    ordered_columns = list(profile_columns)

    if include_location_fallback:
        ordered_columns.extend(IMPORTANT_PASSPORT_FIELDS)
        ordered_columns.extend(IMPORTANT_LOCATION_FIELDS)

    fields: list[str] = []
    for column in ordered_columns:
        if column in NON_EDITABLE_COLUMNS:
            continue
        if column not in gdf.columns:
            continue
        if column not in fields:
            fields.append(column)

    return fields


def missing_profile_columns(gdf: gpd.GeoDataFrame, profile_name: str | None = None) -> list[str]:
    """
    Geef profielkolommen terug die ontbreken in de actieve dataset.

    Dit is vooral nuttig in de UI: een exportprofiel kan inhoudelijk gewenst zijn,
    maar een iASSET-exportbestand bevat niet altijd alle kolommen.
    """
    if gdf is None:
        return []

    return [
        column
        for column in get_export_profile_columns(profile_name)
        if column not in gdf.columns
    ]


def object_label(row: pd.Series | dict[str, Any]) -> str:
    """
    Maak een leesbaar label voor één object.

    We combineren het interne sys_id met herkenbare iASSET-velden. Lege of
    corrupte waarden worden genegeerd, zodat het label ook bij incomplete exports
    bruikbaar blijft.
    """
    getter = row.get if hasattr(row, "get") else dict(row).get

    object_id = clean_display_value(getter("sys_id", ""))
    nummer = clean_display_value(getter("nummer", ""))
    subthema = clean_display_value(getter("subthema", ""))
    project = clean_display_value(getter("Onderhoudsproject", ""))
    metrering = clean_display_value(getter("Metrering", ""))
    wegvaknum = clean_display_value(getter("Wegvaknum", ""))

    parts: list[str] = []
    if nummer:
        parts.append(nummer)
    if subthema:
        parts.append(subthema)
    if wegvaknum or metrering:
        ligging = " / ".join(value for value in [wegvaknum, metrering] if value)
        parts.append(f"wv/hm {ligging}")
    if project:
        parts.append(project)

    prefix = f"ID {object_id}" if object_id else "ID ?"
    if not parts:
        return prefix

    return f"{prefix} · " + " · ".join(parts)


def _row_contains_query(row: pd.Series, query: str, search_columns: list[str]) -> bool:
    """Controleer of een rij de zoektekst bevat in één van de zoekkolommen."""
    q = normalize_text(query)
    if not q:
        return True

    for column in search_columns:
        if column not in row.index:
            continue
        value = normalize_text(clean_display_value(row.get(column, "")))
        if q in value:
            return True

    # Ook zoeken op intern sys_id.
    object_id = normalize_text(clean_display_value(row.get("sys_id", "")))
    return q in object_id


def search_objects(
    gdf: gpd.GeoDataFrame,
    query: str = "",
    *,
    changed_only: bool = False,
    change_log: Iterable[dict[str, Any]] | None = None,
    max_results: int = 200,
) -> list[ObjectSearchResult]:
    """
    Zoek objecten voor de objectinspecteur.

    De functie zoekt bewust in meerdere herkenbare paspoortvelden. Bij lege
    zoektekst worden de eerste objecten teruggegeven, zodat de gebruiker ook
    zonder exacte objectnaam kan bladeren.
    """
    if gdf is None or gdf.empty:
        return []

    working = gdf

    if changed_only:
        changed_ids = collect_changed_ids(change_log or [])
        valid_ids = [object_id for object_id in changed_ids if object_id in gdf.index]
        working = gdf.loc[valid_ids] if valid_ids else gdf.iloc[0:0]

    search_columns = [
        "sys_id",
        "bron_id",
        "nummer",
        "naam",
        "subthema",
        "Onderhoudsproject",
        "Wegnummer",
        "Wegvaknum",
        "Wegvaknum V",
        "Wegvaknum G",
        "Metrering",
        "Metrering V",
        "Metrering G",
        "Situering",
        "Situering V",
        "Besteknummer",
    ]

    results: list[ObjectSearchResult] = []
    for idx, row in working.iterrows():
        if not _row_contains_query(row, query, search_columns):
            continue

        object_id = safe_int(idx)
        if object_id is None:
            object_id = safe_int(row.get("sys_id"))

        if object_id is None:
            continue

        results.append(ObjectSearchResult(object_id=object_id, label=object_label(row)))

        if len(results) >= max_results:
            break

    return results


def object_preview_dataframe(gdf: gpd.GeoDataFrame, object_id: Any, fields: Iterable[str]) -> pd.DataFrame:
    """
    Maak een kleine tabel met veld/waarde voor de inspectieweergave.

    Ontbrekende velden worden overgeslagen. De tabel is vooral bedoeld om onder
    het mutatieformulier snel de huidige paspoortwaarden te controleren.
    """
    coerced_id = safe_int(object_id)
    if coerced_id is None or gdf is None or coerced_id not in gdf.index:
        return pd.DataFrame(columns=["Veld", "Waarde"])

    row = gdf.loc[coerced_id]
    rows = []
    for field in fields:
        if field not in gdf.columns:
            continue
        rows.append({"Veld": field, "Waarde": clean_display_value(row.get(field, ""))})

    return pd.DataFrame(rows)
