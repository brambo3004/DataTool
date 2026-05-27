"""
Overzichtskaart voor rijstroken.

Deze module bouwt een Folium-kaart voor het tabblad "Overzicht".
De kaart is bewust alleen-lezen: hij visualiseert rijstroken per gekozen
attribuut en past geen iASSET-data aan.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any

import folium
import geopandas as gpd
import pandas as pd

from .config import OVERVIEW_ATTRIBUTE_ALIASES, OVERVIEW_POPUP_COLUMNS
from .utils import clean_display_value, normalize_text


UNKNOWN_LABEL = "Onbekend"


def _column_has_display_values(gdf: gpd.GeoDataFrame, column: str) -> bool:
    """
    Controleer of een kolom minstens één inhoudelijke waarde heeft.

    Dit voorkomt dat een lege alias-kolom, die door de loader is aangemaakt,
    een echte bronkolom met data overschaduwt.
    """
    if column not in gdf.columns:
        return False

    return any(clean_display_value(value) for value in gdf[column])


@dataclass(frozen=True)
class LegendItem:
    """Eén regel in de legenda."""

    label: str
    color: str


@dataclass
class OverviewMapResult:
    """Resultaat van de overzichtskaart."""

    folium_map: folium.Map
    row_count: int
    legend_items: list[LegendItem]
    selected_column: str | None


def resolve_overview_attribute(gdf: gpd.GeoDataFrame, attribute_label: str) -> str | None:
    """
    Vertaal een gebruikerslabel naar de beste beschikbare kolom in de data.

    Waarom?
    De oude voorbeeldtool gebruikt bijvoorbeeld `Soort verharding_N`, terwijl
    onze huidige app vaak `verhardingssoort` gebruikt. Door aliases centraal te
    behandelen, blijft het tabblad bruikbaar bij licht afwijkende exports.

    We kiezen bij voorkeur een kolom met echte waarden. Dat is nodig omdat de
    loader ontbrekende verwachte kolommen soms als lege kolom aanmaakt.
    """
    candidates = OVERVIEW_ATTRIBUTE_ALIASES.get(attribute_label, [attribute_label])
    first_existing: str | None = None

    for column in candidates:
        if column not in gdf.columns:
            continue

        if first_existing is None:
            first_existing = column

        if _column_has_display_values(gdf, column):
            return column

    return first_existing


def available_overview_attributes(gdf: gpd.GeoDataFrame) -> list[str]:
    """
    Geef de attributen terug die daadwerkelijk gevisualiseerd kunnen worden.

    Een attribuut wordt alleen aangeboden als er op de rijstroken minstens één
    inhoudelijke waarde beschikbaar is.
    """
    rijstroken = _rijstroken_only(gdf)
    if rijstroken.empty:
        return []

    available: list[str] = []
    for label in OVERVIEW_ATTRIBUTE_ALIASES:
        column = resolve_overview_attribute(rijstroken, label)
        if column is not None and _column_has_display_values(rijstroken, column):
            available.append(label)

    return available


def _rijstroken_only(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Filter de data op rijstroken.

    We gebruiken `subthema_clean` als die bestaat; anders normaliseren we
    `subthema` zelf. Zo blijft de functie ook testbaar met kleine testframes.
    """
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=getattr(gdf, "crs", None))

    if "subthema_clean" in gdf.columns:
        mask = gdf["subthema_clean"].apply(normalize_text) == "rijstrook"
    elif "subthema" in gdf.columns:
        mask = gdf["subthema"].apply(normalize_text) == "rijstrook"
    else:
        mask = pd.Series(False, index=gdf.index)

    return gdf.loc[mask].copy()


def _display_value(value: Any) -> str:
    """Maak een attribuutwaarde geschikt voor legenda, tooltip en popup."""
    cleaned = clean_display_value(value)
    return cleaned if cleaned else UNKNOWN_LABEL


def _numeric_sort_key(value: str) -> tuple[int, float | str]:
    """
    Sorteer legenda-items: numerieke waarden oplopend, onbekend achteraan.
    """
    if value == UNKNOWN_LABEL:
        return (1, value)

    number = pd.to_numeric(str(value).replace(",", "."), errors="coerce")
    if pd.notna(number):
        return (0, float(number))

    return (0, value.lower())


def _is_numeric_attribute(values: list[str]) -> bool:
    """
    Bepaal of een attribuut vooral numeriek is.

    Bij jaren en wegvaknummers willen we de legenda oplopend sorteren.
    """
    real_values = [value for value in values if value != UNKNOWN_LABEL]
    if not real_values:
        return False

    numeric_count = 0
    for value in real_values:
        number = pd.to_numeric(str(value).replace(",", "."), errors="coerce")
        if pd.notna(number):
            numeric_count += 1

    return numeric_count == len(real_values)


def _sort_legend_values(values: list[str]) -> list[str]:
    """Sorteer legenda-items numeriek als dat kan, anders alfabetisch."""
    unique_values = list(dict.fromkeys(values))

    if _is_numeric_attribute(unique_values):
        return sorted(unique_values, key=_numeric_sort_key)

    return sorted(unique_values, key=lambda value: (value == UNKNOWN_LABEL, value.lower()))


def _color_for_index(index: int, total: int) -> str:
    """
    Geef een stabiele kleur terug voor een legendawaarde.

    We gebruiken een vaste HSL-spreiding. Daardoor zijn veel verschillende
    waarden beter te onderscheiden dan met een korte vaste kleurenlijst.
    """
    if total <= 1:
        hue = 205
    else:
        # Golden-angle spreiding voorkomt dat opeenvolgende categorieën te veel
        # op elkaar lijken.
        hue = (205 + index * 137.508) % 360

    saturation = 68
    lightness = 54
    return f"hsl({hue:.1f}, {saturation}%, {lightness}%)"


def build_value_color_mapping(values: list[str]) -> tuple[dict[str, str], list[LegendItem]]:
    """
    Bouw kleurmapping en legenda-items voor de opgegeven attribuutwaarden.
    """
    sorted_values = _sort_legend_values(values)
    total = len(sorted_values)

    color_by_value: dict[str, str] = {}
    legend_items: list[LegendItem] = []

    for index, value in enumerate(sorted_values):
        if value == UNKNOWN_LABEL:
            color = "#bdbdbd"
        else:
            color = _color_for_index(index, total)

        color_by_value[value] = color
        legend_items.append(LegendItem(label=value, color=color))

    return color_by_value, legend_items


def _base_map(gdf_web: gpd.GeoDataFrame) -> folium.Map:
    """Maak een basiskaart die op de rijstroken wordt ingezoomd."""
    minx, miny, maxx, maxy = gdf_web.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap",
        max_zoom=22,
    )

    if gdf_web.total_bounds is not None:
        m.fit_bounds([[miny, minx], [maxy, maxx]])

    return m


def _build_popup_html(row: pd.Series, selected_label: str, selected_column: str) -> str:
    """Maak compacte popup-inhoud voor één rijstrook."""
    lines = []

    name = clean_display_value(row.get("naam", "")) or clean_display_value(row.get("nummer", ""))
    if name:
        lines.append(f"<b>Naam:</b> {html.escape(name)}")

    selected_value = _display_value(row.get(selected_column, ""))
    lines.append(f"<b>{html.escape(selected_label)}:</b> {html.escape(selected_value)}")

    for label, candidates in OVERVIEW_POPUP_COLUMNS.items():
        column = next((candidate for candidate in candidates if candidate in row.index), None)
        if column is None:
            continue

        value = _display_value(row.get(column, ""))
        lines.append(f"<b>{html.escape(label)}:</b> {html.escape(value)}")

    return "<br>".join(lines)


def _add_legend(m: folium.Map, title: str, legend_items: list[LegendItem]) -> None:
    """Voeg een Leaflet-achtige legenda linksonder toe."""
    if not legend_items:
        legend_html = "Geen data beschikbaar."
    else:
        rows = []
        for item in legend_items:
            rows.append(
                "<div style='display:flex;align-items:center;margin:2px 0;'>"
                f"<span style='background:{html.escape(item.color)};"
                "width:18px;height:18px;display:inline-block;margin-right:8px;"
                "opacity:0.8;border:1px solid #333;'></span>"
                f"<span>{html.escape(item.label)}</span>"
                "</div>"
            )
        legend_html = "\n".join(rows)

    html_block = f"""
    <div style="
        position: fixed;
        bottom: 18px;
        left: 18px;
        z-index: 9999;
        background: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
        max-height: 70vh;
        max-width: 280px;
        overflow-y: auto;
        font-family: sans-serif;
        font-size: 12px;
    ">
        <b>{html.escape(title)}</b><br>
        {legend_html}
    </div>
    """
    m.get_root().html.add_child(folium.Element(html_block))


def build_overview_map(
    road_gdf: gpd.GeoDataFrame,
    attribute_label: str,
) -> OverviewMapResult:
    """
    Bouw de alleen-lezen overzichtskaart voor rijstroken.

    Parameters
    ----------
    road_gdf:
        GeoDataFrame van de geselecteerde weg in RD New (EPSG:28992).
    attribute_label:
        Label zoals gekozen in de UI, bijvoorbeeld `Jaar deklaag`.
    """
    rijstroken = _rijstroken_only(road_gdf)

    if rijstroken.empty:
        # Fallbackkaart op Nederland, zodat Streamlit toch iets kan tonen.
        fallback_map = folium.Map(location=[52.2, 5.4], zoom_start=8, tiles="OpenStreetMap", max_zoom=22)
        _add_legend(fallback_map, attribute_label, [])
        return OverviewMapResult(
            folium_map=fallback_map,
            row_count=0,
            legend_items=[],
            selected_column=None,
        )

    selected_column = resolve_overview_attribute(rijstroken, attribute_label)
    if selected_column is None:
        fallback_map = folium.Map(location=[52.2, 5.4], zoom_start=8, tiles="OpenStreetMap", max_zoom=22)
        _add_legend(fallback_map, attribute_label, [])
        return OverviewMapResult(
            folium_map=fallback_map,
            row_count=len(rijstroken),
            legend_items=[],
            selected_column=None,
        )

    values = [_display_value(value) for value in rijstroken[selected_column]]
    color_by_value, legend_items = build_value_color_mapping(values)

    rijstroken = rijstroken.copy()
    rijstroken["__overview_value"] = values
    rijstroken["__overview_color"] = [color_by_value[value] for value in values]

    rijstroken_web = rijstroken.to_crs(epsg=4326)
    m = _base_map(rijstroken_web)

    # Popupvelden voorbereiden in eenvoudige tekstkolommen. We selecteren straks
    # alleen deze kolommen voor GeoJSON, zodat exotische pandas-types uit de
    # bronexport niet per ongeluk JSON-serialisatieproblemen geven.
    popup_fields: list[str] = []
    popup_aliases: list[str] = []

    rijstroken_web["__popup_naam"] = [
        clean_display_value(row.get("naam", "")) or clean_display_value(row.get("nummer", ""))
        for _, row in rijstroken_web.iterrows()
    ]
    if rijstroken_web["__popup_naam"].astype(str).str.len().gt(0).any():
        popup_fields.append("__popup_naam")
        popup_aliases.append("Naam")

    rijstroken_web["__popup_selected"] = rijstroken_web["__overview_value"]
    popup_fields.append("__popup_selected")
    popup_aliases.append(attribute_label)

    for label, candidates in OVERVIEW_POPUP_COLUMNS.items():
        if label == attribute_label:
            continue

        column = next((candidate for candidate in candidates if candidate in rijstroken_web.columns), None)
        if column is None:
            continue

        helper_column = "__popup_" + "".join(ch if ch.isalnum() else "_" for ch in label.lower())
        rijstroken_web[helper_column] = rijstroken_web[column].apply(_display_value)
        popup_fields.append(helper_column)
        popup_aliases.append(label)

    def style_fn(feature):
        props = feature.get("properties", {})
        color = props.get("__overview_color", "#808080")
        return {
            "color": color,
            "fillColor": color,
            "weight": 4,
            "opacity": 0.85,
            "fillOpacity": 0.65,
        }

    tooltip = folium.GeoJsonTooltip(
        fields=["__overview_value"],
        aliases=[attribute_label],
        style="font-size: 11px;",
    )

    popup = folium.GeoJsonPopup(
        fields=popup_fields,
        aliases=popup_aliases,
        localize=False,
        labels=True,
        style="font-size: 12px;",
        max_width=350,
    )

    geojson_columns = list(dict.fromkeys(["geometry", "__overview_value", "__overview_color", *popup_fields]))

    folium.GeoJson(
        rijstroken_web[geojson_columns],
        style_function=style_fn,
        tooltip=tooltip,
        popup=popup,
    ).add_to(m)

    _add_legend(m, attribute_label, legend_items)

    return OverviewMapResult(
        folium_map=m,
        row_count=len(rijstroken),
        legend_items=legend_items,
        selected_column=selected_column,
    )
