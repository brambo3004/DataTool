"""
Streamlit-front-end voor de iASSET Advisor.

Belangrijk ontwerpprincipe:
Deze file bevat alleen UI-flow: knoppen, formulieren, layout en Streamlit state.
De GIS-, regel-, advies-, kaart- en exportlogica staat in `iasset_tool/`.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from iasset_tool.advisor import generate_grouped_proposals
from iasset_tool.changes import (
    add_log_entry,
    apply_change_to_data,
    available_export_profiles,
    build_export_dataframe,
    collect_changed_ids,
    get_export_profile_columns,
    load_autosave,
    save_autosave,
    summarize_export_profile,
)
from iasset_tool.config import (
    APP_VERSION,
    AUTOSAVE_FILE,
    DEFAULT_EXPORT_PROFILE,
    HIERARCHY_RANK,
    ISSUE_CATEGORIES,
    MAINTENANCE_RULES_WARNINGS,
    SEGMENTATION_ATTRIBUTES,
    SORT_DIAGNOSTICS_SCHEMA_VERSION,
    maintenance_rules_summary,
)
from iasset_tool.data_loader import LoadResult, load_iasset_data
from iasset_tool.geometry import build_graph_from_geometry
from iasset_tool.map_view import build_road_map
from iasset_tool.maintenance_control import (
    ACTION_FOLLOW_UP_COLUMNS,
    ACTION_FOLLOW_UP_STATUS_OPTIONS,
    action_work_queue_summary,
    build_control_point_object_details,
    build_maintenance_control,
    build_maintenance_control_workbook,
    filter_action_work_queue,
    merge_action_work_queue_edits,
    summarize_action_projects,
    read_action_lists_safely,
    read_maintenance_exports,
)
from iasset_tool.maintenance_map import build_maintenance_control_map
from iasset_tool.object_editor import (
    editable_fields_for_profile,
    missing_profile_columns,
    object_preview_dataframe,
    search_objects,
)
from iasset_tool.overview_map import available_overview_attributes, build_overview_map, render_overview_map_html
from iasset_tool.pdok import get_pdok_hectopunten_visual_only
from iasset_tool.performance import measure_step, performance_dataframe
from iasset_tool.rules import category_counts, check_rules, violation_key
from iasset_tool.sorting_diagnostics import build_sort_diagnostics
from iasset_tool.state import (
    init_session_state,
    reset_after_data_source_change,
    reset_after_road_change,
    reset_selection,
)
from iasset_tool.utils import clean_display_value, make_short_hash, normalize_text, parse_hm_sort, sanitize_filename


st.set_page_config(layout="wide", page_title="iASSET Tool - Smart Advisor")

# Meet alleen de huidige Streamlit-run.
# In v0.9 bleef de performance-log over meerdere interacties staan. Daardoor
# leek het totaal in de zijbalk soms veel hoger dan de wachttijd van de actie
# die je net uitvoerde.
st.session_state["performance_log"] = {}


def render_version_badge() -> None:
    """
    Toon het actuele versienummer klein in beeld.

    Waarom?
    Streamlit Cloud/GitHub kan soms nog een oudere build tonen. Een zichtbaar
    versienummer maakt meteen duidelijk welke codeversie echt draait.
    """
    st.sidebar.caption(f"Versie {APP_VERSION}")

    rules_summary = maintenance_rules_summary()
    with st.sidebar.expander("Beheerregels", expanded=False):
        st.caption(
            "v0.26 gebruikt één configuratiebestand voor domeinregels zoals "
            "primaire subthema's, uitzonderingen, categoriefamilies en knipvelden."
        )
        st.write(f"Bron: {rules_summary.get('bron', 'onbekend')}")
        st.caption(str(rules_summary.get("pad", "")))

        if MAINTENANCE_RULES_WARNINGS:
            for warning in MAINTENANCE_RULES_WARNINGS:
                st.warning(warning)
        else:
            st.success("Beheerregels geladen.")

        st.write(
            {
                "primaire_subthema's": rules_summary.get("primaire_subthemas", 0),
                "uitzonderingen": rules_summary.get("uitzonderingssubthemas", 0),
                "categoriefamilies": rules_summary.get("categoriefamilies", 0),
                "knipvelden": rules_summary.get("knipvelden_project_adviseur", 0),
                "wegrichtingen": rules_summary.get("wegrichtingen", 0),
            }
        )

    st.markdown(
        f"""
        <style>
        .iasset-version-badge {{
            position: fixed;
            right: 0.75rem;
            bottom: 0.45rem;
            z-index: 9999;
            padding: 0.15rem 0.45rem;
            border-radius: 0.4rem;
            background: rgba(240, 242, 246, 0.92);
            color: #555;
            font-size: 0.72rem;
            border: 1px solid rgba(49, 51, 63, 0.12);
        }}
        </style>
        <div class="iasset-version-badge">iASSET DataTool {APP_VERSION}</div>
        """,
        unsafe_allow_html=True,
    )


render_version_badge()


@st.cache_data(show_spinner=False)
def cached_load_default_data() -> LoadResult:
    """
    Laad de vaste bronbestanden één keer per Streamlit-cache.

    Let op: de werkdata in session_state wordt daarna gemuteerd.
    We muteren dus niet rechtstreeks het gecachete object.
    """
    return load_iasset_data()


@st.cache_data(show_spinner=False)
def cached_load_uploaded_data(file_payloads: tuple[tuple[str, bytes], ...]) -> LoadResult:
    """
    Laad geüploade bronbestanden uit bytes.

    De upload staat in het geheugen van de Streamlit-sessie. We slaan hem hier
    niet automatisch als bronbestand op schijf op.
    """
    return load_iasset_data(input_files=file_payloads)


@st.cache_data(show_spinner=False)
def cached_load_maintenance_exports(file_payloads: tuple[tuple[str, bytes], ...]):
    """
    Laad onderhoudsexports voor de onderhoudscontrole.

    Dit is bewust een aparte uploadstroom naast de paspoortexport. De
    onderhoudsexport heeft meestal geen geometrie en hoort daarom niet door de
    gewone iASSET-geometrielader te lopen.
    """
    return read_maintenance_exports(file_payloads)


@st.cache_data(show_spinner=False)
def cached_load_previous_action_lists(file_payloads: tuple[tuple[str, bytes], ...]):
    """
    Laad eerder ingevulde Onderhoudscontrole-actielijsten.

    Deze upload is optioneel. De inhoud wordt alleen gebruikt om
    databeheerdervelden opnieuw in de nieuwe actielijst te zetten.
    """
    return read_action_lists_safely(file_payloads)


def build_data_source_key(file_payloads: tuple[tuple[str, bytes], ...]) -> str:
    """
    Maak een stabiele sleutel voor de actieve databron.

    Voor de vaste bestanden gebruiken we een vaste sleutel. Voor uploads maken
    we een hash op basis van bestandsnaam, bestandsgrootte en inhoud.
    """
    if not file_payloads:
        return "local_files"

    hash_parts: list[str | bytes] = []
    for file_name, content in file_payloads:
        hash_parts.extend([file_name, str(len(content)), content])

    return f"upload_{make_short_hash(hash_parts)}"


def build_data_source_label(file_payloads: tuple[tuple[str, bytes], ...]) -> str:
    """Maak een leesbare naam voor de actieve databron."""
    if not file_payloads:
        return "Vaste bestandenmap naast app.py"

    names = [name for name, _ in file_payloads]
    if len(names) == 1:
        return f"Upload: {names[0]}"

    return "Upload: " + ", ".join(names[:3]) + (" ..." if len(names) > 3 else "")


def autosave_path_for_source(source_key: str) -> Path:
    """
    Bepaal het autosave-bestand voor de actieve databron.

    Waarom per databron?
    ``sys_id`` is alleen stabiel binnen één ingelezen export. Een autosave van
    export A automatisch toepassen op export B kan dus verkeerde objecten raken.
    """
    if source_key == "local_files":
        return Path(AUTOSAVE_FILE)

    return Path(".autosave") / f"autosave_log_{sanitize_filename(source_key)}.csv"


def current_autosave_file() -> Path:
    """Geef het autosave-pad voor de huidige sessie terug."""
    source_key = st.session_state.get("active_data_source_key", "local_files")
    return autosave_path_for_source(str(source_key))


def persist_change_log() -> None:
    """Schrijf het huidige wijzigingslogboek naar de autosave van deze databron."""
    save_autosave(st.session_state["change_log"], current_autosave_file())


def get_performance_log() -> dict[str, float]:
    """Geef de performance-log voor deze sessie terug."""
    return st.session_state.setdefault("performance_log", {})


def current_data_revision_key() -> str:
    """
    Maak een lichte revisiesleutel voor cachebare wegafhankelijke berekeningen.

    De sleutel verandert wanneer de actieve databron of het wijzigingslogboek
    verandert. Dat is voldoende voor de huidige regels, omdat die vooral op
    paspoortvelden en onderhoudsprojectwaarden draaien.
    """
    source_key = str(st.session_state.get("active_data_source_key", "local_files"))
    parts: list[str] = [source_key, str(len(st.session_state.get("change_log", [])))]

    for entry in st.session_state.get("change_log", []):
        parts.extend(
            [
                str(entry.get("ID", "")),
                str(entry.get("Veld", "")),
                str(entry.get("Nieuw", "")),
                str(entry.get("Status", "")),
            ]
        )

    return make_short_hash(parts)


def road_extent_key(gdf) -> str:
    """
    Maak een compacte sleutel voor de geometrische omvang van een wegselectie.

    Dit gebruiken we voor PDOK-caching. Afronden op decimeters voorkomt dat
    onbelangrijke floating-pointverschillen de cache onnodig ongeldig maken.
    """
    if gdf is None or gdf.empty:
        return "empty"

    try:
        bounds = tuple(round(float(value), 1) for value in gdf.total_bounds)
    except Exception:
        return f"rows_{len(gdf)}"

    return f"rows_{len(gdf)}_bounds_{bounds}"


def get_quality_issues_for_road(selected_road: str, road_gdf, graph) -> list[dict]:
    """
    Bereken datakwaliteitsissues één keer per weg/datasetrevisie.

    In v0.8 werd `check_rules()` binnen één Streamlit-run zowel in de werklijst
    als bij de kaartkleuring aangeroepen. Deze cache voorkomt die dubbele zware
    regelcheck zonder de domeinfunctie zelf Streamlit-afhankelijk te maken.
    """
    cache = st.session_state.setdefault("quality_issues_cache", {})
    cache_key = (
        str(st.session_state.get("active_data_source_key", "local_files")),
        selected_road,
        current_data_revision_key(),
    )

    if cache.get("key") != cache_key:
        cache.clear()
        cache["key"] = cache_key
        cache["issues"] = measure_step(
            get_performance_log(),
            "Data Kwaliteit regels",
            check_rules,
            road_gdf,
            graph,
        )

    return list(cache.get("issues", []))


def is_violation_ignored(violation: dict) -> bool:
    """
    Controleer of een issue genegeerd is.

    Oude sessies bewaarden alleen object-id's. Nieuwe sessies bewaren
    rule-code + object-id, zodat meerdere issues op hetzelfde object apart
    kunnen blijven bestaan.
    """
    ignored = st.session_state.get("ignored_errors", set())
    return violation_key(violation) in ignored or violation.get("id") in ignored


def get_pdok_hectopunten_cached(selected_road: str, road_gdf):
    """
    Haal PDOK-hectometerpunten op met sessiecache per weg/extent.

    PDOK is alleen visuele ondersteuning. Daarom halen we deze laag pas op als
    de gebruiker hem aanzet en bewaren we het resultaat voor dezelfde weg.
    """
    cache = st.session_state.setdefault("pdok_hectopunten_cache", {})
    cache_key = (
        str(st.session_state.get("active_data_source_key", "local_files")),
        selected_road,
        road_extent_key(road_gdf),
    )

    if cache_key not in cache:
        cache[cache_key] = measure_step(
            get_performance_log(),
            "PDOK hectometerpunten",
            get_pdok_hectopunten_visual_only,
            road_gdf,
        )

    return cache[cache_key]


def register_change(object_id: int, field: str, old_value, new_value) -> None:
    """
    Pas wijziging toe op data én logboek.

    Deze wrapper houdt de UI-code kort en zorgt dat autosave altijd meeloopt.
    Afgeleide kolommen zoals ``subthema_clean`` en ``hm_sort`` worden hier ook
    bijgewerkt, zodat een paspoortmutatie direct doorwerkt in regels, sortering
    en advieslogica.
    """
    raw_gdf = st.session_state["data_complete"]
    applied = apply_change_to_data(raw_gdf, object_id, field, new_value)

    if applied and object_id in raw_gdf.index:
        if field == "subthema":
            subthema_clean = normalize_text(new_value)
            raw_gdf.at[object_id, "subthema_clean"] = subthema_clean
            raw_gdf.at[object_id, "Rank"] = HIERARCHY_RANK.get(subthema_clean, 4)

        if field == "Metrering":
            raw_gdf.at[object_id, "hm_sort"] = parse_hm_sort(new_value)

    status = "Succes" if applied else "Niet toegepast"
    add_log_entry(st.session_state["change_log"], object_id, field, old_value, new_value, status=status)

    # Een wijziging kan Data Kwaliteit of kaartkleuring beïnvloeden.
    # Daarom gooien we afgeleide caches weg; de brondata zelf blijft uiteraard
    # in session_state staan.
    st.session_state["quality_issues_cache"] = {}

    fields_affecting_groups = {
        "Onderhoudsproject",
        "subthema",
        "verhardingssoort",
        "Soort verharding_N",
        "Soort deklaag specifiek",
        "Soort conservering",
        "Jaar aanleg",
        "Jaar deklaag",
        "Jaar conservering",
        "Jaar herstrating",
        "Besteknummer",
        *SEGMENTATION_ATTRIBUTES,
    }

    if field in fields_affecting_groups:
        st.session_state["computed_groups"] = None

    # Als de wegselectie of het subthema verandert, moet het wegafhankelijke
    # netwerk opnieuw worden opgebouwd bij de volgende rerun.
    if field in {"Wegnummer", "subthema"}:
        st.session_state.pop("graph_current", None)
        st.session_state.pop("last_road", None)

    persist_change_log()


def restore_group_for_object(object_id: int) -> None:
    """
    Zet een adviesgroep weer open als een Onderhoudsproject-wijziging wordt teruggedraaid.
    """
    groups = st.session_state.get("computed_groups") or {}

    for group_id, group_data in groups.items():
        if object_id in group_data.get("ids", []):
            st.session_state["processed_groups"].discard(group_id)
            return


def load_data_into_session() -> None:
    """
    Laad data en speel autosave-wijzigingen opnieuw af.
    """
    source_key = st.session_state.get("active_data_source_key", "local_files")
    if source_key == "local_files":
        result = measure_step(get_performance_log(), "Data laden", cached_load_default_data)
    else:
        file_payloads = tuple(st.session_state.get("active_upload_payloads") or ())
        result = measure_step(get_performance_log(), "Data laden", cached_load_uploaded_data, file_payloads)

    st.session_state["autosave_file"] = str(current_autosave_file())

    # Belangrijk: we maken een copy, zodat session_state losstaat van de cache.
    st.session_state["data_complete"] = result.gdf.copy()
    st.session_state["invalid_geometry_rows"] = result.invalid_geometry_rows
    st.session_state["load_warnings"] = result.warnings

    autosave_log = load_autosave(current_autosave_file())
    st.session_state["change_log"] = autosave_log

    restored = 0
    for entry in autosave_log:
        if apply_change_to_data(
            st.session_state["data_complete"],
            entry.get("ID"),
            entry.get("Veld"),
            entry.get("Nieuw"),
        ):
            restored += 1

    if restored:
        st.toast(f"🔄 {restored} wijzigingen hersteld uit autosave.", icon="💾")


# --- Databron kiezen -------------------------------------------------------

st.sidebar.title("iASSET Advisor")

with st.sidebar.expander("Databron", expanded="data_complete" not in st.session_state):
    uploaded_files = st.file_uploader(
        "Upload actuele iASSET-export",
        type=["csv", "xlsx", "xls", "xlsm"],
        accept_multiple_files=True,
        help=(
            "Upload één gecombineerd exportbestand, of meerdere deelbestanden. "
            "Laat dit leeg om de vaste CSV-bestanden naast app.py te gebruiken."
        ),
    )

    candidate_payloads: tuple[tuple[str, bytes], ...] = tuple(
        (uploaded_file.name, uploaded_file.getvalue()) for uploaded_file in uploaded_files
    ) if uploaded_files else tuple()

    candidate_source_key = build_data_source_key(candidate_payloads)
    candidate_source_label = build_data_source_label(candidate_payloads)

    if "active_data_source_key" not in st.session_state:
        st.session_state["active_data_source_key"] = candidate_source_key
        st.session_state["active_upload_payloads"] = candidate_payloads
        st.session_state["active_data_source_label"] = candidate_source_label

    active_source_key = st.session_state.get("active_data_source_key", "local_files")
    active_source_label = st.session_state.get("active_data_source_label", "Vaste bestandenmap naast app.py")

    st.caption(f"Actieve databron: {active_source_label}")

    if candidate_source_key != active_source_key:
        st.warning("Er is een andere databron gekozen dan de dataset die nu actief is.")

        if st.session_state.get("change_log"):
            st.warning(
                "Er staan nog wijzigingen in het logboek. Exporteer die eerst, "
                "of laad de nieuwe databron bewust om het werkgeheugen te vervangen."
            )

        if st.button("Gebruik deze databron"):
            reset_after_data_source_change(st.session_state)
            st.session_state["active_data_source_key"] = candidate_source_key
            st.session_state["active_upload_payloads"] = candidate_payloads
            st.session_state["active_data_source_label"] = candidate_source_label
            st.rerun()

    if st.button("Actieve dataset opnieuw laden", help="Leest de actieve bron opnieuw in en wist tijdelijke selecties."):
        # De cache kan nog de vorige inhoud bevatten; daarom maken we hem leeg
        # wanneer de gebruiker bewust opnieuw wil laden.
        #
        # Belangrijk: herladen mag niet stiekem overschakelen naar een net gekozen
        # upload. Daarvoor is de knop "Gebruik deze databron" bedoeld, zeker als
        # er nog mutaties in het logboek staan.
        reload_source_key = st.session_state.get("active_data_source_key", "local_files")
        reload_payloads = tuple(st.session_state.get("active_upload_payloads") or ())
        reload_label = st.session_state.get("active_data_source_label", "Vaste bestandenmap naast app.py")

        st.cache_data.clear()
        reset_after_data_source_change(st.session_state)
        st.session_state["active_data_source_key"] = reload_source_key
        st.session_state["active_upload_payloads"] = reload_payloads
        st.session_state["active_data_source_label"] = reload_label
        st.rerun()


with st.sidebar.expander("Onderhoudsexport voor controle", expanded=False):
    maintenance_uploaded_files = st.file_uploader(
        "Upload onderhoudsexport",
        type=["csv", "xlsx", "xls", "xlsm"],
        accept_multiple_files=True,
        key="maintenance_export_uploads",
        help=(
            "Gebruik dit voor de onderhoudscontrole: de onderhoudsregels/projecten "
            "worden naast de actieve paspoortexport gelegd."
        ),
    )

    maintenance_payloads: tuple[tuple[str, bytes], ...] = tuple(
        (uploaded_file.name, uploaded_file.getvalue()) for uploaded_file in maintenance_uploaded_files
    ) if maintenance_uploaded_files else tuple()

    st.session_state["maintenance_upload_payloads"] = maintenance_payloads

    if maintenance_payloads:
        st.caption(
            "Onderhoudsexport klaar voor controle: "
            + ", ".join(name for name, _ in maintenance_payloads[:3])
            + (" ..." if len(maintenance_payloads) > 3 else "")
        )
    else:
        st.caption("Geen onderhoudsexport geüpload.")


    previous_action_files = st.file_uploader(
        "Optioneel: oude ingevulde Onderhoudscontrole-actielijst",
        type=["csv", "xlsx", "xls", "xlsm"],
        accept_multiple_files=True,
        key="previous_action_list_uploads",
        help=(
            "Upload hier een eerder ingevulde Onderhoudscontrole_Actielijst of oude Fase4_Actielijst. Als dezelfde "
            "controlepunten opnieuw voorkomen, neemt de tool beoordeling, "
            "afhandelstatus, actiehouder en opmerkingen opnieuw mee."
        ),
    )

    previous_action_payloads: tuple[tuple[str, bytes], ...] = tuple(
        (uploaded_file.name, uploaded_file.getvalue()) for uploaded_file in previous_action_files
    ) if previous_action_files else tuple()

    st.session_state["previous_action_list_payloads"] = previous_action_payloads

    if previous_action_payloads:
        st.caption(
            "Eerdere actielijst klaar voor overname: "
            + ", ".join(name for name, _ in previous_action_payloads[:3])
            + (" ..." if len(previous_action_payloads) > 3 else "")
        )
    else:
        st.caption("Geen eerdere actielijst geüpload.")


# --- Applicatie-initialisatie ---------------------------------------------

if "data_complete" not in st.session_state:
    with st.spinner("Data laden..."):
        load_data_into_session()

init_session_state(st.session_state)

raw_gdf = st.session_state["data_complete"]

if raw_gdf.empty:
    st.error("Geen geldige iASSET-objecten gevonden. Controleer de bronbestanden en de kolom 'gps coordinaten'.")
    for warning in st.session_state.get("load_warnings", []):
        st.warning(warning)
    st.stop()

if "sys_id" not in raw_gdf.columns:
    st.cache_data.clear()
    st.rerun()


# --- Sidebar ---------------------------------------------------------------

with st.sidebar.expander("Datastatus", expanded=False):
    warnings = st.session_state.get("load_warnings", [])
    invalid_geometry_rows = st.session_state.get("invalid_geometry_rows")

    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.caption("Geen inleeswaarschuwingen.")

    if invalid_geometry_rows is not None and not invalid_geometry_rows.empty:
        st.caption(f"{len(invalid_geometry_rows)} rijen met ongeldige of lege geometrie overgeslagen.")
        st.dataframe(invalid_geometry_rows.head(25), use_container_width=True, hide_index=True)

        invalid_csv = invalid_geometry_rows.to_csv(index=False, sep=";").encode("utf-8-sig")
        st.download_button(
            "📥 Download inleesfouten",
            data=invalid_csv,
            file_name="iASSET_Inleesfouten_Geometrie.csv",
            mime="text/csv",
            help="Download alle overgeslagen geometrie-rijen voor controle of herstel in de bronexport.",
        )

with st.sidebar.expander("Performance", expanded=False):
    df_performance = performance_dataframe(st.session_state.get("performance_log"))

    if df_performance.empty:
        st.caption("Nog geen meetpunten beschikbaar.")
    else:
        st.dataframe(df_performance, use_container_width=True, hide_index=True)
        st.caption(f"Totaal gemeten in deze run: {df_performance['Seconden'].sum():.3f} seconden.")

all_roads = sorted(
    {
        str(value).strip()
        for value in raw_gdf["Wegnummer"].dropna().unique()
        if str(value).strip() and str(value).strip().lower() != "nan"
    }
)

if not all_roads:
    st.error("Geen Wegnummer-waarden gevonden in de data.")
    st.stop()

selected_road = st.sidebar.selectbox("Kies Wegnummer", all_roads)

road_gdf = raw_gdf[raw_gdf["Wegnummer"] == selected_road].copy()

if "graph_current" not in st.session_state or st.session_state.get("last_road") != selected_road:
    with st.spinner("Netwerk analyseren..."):
        st.session_state["graph_current"] = measure_step(
            get_performance_log(),
            "Netwerk analyseren",
            build_graph_from_geometry,
            road_gdf,
        )
        reset_after_road_change(st.session_state, selected_road)

graph_road = st.session_state["graph_current"]

overview_attribute = None
overview_scope = "Geselecteerde weg"
overview_gdf = road_gdf
overview_label = selected_road


# --- Layout ----------------------------------------------------------------

col_map, col_inspector = st.columns([3, 2])


# --- Rechterkolom: werklijst ----------------------------------------------

with col_inspector:
    st.subheader("Werklijst")

    mode = st.radio(
        "Modus:",
        ["🔍 Data Kwaliteit", "🏗️ Project Adviseur", "🛠️ Onderhoudscontrole", "🧾 Objectinspecteur", "🗺️ Overzicht"],
        horizontal=True,
        on_change=lambda: reset_selection(st.session_state),
    )

    st.divider()

    if mode == "🔍 Data Kwaliteit":
        all_violations = get_quality_issues_for_road(selected_road, road_gdf, graph_road)
        open_violations = [
            violation
            for violation in all_violations
            if not is_violation_ignored(violation)
        ]

        counts_by_category = category_counts(open_violations)
        known_categories = [category for category in ISSUE_CATEGORIES if category in counts_by_category]
        extra_categories = sorted(category for category in counts_by_category if category not in ISSUE_CATEGORIES)
        category_options = ["Alle categorieën", *known_categories, *extra_categories]

        if category_options:
            selected_issue_category = st.selectbox(
                "Filter issuecategorie",
                category_options,
                format_func=lambda value: (
                    f"{value} ({len(open_violations)})"
                    if value == "Alle categorieën"
                    else f"{value} ({counts_by_category.get(value, 0)})"
                ),
            )
        else:
            selected_issue_category = "Alle categorieën"

        if selected_issue_category == "Alle categorieën":
            violations = open_violations
        else:
            violations = [
                violation
                for violation in open_violations
                if violation.get("category") == selected_issue_category
            ]

        if not open_violations:
            st.success("Schoon! Geen datakwaliteit issues.")

            if all_violations and st.button("🔄 Reset genegeerde meldingen"):
                st.session_state["ignored_errors"] = set()
                st.rerun()
        elif not violations:
            st.info("Geen open issues binnen deze categorie.")
        else:
            st.write(f"**{len(violations)} issues gevonden**")
            if selected_issue_category != "Alle categorieën":
                st.caption(f"{len(open_violations)} open issues in totaal over alle categorieën.")

            if st.session_state.get("ignored_errors") and st.button("🔄 Reset genegeerde meldingen"):
                st.session_state["ignored_errors"] = set()
                st.rerun()

            with st.container(height=400):
                for index, violation in enumerate(violations):
                    object_id = violation["id"]
                    key_suffix = violation_key(violation)
                    is_selected = st.session_state["selected_error_id"] == object_id
                    container_args = {"border": True} if is_selected else {}

                    with st.container(**container_args):
                        if is_selected:
                            st.markdown("**:blue-background[GESELECTEERD]**")

                        c_text, c_show, c_ignore = st.columns([2, 1, 1])

                        with c_text:
                            st.markdown(f"**{violation['subthema']}**")
                            st.caption(
                                f"{violation.get('category', 'Overig')} · "
                                f"{violation.get('severity', violation.get('type', 'info')).upper()} · "
                                f"{violation.get('rule_code', '-')}"
                            )
                            st.caption(violation["msg"])

                        with c_show:
                            if st.button("👁️", key=f"btn_err_show_{key_suffix}_{index}", help="Toon op kaart"):
                                st.session_state["selected_error_id"] = object_id
                                geom_web = road_gdf.loc[[object_id]].to_crs(epsg=4326).geometry.iloc[0]
                                st.session_state["zoom_bounds"] = geom_web.bounds
                                st.rerun()

                        with c_ignore:
                            if st.button("🗑️", key=f"btn_err_ign_{key_suffix}_{index}", help="Negeer deze melding"):
                                st.session_state["ignored_errors"].add(violation_key(violation))
                                if is_selected:
                                    st.session_state["selected_error_id"] = None
                                    st.session_state["zoom_bounds"] = None
                                st.rerun()

                        if not is_selected:
                            st.divider()

            selected_error_id = st.session_state["selected_error_id"]

            if selected_error_id is not None and selected_error_id in road_gdf.index:
                st.divider()
                st.markdown(f"#### Corrigeer ID {selected_error_id}")

                row = road_gdf.loc[selected_error_id]
                violation_info = next(
                    (violation for violation in all_violations if violation["id"] == selected_error_id),
                    None,
                )

                cols_to_fix = violation_info["missing_cols"] if violation_info else ["Onderhoudsproject"]
                inputs = {}

                if not cols_to_fix:
                    st.info("Voor deze melding is geen automatisch correctieveld beschikbaar.")
                else:
                    for column in cols_to_fix:
                        current_value = clean_display_value(row.get(column, ""))
                        inputs[column] = st.text_input(
                            f"Vul in: {column}",
                            value=current_value,
                            key=f"fix_{column}_{selected_error_id}",
                        )

                    if st.button("Opslaan Correctie"):
                        for column, new_value in inputs.items():
                            old_value = raw_gdf.at[selected_error_id, column] if column in raw_gdf.columns else ""
                            if clean_display_value(old_value) != clean_display_value(new_value):
                                register_change(selected_error_id, column, old_value, new_value)

                        st.success("Opgeslagen.")
                        st.session_state["selected_error_id"] = None
                        st.rerun()

    elif mode == "🏗️ Project Adviseur":
        if st.session_state.get("computed_groups") is None:
            with st.spinner("Adviesgroepen berekenen..."):
                st.session_state["computed_groups"] = measure_step(
                    get_performance_log(),
                    "Project Adviseur",
                    generate_grouped_proposals,
                    road_gdf,
                    graph_road,
                )

        all_groups = st.session_state["computed_groups"] or {}

        active_groups = {
            group_id: group_data
            for group_id, group_data in all_groups.items()
            if group_id not in st.session_state["processed_groups"]
            and group_id not in st.session_state["ignored_groups"]
        }

        if not active_groups:
            st.success("Geen adviezen meer beschikbaar.")

            if st.button("Herberekenen / Reset"):
                st.session_state["computed_groups"] = None
                st.session_state["processed_groups"] = set()
                st.session_state["ignored_groups"] = set()
                st.rerun()
        else:
            st.write(f"**{len(active_groups)} suggesties beschikbaar**")

            sorted_items = sorted(
                active_groups.items(),
                key=lambda item: (
                    item[1].get("volgorde_nr", item[1].get("advies_volgorde", 999999)),
                    item[1].get("rank", 99),
                    item[1].get("sort_value", 0),
                    item[1].get("tie_breaker_dist", 0),
                    item[1].get("fallback_tie_breaker_dist", 0),
                ),
            )

            with st.container(height=400):
                for group_id, group_data in sorted_items:
                    count = len(group_data["ids"])

                    if "RIJBAAN" in group_id:
                        icon = "🛣️"
                    elif "FIETSPAD" in group_id:
                        icon = "🚲"
                    elif "PARALLEL" in group_id:
                        icon = "🛤️"
                    else:
                        icon = "🌳"

                    is_selected = st.session_state["selected_group_id"] == group_id
                    container_args = {"border": True} if is_selected else {}

                    with st.container(**container_args):
                        if is_selected:
                            st.markdown("**:blue-background[GESELECTEERD]**")

                        volgorde_nr = group_data.get("volgorde_nr") or group_data.get("advies_volgorde")
                        volgorde_label = f"{volgorde_nr}. " if volgorde_nr else ""
                        st.markdown(f"**{volgorde_label}{icon} {group_data['subthema'].title()}** ({count} obj)")
                        st.caption(group_data["reason"])

                        tie_breaker_source = group_data.get("tie_breaker_source", "")
                        if tie_breaker_source == "lokale_route_as":
                            st.caption("Volgorde: hectometrering + lokale route-as.")
                        elif tie_breaker_source == "lokale_route_as_overlapcluster":
                            cluster_label = group_data.get("overlap_cluster_id", "")
                            st.caption(
                                "Volgorde: overlappend hm-bereik, lokale route-as bepaalt de positie"
                                + (f" ({cluster_label})." if cluster_label else ".")
                            )

                        assignment_note = group_data.get("assignment_note", "")
                        if assignment_note and assignment_note != "Primaire ruggengraatgroep; secundaire objecten apart toegewezen.":
                            st.caption(f"Logica: {assignment_note}")

                        attached_fietspad_count = len(group_data.get("attached_fietspad_ids", []))
                        if attached_fietspad_count:
                            st.caption(f"Inclusief {attached_fietspad_count} gekoppelde fietspadobject(en).")

                        if group_data.get("review_needed"):
                            st.warning("Controle nodig: fietspadrelatie is onzeker.", icon="⚠️")

                        old_project = group_data.get("current_project", "")
                        st.markdown(
                            f"<small>Huidig: *{old_project if old_project else 'Geen'}*</small>",
                            unsafe_allow_html=True,
                        )

                        c_select, c_ignore = st.columns([1, 1])

                        with c_select:
                            label = "📍 Geselecteerd" if is_selected else "👁️ Selecteer"
                            if st.button(label, key=f"vis_{group_id}", disabled=is_selected):
                                st.session_state["selected_group_id"] = group_id
                                subset_gdf = road_gdf.loc[group_data["ids"]].to_crs(epsg=4326)
                                try:
                                    merged_geometry = subset_gdf.geometry.union_all()
                                except AttributeError:
                                    merged_geometry = subset_gdf.geometry.unary_union

                                st.session_state["zoom_bounds"] = merged_geometry.bounds
                                st.rerun()

                        with c_ignore:
                            if st.button("🗑️ Negeer", key=f"ign_{group_id}"):
                                st.session_state["ignored_groups"].add(group_id)
                                if is_selected:
                                    st.session_state["selected_group_id"] = None
                                    st.session_state["zoom_bounds"] = None
                                st.rerun()

                        if not is_selected:
                            st.divider()

            selected_group_id = st.session_state["selected_group_id"]

            if selected_group_id and selected_group_id in active_groups:
                selected_group = active_groups[selected_group_id]

                st.divider()
                st.markdown(f"#### 🏷️ Naamgeven: {selected_group_id}")
                st.info(f"Bevat {len(selected_group['ids'])} objecten. ({selected_group['reason']})")

                selected_assignment_note = selected_group.get("assignment_note", "")
                if selected_assignment_note:
                    st.caption(f"Toewijzingslogica: {selected_assignment_note}")

                attached_fietspad_count = len(selected_group.get("attached_fietspad_ids", []))
                if attached_fietspad_count:
                    st.caption(f"Deze groep bevat {attached_fietspad_count} fietspadobject(en) die als kruisend/rotondegebonden zijn gekoppeld.")

                if selected_group.get("review_needed"):
                    st.warning(
                        "Deze fietspadgroep is onzeker geclassificeerd. Controleer op de kaart of dit echt een parallelfietspad is.",
                        icon="⚠️",
                    )

                old_project_hint = selected_group.get("current_project", "")
                placeholder_text = old_project_hint if old_project_hint else "bv. N351-HRB-20.1-24.3"

                name_input = st.text_input(
                    "Projectnaam",
                    value="",
                    placeholder=placeholder_text,
                    key="proj_name_input",
                )

                if st.button("✅ Opslaan & Toepassen", type="primary"):
                    if name_input.strip():
                        new_value = clean_display_value(name_input)
                        count_updates = 0

                        for object_id in selected_group["ids"]:
                            if object_id not in raw_gdf.index:
                                continue

                            old_value = raw_gdf.at[object_id, "Onderhoudsproject"]
                            if clean_display_value(old_value) == new_value:
                                continue

                            register_change(object_id, "Onderhoudsproject", old_value, name_input)

                            if "Advies_Bron" in raw_gdf.columns:
                                apply_change_to_data(raw_gdf, object_id, "Advies_Bron", selected_group["reason"])

                            count_updates += 1

                        st.session_state["processed_groups"].add(selected_group_id)
                        st.session_state["selected_group_id"] = None
                        st.session_state["zoom_bounds"] = None

                        if count_updates:
                            st.success(f"Opgeslagen. {count_updates} objecten bijgewerkt.")
                        else:
                            st.info("Geen wijzigingen nodig, naam stond al goed.")

                        st.rerun()



    elif mode == "🛠️ Onderhoudscontrole":
        st.markdown("### 🛠️ Onderhoudscontrole")
        st.caption(
            "Leg de actieve paspoortexport naast een onderhoudsexport. "
            "De controle voert geen mutaties uit; hij laat alleen zien welke "
            "onderhoudscomplexen ontbreken of verweesd zijn."
        )

        control_scope = st.radio(
            "Controlebereik",
            ["Geselecteerde weg", "Hele dataset / wegennet"],
            horizontal=True,
            help=(
                "Gebruik 'Hele dataset / wegennet' wanneer de actieve paspoortexport "
                "alle provinciale wegen bevat. De zijbalkselectie blijft dan alleen "
                "voor kaart- en detailtabbladen gelden."
            ),
            key="maintenance_control_scope",
        )
        network_scope = control_scope == "Hele dataset / wegennet"
        control_passport_df = raw_gdf if network_scope else road_gdf
        control_selected_road = None if network_scope else selected_road
        control_label = "hele wegennet" if network_scope else selected_road
        control_file_suffix = "" if network_scope else f"_{sanitize_filename(selected_road)}"
        control_key_suffix = "netwerk" if network_scope else sanitize_filename(selected_road)

        if network_scope:
            st.info(
                "Netwerkbrede controle actief: de tool gebruikt de volledige actieve paspoortexport "
                "en groepeert de resultaten per wegnummer."
            )
        else:
            st.caption(f"Controle voor geselecteerde weg: {selected_road}")

        maintenance_payloads = tuple(st.session_state.get("maintenance_upload_payloads") or ())

        if not maintenance_payloads:
            st.info(
                "Upload eerst een onderhoudsexport in de zijbalk bij "
                "'Onderhoudsexport voor controle'."
            )
        else:
            maintenance_read = measure_step(
                get_performance_log(),
                "Onderhoudsexport lezen",
                cached_load_maintenance_exports,
                maintenance_payloads,
            )

            for warning in maintenance_read.warnings:
                st.warning(warning)

            previous_action_payloads = tuple(st.session_state.get("previous_action_list_payloads") or ())
            previous_action_list = pd.DataFrame()
            if previous_action_payloads:
                previous_action_list, previous_action_warnings = measure_step(
                    get_performance_log(),
                    "Eerdere actielijst lezen",
                    cached_load_previous_action_lists,
                    previous_action_payloads,
                )
                for warning in previous_action_warnings:
                    st.info(warning)

            if maintenance_read.dataframe.empty:
                st.error("De onderhoudsexport bevat geen herkenbare projectregels.")
            else:
                control_result = measure_step(
                    get_performance_log(),
                    "Onderhoudscontrole",
                    build_maintenance_control,
                    control_passport_df,
                    maintenance_read.dataframe,
                    control_selected_road,
                    previous_action_list,
                )

                for warning in control_result.warnings:
                    st.warning(warning)

                summary = control_result.summary

                data_quality_report = control_result.data_quality_report
                if data_quality_report is not None and not data_quality_report.empty:
                    quality_issues = summary.get("datakwaliteit_issues", 0)
                    q_blocking, q_warning, q_attention = st.columns(3)
                    q_blocking.metric("Datakwaliteit blokkerend", summary.get("datakwaliteit_blokkerend", 0))
                    q_warning.metric("Datakwaliteit waarschuwingen", summary.get("datakwaliteit_waarschuwingen", 0))
                    q_attention.metric("Datakwaliteit aandachtspunten", summary.get("datakwaliteit_aandachtspunten", 0))

                    if summary.get("datakwaliteit_blokkerend", 0):
                        st.error(
                            "De v0.23-voorcontrole ziet blokkerende exportproblemen. "
                            "De tabellen kunnen nog worden opgebouwd, maar conclusies zijn beperkt betrouwbaar."
                        )
                    elif quality_issues:
                        st.warning(
                            f"De v0.23-voorcontrole ziet {quality_issues} datakwaliteitsmelding(en). "
                            "Bekijk deze vóórdat je inhoudelijke conclusies trekt."
                        )
                    else:
                        st.success("Datakwaliteitsvoorcontrole: geen duidelijke exportproblemen gevonden.")

                    with st.expander("Datakwaliteitsrapport van de gebruikte exports", expanded=bool(quality_issues)):
                        st.caption(
                            "Deze v0.23-voorcontrole controleert de invoerbestanden op risico's zoals ontbrekende kolommen, "
                            "lege objectnummers, afwijkende projectnamen, ongeldige metrering en ontbrekende geometrie. "
                            "De tool wijzigt niets; dit rapport helpt bepalen hoe betrouwbaar de vervolgcontrole is."
                        )
                        st.dataframe(data_quality_report, use_container_width=True, hide_index=True)
                        quality_csv = data_quality_report.to_csv(index=False, sep=";").encode("utf-8-sig")
                        st.download_button(
                            "📥 Download datakwaliteitsrapport",
                            data=quality_csv,
                            file_name=f"Onderhoudscontrole_Datakwaliteit{control_file_suffix}.csv",
                            mime="text/csv",
                        )
                c_roads, c_ok, c_object, c_missing, c_total = st.columns(5)
                c_roads.metric("Wegen", summary.get("wegen_gecontroleerd", 0))
                c_ok.metric("OK volledig", summary.get("ok_volledig", summary.get("projecten_ok", 0)))
                c_object.metric("Objectverschil", summary.get("objectverschillen", 0) + summary.get("object_wegnummer_verdacht", 0))
                c_missing.metric("Ontbreekt/verweesd", summary.get("ontbreekt_in_onderhoud", 0) + summary.get("geen_paspoortobjecten", 0))
                c_total.metric("Totaal gecontroleerd", summary.get("projecten_totaal", 0))

                if summary.get("acties_met_overgenomen_beoordeling", 0):
                    st.success(
                        f"{summary.get('acties_met_overgenomen_beoordeling', 0)} eerdere beoordeling(en) "
                        "opnieuw meegenomen in de actielijst."
                    )

                if previous_action_payloads:
                    p_new, p_existing, p_resolved = st.columns(3)
                    p_new.metric("Nieuw sinds vorige controle", summary.get("controlepunten_nieuw", 0))
                    p_existing.metric("Bestaand", summary.get("controlepunten_bestaand", 0))
                    p_resolved.metric("Opgelost/niet meer gevonden", summary.get("controlepunten_opgelost", 0))

                    pg_new, pg_open, pg_resolved, pg_mixed = st.columns(4)
                    pg_new.metric("Nieuwe projectgroepen", summary.get("voortgang_nieuwe_projectgroepen", 0))
                    pg_open.metric("Blijft open", summary.get("voortgang_blijft_open_projectgroepen", 0))
                    pg_resolved.metric("Projectgroepen opgelost/niet gevonden", summary.get("voortgang_opgeloste_projectgroepen", 0))
                    pg_mixed.metric("Deels nieuw/deels bestaand", summary.get("voortgang_deels_nieuw_projectgroepen", 0))

                    if summary.get("controlepunten_opgelost", 0):
                        st.info(
                            "Er zijn controlepunten uit de vorige actielijst die niet meer terugkomen. "
                            "Controleer of ze echt zijn opgelost of buiten de nieuwe exportselectie vallen."
                        )

                    progress_report = control_result.progress_report
                    if progress_report is not None and not progress_report.empty:
                        with st.expander("Voortgangsrapport per weg en onderhoudsproject", expanded=True):
                            st.caption(
                                "Deze v0.24-laag vergelijkt de huidige controle met de vorige actielijst. "
                                "Zo zie je welke projectgroepen nieuw zijn, blijven terugkomen of mogelijk zijn opgelost. "
                                "Ook dit is alleen controle-informatie; de tool voert niets automatisch door."
                            )
                            st.dataframe(progress_report, use_container_width=True, hide_index=True)
                            progress_csv = progress_report.to_csv(index=False, sep=";").encode("utf-8-sig")
                            st.download_button(
                                "📥 Download voortgangsrapport",
                                data=progress_csv,
                                file_name=f"Onderhoudscontrole_Voortgangsrapport{control_file_suffix}.csv",
                                mime="text/csv",
                            )

                if summary.get("acties", 0):
                    st.warning(f"{summary.get('acties', 0)} controleactie(s) in de Onderhoudscontrole-actielijst.")

                if summary.get("mutatievoorstellen", 0):
                    st.info(
                        f"{summary.get('mutatievoorstellen', 0)} veilige mutatievoorstelregel(s) beschikbaar. "
                        "Deze voeren niets automatisch door; gebruik ze als controlelijst voor iASSET."
                    )

                if summary.get("hm_bereik_verdacht", 0):
                    st.info(
                        f"{summary.get('hm_bereik_verdacht', 0)} project(en) hebben een verdacht hm-bereik "
                        "door ongeldige metrering in de paspoortexport."
                    )

                comparison = control_result.comparison

                if comparison.empty:
                    st.info("Geen vergelijkingsregels beschikbaar.")
                else:
                    status_options = ["Alle statussen", *sorted(comparison["status"].dropna().astype(str).unique())]
                    selected_status = st.selectbox("Filter status", status_options)

                    visible_comparison = comparison
                    if selected_status != "Alle statussen":
                        visible_comparison = comparison[comparison["status"].astype(str) == selected_status]

                    action_list = control_result.action_list
                    edited_action_list = action_list
                    if action_list.empty:
                        st.success("Geen onderhoudscontrole-acties nodig: alle gecontroleerde projecten zijn volledig akkoord.")
                    else:
                        st.markdown("#### Onderhoudscontrole werkvoorraad")
                        st.caption(
                            "Deze werkvoorraad vertaalt de technische controles naar concrete controlewerkzaamheden. "
                            "Je kunt de opvolgvelden direct in de tabel invullen en daarna de bijgewerkte actielijst downloaden."
                        )

                        queue_summary = action_work_queue_summary(action_list)
                        q_total, q_open, q_new, q_done = st.columns(4)
                        q_total.metric("Controlepunten", queue_summary.get("controlepunten", 0))
                        q_open.metric("Open", queue_summary.get("open", 0))
                        q_new.metric("Nieuw", queue_summary.get("nieuw", 0))
                        q_done.metric("Afgehandeld", queue_summary.get("afgehandeld", 0))

                        q_warn, q_attention, q_investigate, q_fix = st.columns(4)
                        q_warn.metric("Waarschuwingen", queue_summary.get("waarschuwingen", 0))
                        q_attention.metric("Aandachtspunten", queue_summary.get("aandachtspunten", 0))
                        q_investigate.metric("In onderzoek", queue_summary.get("in_onderzoek", 0))
                        q_fix.metric("Te corrigeren", queue_summary.get("te_corrigeren", 0))

                        p_high, p_mid, p_low = st.columns(3)
                        p_high.metric("Prioriteit hoog", queue_summary.get("prioriteit_hoog", 0))
                        p_mid.metric("Prioriteit middel", queue_summary.get("prioriteit_middel", 0))
                        p_low.metric("Prioriteit laag", queue_summary.get("prioriteit_laag", 0))

                        project_summary = summarize_action_projects(action_list)
                        if not project_summary.empty:
                            with st.expander("Samenvatting per onderhoudsproject", expanded=True):
                                st.caption(
                                    "Deze v0.22-samenvatting groepeert de werkvoorraad per onderhoudsproject. "
                                    "Begin bij hoge prioriteit en bij projecten met primaire objecten."
                                )
                                st.dataframe(project_summary, use_container_width=True, hide_index=True)
                                project_summary_csv = project_summary.to_csv(index=False, sep=";").encode("utf-8-sig")
                                st.download_button(
                                    "📥 Download projectsamenvatting",
                                    data=project_summary_csv,
                                    file_name=f"Onderhoudscontrole_Projectsamenvatting{control_file_suffix}.csv",
                                    mime="text/csv",
                                )

                        possible_matches = int(
                            (
                                action_list.get("mogelijke_vervangende_projectnaam", pd.Series(dtype=str))
                                .fillna("")
                                .astype(str)
                                .str.strip()
                                .ne("")
                                | action_list.get("mogelijke_onderhoudsmatch", pd.Series(dtype=str))
                                .fillna("")
                                .astype(str)
                                .str.strip()
                                .ne("")
                            ).sum()
                        )
                        if possible_matches:
                            st.info(
                                f"{possible_matches} controlepunt(en) hebben een mogelijke vervangende projectnaam. "
                                "Gebruik dit als controlehint, niet als automatische correctie."
                            )

                        d_fault, d_border, d_export = st.columns(3)
                        d_fault.metric("Duiding: waarschijnlijke fout", summary.get("acties_waarschijnlijke_fout", 0))
                        d_border.metric("Duiding: grensgeval/twijfel", summary.get("acties_twijfel_of_grensgeval", 0))
                        d_export.metric("Duiding: export/naamkwestie", summary.get("acties_export_of_naamkwestie", 0))

                        extra_duiding = (
                            summary.get("acties_objectkoppeling_controleren", 0)
                            + summary.get("acties_verweesd_project_of_exportselectie", 0)
                        )
                        if extra_duiding:
                            st.caption(
                                f"Daarnaast {extra_duiding} controlepunt(en) met objectkoppeling/verweesd-project-duiding."
                            )

                        if network_scope and "wegnummer" in action_list.columns:
                            per_weg = (
                                action_list.groupby("wegnummer", dropna=False)
                                .size()
                                .reset_index(name="controlepunten")
                                .sort_values(["controlepunten", "wegnummer"], ascending=[False, True])
                            )
                            with st.expander("Controlepunten per weg", expanded=False):
                                st.dataframe(per_weg, use_container_width=True, hide_index=True)

                        selected_action_road = "Alle wegen"
                        if network_scope and "wegnummer" in action_list.columns:
                            road_options = [
                                value
                                for value in sorted(action_list["wegnummer"].fillna("").astype(str).unique())
                                if value.strip()
                            ]
                            selected_action_road = st.selectbox(
                                "Filter weg",
                                ["Alle wegen", *road_options],
                                key=f"maintenance_action_road_{control_key_suffix}",
                            )

                        filter_row_1 = st.columns(5)
                        with filter_row_1[0]:
                            priority_options = [
                                "Alle prioriteiten",
                                *[value for value in ("hoog", "middel", "laag") if value in set(action_list.get("prioriteit", pd.Series(dtype=str)).dropna().astype(str).str.lower())],
                            ]
                            selected_priority = st.selectbox(
                                "Filter prioriteit",
                                priority_options,
                                key=f"maintenance_priority_{control_key_suffix}",
                            )
                        with filter_row_1[1]:
                            severity_options = ["Alle ernstniveaus", *sorted(action_list["ernst"].dropna().astype(str).unique())]
                            selected_action_severity = st.selectbox(
                                "Filter ernst",
                                severity_options,
                                key=f"fase4_action_severity_{control_key_suffix}",
                            )
                        with filter_row_1[2]:
                            action_status_options = ["Alle technische statussen", *sorted(action_list["status"].dropna().astype(str).unique())]
                            selected_action_status = st.selectbox(
                                "Filter technische status",
                                action_status_options,
                                key=f"fase4_action_status_{control_key_suffix}",
                            )
                        with filter_row_1[3]:
                            practical_options = [
                                "Alle praktische categorieën",
                                *sorted(action_list["praktische_categorie"].dropna().astype(str).unique()),
                            ]
                            selected_practical_category = st.selectbox(
                                "Filter praktische categorie",
                                practical_options,
                                key=f"fase4_practical_category_{control_key_suffix}",
                            )
                        with filter_row_1[4]:
                            duiding_group_options = [
                                "Alle duidingsgroepen",
                                *sorted(action_list.get("duiding_groep", pd.Series(dtype=str)).dropna().astype(str).unique()),
                            ]
                            selected_duiding_group = st.selectbox(
                                "Filter duidingsgroep",
                                duiding_group_options,
                                key=f"fase4_duiding_group_{control_key_suffix}",
                            )

                        filter_row_2 = st.columns(4)
                        with filter_row_2[0]:
                            follow_up_options = [
                                "Alle afhandelstatussen",
                                *sorted(action_list["afhandelstatus"].dropna().astype(str).unique()),
                            ]
                            selected_follow_up_status = st.selectbox(
                                "Filter afhandelstatus",
                                follow_up_options,
                                key=f"fase4_followup_status_{control_key_suffix}",
                            )
                        with filter_row_2[1]:
                            progress_options = [
                                "Alle voortgangsstatussen",
                                *sorted(action_list.get("voortgang_status", pd.Series(dtype=str)).dropna().astype(str).unique()),
                            ]
                            selected_progress_status = st.selectbox(
                                "Filter voortgang",
                                progress_options,
                                key=f"maintenance_progress_status_{control_key_suffix}",
                            )
                        with filter_row_2[2]:
                            owner_values = [
                                value
                                for value in sorted(action_list["actiehouder"].fillna("").astype(str).unique())
                                if value.strip()
                            ]
                            selected_owner = st.selectbox(
                                "Filter actiehouder",
                                ["Alle actiehouders", *owner_values],
                                key=f"fase4_owner_{control_key_suffix}",
                            )
                        with filter_row_2[3]:
                            action_search = st.text_input(
                                "Zoek in werkvoorraad",
                                value="",
                                placeholder="Project, objectnummer, oorzaak, actie...",
                                key=f"fase4_action_search_{control_key_suffix}",
                            )

                        visible_action_list = filter_action_work_queue(
                            action_list,
                            ernst=selected_action_severity,
                            status=selected_action_status,
                            praktische_categorie=selected_practical_category,
                            duiding_groep=selected_duiding_group,
                            voortgang_status=selected_progress_status,
                            prioriteit=selected_priority,
                            afhandelstatus=selected_follow_up_status,
                            actiehouder=selected_owner,
                            zoektekst=action_search,
                            wegnummer=selected_action_road,
                        )

                        st.caption(
                            f"{len(visible_action_list)} van {len(action_list)} controlepunt(en) zichtbaar met deze filters."
                        )

                        if visible_action_list.empty:
                            st.info("Geen controlepunten gevonden met deze filterinstelling.")
                            edited_action_list = action_list
                        else:
                            disabled_columns = [
                                column
                                for column in visible_action_list.columns
                                if column not in ACTION_FOLLOW_UP_COLUMNS
                            ]
                            edited_visible_action_list = st.data_editor(
                                visible_action_list,
                                use_container_width=True,
                                hide_index=True,
                                num_rows="fixed",
                                disabled=disabled_columns,
                                column_config={
                                    "afhandelstatus": st.column_config.SelectboxColumn(
                                        "afhandelstatus",
                                        options=list(ACTION_FOLLOW_UP_STATUS_OPTIONS),
                                        help="Kies een gedeelde afhandelstatus voor dit controlepunt.",
                                    ),
                                    "beoordeling_databeheerder": st.column_config.TextColumn(
                                        "beoordeling_databeheerder",
                                        help="Korte inhoudelijke beoordeling van de databeheerder.",
                                    ),
                                    "actiehouder": st.column_config.TextColumn(
                                        "actiehouder",
                                        help="Naam of rol die de actie oppakt.",
                                    ),
                                    "opmerking_afhandeling": st.column_config.TextColumn(
                                        "opmerking_afhandeling",
                                        help="Vrije toelichting of vervolgafspraak.",
                                    ),
                                },
                                key=f"fase4_action_editor_{control_key_suffix}_{current_data_revision_key()}",
                            )
                            edited_action_list = merge_action_work_queue_edits(action_list, edited_visible_action_list)

                            detail_options = list(range(len(visible_action_list)))
                            selected_detail_index = st.selectbox(
                                "Detail controlepunt",
                                detail_options,
                                format_func=lambda idx: (
                                    f"{visible_action_list.iloc[idx].get('onderhoudsproject', '')} — "
                                    f"{visible_action_list.iloc[idx].get('controlecategorie', '')}"
                                ),
                                key=f"fase4_action_detail_{control_key_suffix}",
                            )
                            detail_row = visible_action_list.iloc[int(selected_detail_index)]
                            with st.expander("Leesbare toelichting bij geselecteerd controlepunt", expanded=False):
                                st.markdown(f"**Onderhoudsproject:** {clean_display_value(detail_row.get('onderhoudsproject', ''))}")
                                st.markdown(f"**Status:** {clean_display_value(detail_row.get('status', ''))}")
                                st.markdown(f"**Prioriteit:** {clean_display_value(detail_row.get('prioriteit', ''))} ({clean_display_value(detail_row.get('prioriteit_score', ''))})")
                                st.markdown(f"**Prioriteit uitleg:** {clean_display_value(detail_row.get('prioriteit_uitleg', ''))}")
                                st.markdown(f"**Projectsamenvatting:** {clean_display_value(detail_row.get('project_samenvatting', ''))}")
                                st.markdown(f"**Praktische categorie:** {clean_display_value(detail_row.get('praktische_categorie', ''))}")
                                st.markdown(f"**Duiding:** {clean_display_value(detail_row.get('duiding', ''))}")
                                st.markdown(f"**Duidingsgroep:** {clean_display_value(detail_row.get('duiding_groep', ''))}")
                                st.markdown(f"**Duiding uitleg:** {clean_display_value(detail_row.get('duiding_uitleg', ''))}")
                                st.markdown(f"**Voortgang:** {clean_display_value(detail_row.get('voortgang_status', ''))}")
                                st.markdown(f"**Voortgang uitleg:** {clean_display_value(detail_row.get('voortgang_uitleg', ''))}")
                                st.markdown(f"**Betrokken objecten:** {clean_display_value(detail_row.get('betrokken_objecten', ''))}")
                                if clean_display_value(detail_row.get("mogelijke_vervangende_projectnaam", "")):
                                    st.markdown(
                                        f"**Mogelijke vervangende projectnaam:** "
                                        f"{clean_display_value(detail_row.get('mogelijke_vervangende_projectnaam', ''))}"
                                    )
                                    st.markdown(f"**Matchscore:** {clean_display_value(detail_row.get('vervangende_projectnaam_score', ''))}")
                                    st.markdown(f"**Matchcriteria:** {clean_display_value(detail_row.get('vervangende_projectnaam_criteria', ''))}")
                                    st.markdown(f"**Waarom deze match?:** {clean_display_value(detail_row.get('vervangende_projectnaam_uitleg', ''))}")
                                elif clean_display_value(detail_row.get("mogelijke_onderhoudsmatch", "")):
                                    st.markdown(f"**Mogelijke onderhoudsmatch:** {clean_display_value(detail_row.get('mogelijke_onderhoudsmatch', ''))}")
                                    st.markdown(f"**Waarom deze match?:** {clean_display_value(detail_row.get('onderhoudsmatch_uitleg', ''))}")
                                st.markdown(f"**Uitleg:** {clean_display_value(detail_row.get('uitleg', ''))}")
                                st.markdown(f"**Mogelijke oorzaak:** {clean_display_value(detail_row.get('mogelijke_oorzaak', ''))}")
                                st.markdown(f"**Voorgestelde actie:** {clean_display_value(detail_row.get('voorgestelde_actie', ''))}")

                            with st.expander("Objectdetails en kaart voor geselecteerd controlepunt", expanded=False):
                                detail_objects = build_control_point_object_details(
                                    control_passport_df,
                                    maintenance_read.dataframe,
                                    detail_row,
                                    control_result.object_differences,
                                    selected_road=control_selected_road,
                                )
                                if detail_objects.empty:
                                    st.info("Geen objectdetails beschikbaar voor dit controlepunt.")
                                else:
                                    st.caption(
                                        "Deze tabel toont de betrokken objecten uit paspoort- en onderhoudsexport. "
                                        "Objecten zonder paspoortgeometrie kunnen niet op de kaart worden getekend."
                                    )
                                    st.dataframe(detail_objects, use_container_width=True, hide_index=True)

                                    detail_csv = detail_objects.to_csv(index=False, sep=";").encode("utf-8-sig")
                                    st.download_button(
                                        "📥 Download objectdetails controlepunt",
                                        data=detail_csv,
                                        file_name=(
                                            "Onderhoudscontrole_Objectdetails_"
                                            f"{sanitize_filename(clean_display_value(detail_row.get('onderhoudsproject', 'controlepunt')))}.csv"
                                        ),
                                        mime="text/csv",
                                    )

                                    map_result = build_maintenance_control_map(control_passport_df, detail_objects, detail_row)
                                    if map_result.folium_map is None:
                                        st.info(map_result.message or "Geen kaart beschikbaar voor deze objectselectie.")
                                    else:
                                        st.caption(
                                            f"{map_result.mapped_object_count} object(en) op kaart getoond "
                                            f"({map_result.primary_object_count} primair, "
                                            f"{map_result.secondary_object_count} secundair, "
                                            f"{map_result.exempt_object_count} uitgezonderd). "
                                            f"{map_result.missing_passport_object_count} object(en) hebben geen paspoortgeometrie in deze export."
                                        )
                                        if map_result.difference_type_counts:
                                            st.caption(
                                                "Verschiltypen op kaart: "
                                                + ", ".join(
                                                    f"{clean_display_value(key)}: {value}"
                                                    for key, value in map_result.difference_type_counts.items()
                                                )
                                            )
                                        st_folium(
                                            map_result.folium_map,
                                            width="100%",
                                            height=450,
                                            key=f"maintenance_detail_map_{control_key_suffix}_{selected_detail_index}_{current_data_revision_key()}",
                                        )

                                        map_html = map_result.folium_map.get_root().render().encode("utf-8")
                                        st.download_button(
                                            "📥 Download kaart controlepunt (HTML)",
                                            data=map_html,
                                            file_name=(
                                                "Onderhoudscontrole_Kaart_"
                                                f"{sanitize_filename(clean_display_value(detail_row.get('onderhoudsproject', 'controlepunt')))}.html"
                                            ),
                                            mime="text/html",
                                            help=(
                                                "Exporteert de kaart als los HTML-controlebeeld. "
                                                "Dit is alleen documentatie; de tool wijzigt niets in iASSET."
                                            ),
                                        )

                        action_csv = edited_action_list.to_csv(index=False, sep=";").encode("utf-8-sig")
                        st.download_button(
                            "📥 Download bijgewerkte Onderhoudscontrole actielijst",
                            data=action_csv,
                            file_name=f"Onderhoudscontrole_Actielijst{control_file_suffix}.csv",
                            mime="text/csv",
                        )

                        if not control_result.resolved_actions.empty:
                            resolved_csv = control_result.resolved_actions.to_csv(index=False, sep=";").encode("utf-8-sig")
                            st.download_button(
                                "📥 Download opgeloste/niet meer gevonden controlepunten",
                                data=resolved_csv,
                                file_name=f"Onderhoudscontrole_Opgelost{control_file_suffix}.csv",
                                mime="text/csv",
                                help=(
                                    "Controlepunten uit de vorige actielijst die niet meer terugkomen in de nieuwe controle. "
                                    "Controleer of ze echt zijn opgelost of buiten de nieuwe exportselectie vallen."
                                ),
                            )

                    mutation_suggestions = control_result.mutation_suggestions
                    if mutation_suggestions.empty:
                        st.success("Geen mutatievoorstellen nodig op basis van de huidige onderhoudscontrole.")
                    else:
                        st.markdown("#### Veilige mutatievoorstellen")
                        st.warning(
                            "Veiligheidsregel: de tool voert nooit automatisch aanpassingen door. "
                            "Gebruik deze tabel alleen als controlelijst; de databeheerder beslist en verwerkt handmatig."
                        )
                        st.caption(
                            "Deze tabel vertaalt controlepunten naar mogelijke correctieregels. "
                            "Elke regel is een conceptvoorstel met menselijke controle verplicht."
                        )
                        st.dataframe(mutation_suggestions, use_container_width=True, hide_index=True)
                        mutation_suggestions_csv = mutation_suggestions.to_csv(index=False, sep=";").encode("utf-8-sig")
                        st.download_button(
                            "📥 Download Onderhoudscontrole mutatievoorstellen",
                            data=mutation_suggestions_csv,
                            file_name=f"Onderhoudscontrole_Mutatievoorstellen{control_file_suffix}.csv",
                            mime="text/csv",
                        )

                    control_package_bytes = build_maintenance_control_workbook(
                        control_result,
                        action_list=edited_action_list,
                        scope_label=control_label,
                    )
                    st.download_button(
                        "📦 Download Onderhoudscontrole controlepakket (Excel)",
                        data=control_package_bytes,
                        file_name=f"Onderhoudscontrole_Controlepakket{control_file_suffix}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help=(
                            "Excelbestand met samenvatting, werkvoorraad, mutatievoorstellen, "
                            "resultaten en objectverschillen. Dit is alleen een controlepakket; "
                            "de tool voert niets automatisch door."
                        ),
                    )

                    st.markdown("#### Vergelijking paspoortexport ↔ onderhoudsexport")
                    st.dataframe(visible_comparison, use_container_width=True, hide_index=True)

                    comparison_csv = visible_comparison.to_csv(index=False, sep=";").encode("utf-8-sig")
                    st.download_button(
                        "📥 Download zichtbare onderhoudscontrole",
                        data=comparison_csv,
                        file_name=f"Onderhoudscontrole_Resultaten{control_file_suffix}.csv",
                        mime="text/csv",
                    )

                    c_dl_1, c_dl_2 = st.columns(2)
                    with c_dl_1:
                        passport_csv = control_result.passport_projects.to_csv(index=False, sep=";").encode("utf-8-sig")
                        st.download_button(
                            "📥 Download samenvatting paspoortprojecten",
                            data=passport_csv,
                            file_name=f"Onderhoudscontrole_Paspoortprojecten{control_file_suffix}.csv",
                            mime="text/csv",
                        )

                    with c_dl_2:
                        maintenance_csv = control_result.maintenance_projects.to_csv(index=False, sep=";").encode("utf-8-sig")
                        st.download_button(
                            "📥 Download samenvatting onderhoudsexport",
                            data=maintenance_csv,
                            file_name=f"Onderhoudscontrole_Onderhoudsexport{control_file_suffix}.csv",
                            mime="text/csv",
                        )

                    object_differences = control_result.object_differences
                    if object_differences.empty:
                        st.success("Geen objectverschillen gevonden tussen paspoortexport en onderhoudsexport.")
                    else:
                        st.markdown("#### Objectverschillen")
                        st.dataframe(object_differences, use_container_width=True, hide_index=True)
                        object_diff_csv = object_differences.to_csv(index=False, sep=";").encode("utf-8-sig")
                        st.download_button(
                            "📥 Download objectverschillen",
                            data=object_diff_csv,
                            file_name=f"Onderhoudscontrole_Objectverschillen{control_file_suffix}.csv",
                            mime="text/csv",
                        )


    elif mode == "🧾 Objectinspecteur":
        st.markdown("### 🧾 Objectinspecteur")
        st.caption(
            "Selecteer één object, controleer de paspoortdata en pas alleen velden aan "
            "die binnen een gekozen iASSET-exportprofiel passen."
        )

        search_query = st.text_input(
            "Zoek object",
            value="",
            placeholder="Zoek op objectnummer, naam, onderhoudsproject, metrering, besteknummer...",
            key=f"object_search_{selected_road}",
        )

        changed_only = st.checkbox(
            "Toon alleen objecten met open wijzigingen",
            value=False,
            help="Handig om snel terug te vinden welke objecten al in het wijzigingslogboek staan.",
        )

        object_results = search_objects(
            road_gdf,
            search_query,
            changed_only=changed_only,
            change_log=st.session_state.get("change_log", []),
            max_results=300,
        )

        if not object_results:
            st.info("Geen objecten gevonden met deze zoek/filterinstelling.")
            st.session_state["selected_object_id"] = None
        else:
            current_selected_id = st.session_state.get("selected_object_id")
            default_index = 0
            for result_index, result in enumerate(object_results):
                if result.object_id == current_selected_id:
                    default_index = result_index
                    break

            selected_result = st.selectbox(
                "Object",
                object_results,
                index=default_index,
                format_func=lambda result: result.label,
                help="De lijst toont maximaal 300 resultaten. Gebruik de zoektekst om te verfijnen.",
            )

            selected_object_id = selected_result.object_id
            st.session_state["selected_object_id"] = selected_object_id
            st.session_state["selected_error_id"] = None
            st.session_state["selected_group_id"] = None

            selected_row = raw_gdf.loc[selected_object_id]

            c_zoom, c_clear = st.columns([1, 1])
            with c_zoom:
                if st.button("👁️ Toon op kaart", key=f"zoom_object_{selected_object_id}"):
                    geom_web = raw_gdf.loc[[selected_object_id]].to_crs(epsg=4326).geometry.iloc[0]
                    st.session_state["zoom_bounds"] = geom_web.bounds
                    st.rerun()

            with c_clear:
                if st.button("Zoom resetten", key=f"reset_object_zoom_{selected_object_id}"):
                    st.session_state["zoom_bounds"] = None
                    st.rerun()

            st.divider()

            st.markdown("#### Huidige paspoortkern")
            preview_fields = [
                "nummer",
                "subthema",
                "naam",
                "Wegnummer",
                "Wegvaknum",
                "Metrering",
                "Situering",
                "verhardingssoort",
                "Soort deklaag specifiek",
                "Jaar aanleg",
                "Jaar deklaag",
                "Besteknummer",
                "Onderhoudsproject",
            ]
            st.dataframe(
                object_preview_dataframe(raw_gdf, selected_object_id, preview_fields),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Paspoortdata aanpassen")
            st.caption(
                "Geometrie en bron-id's zijn bewust niet bewerkbaar. De app bereidt "
                "paspoortmutaties voor; iASSET blijft de bronregistratie."
            )

            edit_profiles = available_export_profiles()
            default_edit_profile_index = (
                edit_profiles.index(DEFAULT_EXPORT_PROFILE)
                if DEFAULT_EXPORT_PROFILE in edit_profiles
                else 0
            )
            selected_edit_profile = st.selectbox(
                "Mutatie-/exportprofiel voor dit formulier",
                edit_profiles,
                index=default_edit_profile_index,
                key="object_editor_profile",
                help=(
                    "De aangeboden velden volgen het gekozen exportprofiel. "
                    "Daarmee blijft duidelijk welke kolommenset later samen naar iASSET gaat."
                ),
            )

            show_extra_available_fields = st.checkbox(
                "Toon ook andere aanwezige belangrijke paspoort-/liggingvelden",
                value=False,
                help=(
                    "Gebruik dit vooral voor inspectie of voorbereiding. Voor iASSET-import "
                    "blijft het gekozen exportprofiel onderaan bij de export leidend."
                ),
            )

            editable_fields = editable_fields_for_profile(
                raw_gdf,
                selected_edit_profile,
                include_location_fallback=show_extra_available_fields,
            )
            unavailable_profile_columns = missing_profile_columns(raw_gdf, selected_edit_profile)

            if unavailable_profile_columns:
                st.warning(
                    "Deze kolommen uit het gekozen profiel ontbreken in de actieve export en "
                    "kunnen daarom niet worden aangepast of geëxporteerd: "
                    + ", ".join(unavailable_profile_columns),
                    icon="⚠️",
                )

            if not editable_fields:
                st.info("Geen bewerkbare velden beschikbaar binnen dit profiel voor deze dataset.")
            else:
                with st.form(key=f"object_edit_form_{selected_object_id}_{selected_edit_profile}"):
                    input_values = {}

                    for field in editable_fields:
                        current_value = clean_display_value(selected_row.get(field, ""))
                        input_values[field] = st.text_input(
                            field,
                            value=current_value,
                            key=f"object_edit_{selected_object_id}_{selected_edit_profile}_{field}",
                        )

                    submitted = st.form_submit_button("💾 Wijzigingen opslaan")

                if submitted:
                    saved_count = 0

                    for field, new_value in input_values.items():
                        old_value = raw_gdf.at[selected_object_id, field] if field in raw_gdf.columns else ""
                        if clean_display_value(old_value) == clean_display_value(new_value):
                            continue

                        register_change(selected_object_id, field, old_value, new_value)
                        saved_count += 1

                    if saved_count:
                        st.success(f"{saved_count} veld(en) opgeslagen in het wijzigingslogboek.")
                    else:
                        st.info("Geen gewijzigde waarden gevonden.")

                    st.rerun()

            with st.expander("Ruwe beschikbare objectdata", expanded=False):
                object_columns = [
                    column
                    for column in raw_gdf.columns
                    if column != "geometry" and not str(column).startswith("_")
                ]
                st.dataframe(
                    object_preview_dataframe(raw_gdf, selected_object_id, object_columns),
                    use_container_width=True,
                    hide_index=True,
                )


    elif mode == "🗺️ Overzicht":
        st.markdown("### 🗺️ Overzicht")
        st.caption(
            "Alleen-lezen visualisatie van rijstroken. In dit tabblad worden geen iASSET-waarden aangepast."
        )

        overview_scope = st.radio(
            "Kaartbereik:",
            ["Geselecteerde weg", "Alle wegen"],
            horizontal=True,
            key="overview_scope",
        )

        overview_gdf = raw_gdf if overview_scope == "Alle wegen" else road_gdf
        overview_label = "alle wegen" if overview_scope == "Alle wegen" else selected_road

        overview_attributes = available_overview_attributes(overview_gdf)

        if not overview_attributes:
            st.warning(f"Geen bruikbare visualisatiekolommen gevonden voor {overview_label}.")
        else:
            default_index = 0
            if "Jaar deklaag" in overview_attributes:
                default_index = overview_attributes.index("Jaar deklaag")

            overview_attribute = st.selectbox(
                "Visualiseer op:",
                overview_attributes,
                index=default_index,
                key=f"overview_attr_{sanitize_filename(overview_scope)}_{selected_road}",
            )

            rijstrook_count = 0
            if "subthema_clean" in overview_gdf.columns:
                rijstrook_count = int((overview_gdf["subthema_clean"].astype(str).str.lower().str.strip() == "rijstrook").sum())

            st.info(
                f"Deze kaart toont alleen rijstroken. Bereik: {overview_label}. "
                f"Aantal rijstrookobjecten: {rijstrook_count}."
            )
            st.caption(
                "De legenda staat linksonder in de kaart. Klik op een object voor de beschikbare paspoortdata. "
                "De HTML-export gebruikt hetzelfde bereik en hetzelfde attribuut."
            )


# --- Linkerkolom: kaart ----------------------------------------------------

with col_map:
    if mode == "🗺️ Overzicht":
        st.subheader(f"Overzicht: {overview_label}")

        if overview_gdf.empty:
            st.warning(f"Geen data gevonden voor {overview_label}.")
            st.stop()

        if overview_attribute is None:
            st.warning("Kies eerst een attribuut om te visualiseren.")
        else:
            overview_map_result = measure_step(
                get_performance_log(),
                "Overzichtkaart opbouwen",
                build_overview_map,
                overview_gdf,
                overview_attribute,
            )

            if overview_map_result.row_count == 0:
                st.warning(f"Geen rijstrookobjecten gevonden voor {overview_label}.")
            elif overview_map_result.selected_column is None:
                st.warning(f"Attribuut '{overview_attribute}' is niet beschikbaar in de data.")
            else:
                st.caption(
                    f"{overview_map_result.row_count} rijstrookobjecten gevisualiseerd op "
                    f"`{overview_attribute}` via kolom `{overview_map_result.selected_column}`. "
                    f"Legenda-items: {len(overview_map_result.legend_items)}."
                )

                export_title = f"iASSET Overzicht - {overview_label}"
                export_subtitle = (
                    f"Visualisatie: {overview_attribute} | "
                    f"Rijstrookobjecten: {overview_map_result.row_count}"
                )
                export_html = render_overview_map_html(
                    overview_map_result,
                    title=export_title,
                    subtitle=export_subtitle,
                )

                file_scope = "alle_wegen" if overview_scope == "Alle wegen" else selected_road
                file_attr = sanitize_filename(overview_attribute)
                st.download_button(
                    "⬇️ Download Overzichtkaart als HTML",
                    data=export_html.encode("utf-8"),
                    file_name=f"iASSET_Overzicht_{sanitize_filename(file_scope)}_{file_attr}.html",
                    mime="text/html",
                    help="Exporteert de huidige Overzicht-instelling als interactieve Leaflet/Folium-kaart.",
                )

            st_folium(
                overview_map_result.folium_map,
                width=None,
                height=720,
                returned_objects=[],
                key=f"overview_map_{sanitize_filename(overview_scope)}_{selected_road}_{overview_attribute}",
            )

    else:
        st.subheader(f"Kaart: {selected_road}")

        if road_gdf.empty:
            st.warning("Geen data gevonden voor deze weg.")
            st.stop()

        st.markdown("### 🛠️ Weergave opties")
        show_network = st.toggle("🕸️ Toon Netwerk (Lijnen & Bollen)", value=False)
        show_pdok = st.toggle(
            "📍 Toon hectometerpunten (PDOK)",
            value=False,
            help=(
                "PDOK is alleen visuele ondersteuning en kan traag zijn. "
                "De punten worden pas opgehaald wanneer deze optie aan staat."
            ),
        )

        current_violations = get_quality_issues_for_road(selected_road, road_gdf, graph_road)
        error_ids = {
            violation["id"]
            for violation in current_violations
            if not is_violation_ignored(violation)
        }

        pdok_hm = get_pdok_hectopunten_cached(selected_road, road_gdf) if show_pdok else None

        map_result = measure_step(
            get_performance_log(),
            "Kaart opbouwen",
            build_road_map,
            road_gdf,
            graph_road,
            zoom_bounds=st.session_state.get("zoom_bounds"),
            selected_error_id=st.session_state.get("selected_error_id"),
            selected_group_id=st.session_state.get("selected_group_id"),
            selected_object_id=st.session_state.get("selected_object_id"),
            computed_groups=st.session_state.get("computed_groups"),
            processed_groups=st.session_state.get("processed_groups"),
            ignored_groups=st.session_state.get("ignored_groups"),
            error_ids=error_ids,
            show_network=show_network,
            pdok_hm=pdok_hm,
        )

        if show_network:
            st.caption(
                f"Netwerk actief: {map_result.network_node_count} bollen "
                f"en {map_result.network_edge_count} lijnen."
            )

        st_folium(
            map_result.folium_map,
            width=None,
            height=600,
            returned_objects=["last_object_clicked"],
            key="folium_map",
        )

        st.divider()
        st.markdown("### 🕵️ Sorteerdiagnose")

        computed_groups = st.session_state.get("computed_groups") or {}
        if not computed_groups:
            st.caption("Project Adviseur is nog niet berekend. Open Project Adviseur om de groepsdiagnose te vullen.")
        else:
            active_diagnostic_groups = {
                group_id: group_data
                for group_id, group_data in computed_groups.items()
                if group_id not in st.session_state["processed_groups"]
            }

            with st.expander("Diagnose huidige projectvolgorde", expanded=False):
                st.caption(
                    "Deze diagnose verandert de Project Adviseur nog niet. Hij laat zien of de huidige "
                    "volgorde vooral op metrering steunt, of dat binnen hetzelfde wegvak/metrering "
                    "een extra controle nodig is."
                )

                if st.button("Bereken sorteerdiagnose", key=f"build_sort_diag_{selected_road}"):
                    object_diag, group_diag, axis_result = measure_step(
                        get_performance_log(),
                        "Sorteerdiagnose",
                        build_sort_diagnostics,
                        road_gdf,
                        active_diagnostic_groups,
                        selected_road=selected_road,
                    )
                    st.session_state["sort_diagnostics"] = {
                        "road": selected_road,
                        "revision": current_data_revision_key(),
                        "schema_version": SORT_DIAGNOSTICS_SCHEMA_VERSION,
                        "app_version": APP_VERSION,
                        "objects": object_diag,
                        "groups": group_diag,
                        "axis_source": axis_result.source,
                        "axis_anchor_count": axis_result.anchor_count,
                        "axis_warning": axis_result.warning,
                    }

                sort_diag = st.session_state.get("sort_diagnostics") or {}
                if (
                    sort_diag.get("road") != selected_road
                    or sort_diag.get("revision") != current_data_revision_key()
                    or sort_diag.get("schema_version") != SORT_DIAGNOSTICS_SCHEMA_VERSION
                ):
                    st.info(
                        "Klik op 'Bereken sorteerdiagnose' om de diagnose voor deze weg, "
                        "datasetrevisie en appversie te maken."
                    )
                else:
                    axis_warning = sort_diag.get("axis_warning", "")
                    st.caption(
                        f"Diagnose: {sort_diag.get('app_version', APP_VERSION)} / "
                        f"{sort_diag.get('schema_version', 'onbekend')}. "
                        f"Lokale route-as: {sort_diag.get('axis_source', 'onbekend')} "
                        f"met {sort_diag.get('axis_anchor_count', 0)} ankerpunten."
                    )
                    if axis_warning:
                        st.warning(axis_warning, icon="⚠️")

                    group_diag = sort_diag.get("groups")
                    if isinstance(group_diag, pd.DataFrame) and not group_diag.empty:
                        st.markdown("#### Groepsdiagnose")
                        st.dataframe(group_diag, use_container_width=True, hide_index=True)

                        group_csv = group_diag.to_csv(index=False, sep=";").encode("utf-8-sig")
                        st.download_button(
                            "📥 Download groepsdiagnose",
                            data=group_csv,
                            file_name=f"Sorteerdiagnose_Groepen_{sanitize_filename(selected_road)}.csv",
                            mime="text/csv",
                        )
                    else:
                        st.info("Geen groepsdiagnose beschikbaar.")

                    show_object_diag = st.checkbox(
                        "Toon objectdiagnose",
                        value=False,
                        help=(
                            "Objectniveau is vooral nuttig om dubbele objecten binnen hetzelfde "
                            "wegvak/metrering te controleren."
                        ),
                    )

                    object_diag = sort_diag.get("objects")
                    if show_object_diag and isinstance(object_diag, pd.DataFrame) and not object_diag.empty:
                        attention_filter = st.selectbox(
                            "Objectdiagnose-filter",
                            [
                                "Alle aandachtspunten",
                                "Alleen waarschuwingen",
                                "Alleen info",
                                "Alle objecten",
                            ],
                            index=0,
                            key=f"sort_diag_attention_filter_{selected_road}",
                            help=(
                                "Aandachtspunten zijn regels met een INFO of WAARSCHUWING. "
                                "Gebruik 'Alleen waarschuwingen' voor echte risico's."
                            ),
                        )

                        display_object_diag = object_diag
                        if "sort_severity" in display_object_diag.columns:
                            severity_series = display_object_diag["sort_severity"].astype(str).str.strip().str.lower()
                            if attention_filter == "Alle aandachtspunten":
                                display_object_diag = display_object_diag[severity_series != ""]
                            elif attention_filter == "Alleen waarschuwingen":
                                display_object_diag = display_object_diag[severity_series == "waarschuwing"]
                            elif attention_filter == "Alleen info":
                                display_object_diag = display_object_diag[severity_series == "info"]
                        elif attention_filter != "Alle objecten" and "sort_warning" in display_object_diag.columns:
                            # Fallback voor oude diagnoseframes zonder sort_severity.
                            warning_text = display_object_diag["sort_warning"].astype(str).str.strip()
                            display_object_diag = display_object_diag[warning_text != ""]

                        st.markdown("#### Objectdiagnose")
                        st.dataframe(display_object_diag, use_container_width=True, hide_index=True)

                        # Download exact dezelfde regels als zichtbaar zijn.
                        object_csv = display_object_diag.to_csv(index=False, sep=";").encode("utf-8-sig")
                        suffix_map = {
                            "Alle aandachtspunten": "_Aandachtspunten",
                            "Alleen waarschuwingen": "_Waarschuwingen",
                            "Alleen info": "_Info",
                            "Alle objecten": "",
                        }
                        export_suffix = suffix_map.get(attention_filter, "")
                        st.download_button(
                            "📥 Download zichtbare objectdiagnose",
                            data=object_csv,
                            file_name=(
                                f"Sorteerdiagnose_Objecten_{sanitize_filename(selected_road)}"
                                f"{export_suffix}.csv"
                            ),
                            mime="text/csv",
                        )


# --- Logboek en export -----------------------------------------------------

st.divider()
st.subheader("📝 Logboek Wijzigingen & Export")

if st.session_state["change_log"]:
    c_all_1, c_all_2 = st.columns([1, 5])

    with c_all_1:
        if st.button("⚠️ Alles Herstellen", type="primary", help="Draai alle wijzigingen in één keer terug"):
            for entry in reversed(st.session_state["change_log"]):
                object_id = entry["ID"]
                field = entry["Veld"]

                apply_change_to_data(raw_gdf, object_id, field, entry["Oud"])

                if field == "Onderhoudsproject":
                    restore_group_for_object(object_id)

            st.session_state["change_log"] = []
            persist_change_log()
            st.success("Alle wijzigingen zijn ongedaan gemaakt.")
            st.rerun()

    with c_all_2:
        st.caption(f"Er staan {len(st.session_state['change_log'])} wijzigingen in de wachtrij.")

    st.divider()

    reversed_log = list(reversed(list(enumerate(st.session_state["change_log"]))))

    with st.container(height=300):
        for index, entry in reversed_log:
            c_time, c_id, c_change, c_undo = st.columns([1, 2, 4, 1])

            c_time.text(entry["Tijd"])
            c_id.text(f"ID: {entry['ID']}")
            c_change.text(f"{entry['Veld']}: {entry['Oud']} ➡ {entry['Nieuw']}")

            if c_undo.button("↩️ Herstel", key=f"undo_{index}"):
                apply_change_to_data(raw_gdf, entry["ID"], entry["Veld"], entry["Oud"])

                if entry["Veld"] == "Onderhoudsproject":
                    restore_group_for_object(entry["ID"])

                del st.session_state["change_log"][index]
                persist_change_log()
                st.success("Wijziging ongedaan gemaakt.")
                st.rerun()
else:
    st.caption("Nog geen wijzigingen aangebracht.")


changed_ids = collect_changed_ids(st.session_state["change_log"])

if changed_ids:
    profiles = available_export_profiles()
    default_profile_index = profiles.index(DEFAULT_EXPORT_PROFILE) if DEFAULT_EXPORT_PROFILE in profiles else 0

    selected_export_profile = st.selectbox(
        "iASSET-exportprofiel",
        profiles,
        index=default_profile_index,
        help=(
            "Kies welke kolommenset voor alle gewijzigde objecten wordt meegeschreven. "
            "iASSET verwerkt per importbestand één vaste kolommenset."
        ),
    )

    export_columns = get_export_profile_columns(selected_export_profile)
    export_summary = summarize_export_profile(raw_gdf, st.session_state["change_log"], selected_export_profile)
    df_export = build_export_dataframe(raw_gdf, changed_ids, export_columns=export_columns)

    st.success(f"📦 Er staan {len(df_export)} gewijzigde objecten klaar voor export.")

    c_obj, c_cell, c_written, c_unchanged = st.columns(4)
    c_obj.metric("Gewijzigde objecten", export_summary.changed_object_count)
    c_cell.metric("Gewijzigde cellen", export_summary.changed_cell_count)
    c_written.metric("Meegeschreven waarden", export_summary.written_value_count)
    c_unchanged.metric("Waarvan ongewijzigd", export_summary.unchanged_written_value_count)

    st.warning(
        "Let op: alle beschikbare kolommen in het gekozen exportprofiel worden voor alle "
        "gewijzigde objecten meegeschreven. Gebruik daarom bij voorkeur een actuele "
        "iASSET-export als basis voor mutaties.",
        icon="⚠️",
    )

    if export_summary.omitted_changed_fields:
        st.warning(
            "De volgende gewijzigde velden zitten niet in dit exportprofiel en komen dus niet mee "
            "in deze export: "
            + ", ".join(export_summary.omitted_changed_fields),
            icon="⚠️",
        )

    with st.expander("Kolommen in dit exportprofiel", expanded=False):
        if export_summary.valid_export_columns:
            st.write(", ".join(export_summary.valid_export_columns))
        else:
            st.info("Geen beschikbare kolommen gevonden voor dit exportprofiel in de actieve dataset.")

    c_dl1, c_dl2 = st.columns(2)
    file_profile = sanitize_filename(selected_export_profile)

    with c_dl1:
        csv = df_export.to_csv(index=False, sep=";").encode("utf-8-sig")
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"iASSET_Mutaties_{file_profile}.csv",
            mime="text/csv",
        )

    with c_dl2:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_export.to_excel(writer, index=False, sheet_name="Verhardingen")

        st.download_button(
            label="📊 Download Excel (.xlsx)",
            data=buffer.getvalue(),
            file_name=f"iASSET_Mutaties_{file_profile}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
else:
    st.info("Er zijn nog geen wijzigingen aangebracht. Voer eerst wijzigingen door om te kunnen exporteren.")
