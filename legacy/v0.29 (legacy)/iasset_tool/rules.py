"""
Datakwaliteitsregels.

Deze module bevat alleen de controles. De UI bepaalt daarna hoe meldingen
getoond, genegeerd of gecorrigeerd worden.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import networkx as nx

from .config import BACKBONE_TYPES
from .domain import (
    is_maintenance_project_exempt,
    is_project_value_empty,
    maintenance_project_exemption_reason,
)
from .utils import clean_display_value, is_empty_value, normalize_text, parse_hm_sort


CATEGORY_MISSING_PROJECT = "Onderhoudsprojectplicht"
CATEGORY_WRONG_PROJECT = "Onterecht onderhoudsproject"
CATEGORY_TOPOLOGY = "Topologie"
CATEGORY_PROJECT_CONSISTENCY = "Projectconsistentie"
CATEGORY_LOCATION_DATA = "Liggingdata"


def _clean_list(values: list[str]) -> set[str]:
    """Maak een set met lowercase domeinwaarden."""
    return {normalize_text(value) for value in values}


def _exemption_mask(gdf: gpd.GeoDataFrame):
    """
    Bepaal per rij of het object is uitgezonderd van onderhoudsprojectplicht.

    We doen dit rijgewijs omdat de uitzondering "oorspronkelijke BGT-data" niet
    altijd in dezelfde kolom staat.
    """
    return gdf.apply(is_maintenance_project_exempt, axis=1)


def make_violation(
    *,
    severity: str,
    category: str,
    rule_code: str,
    object_id: Any,
    subthema: Any,
    message: str,
    missing_cols: list[str] | None = None,
) -> dict[str, Any]:
    """
    Maak één gestandaardiseerde datakwaliteitsmelding.

    Waarom deze helper?
    De UI, export en tests kunnen dan vertrouwen op dezelfde velden
    (`severity`, `category`, `rule_code` en `issue_key`). Het oude veld `type`
    blijft bestaan voor achterwaartse compatibiliteit met bestaande code.
    """
    return {
        "type": severity,
        "severity": severity,
        "category": category,
        "rule_code": rule_code,
        "issue_key": f"{rule_code}:{object_id}",
        "id": object_id,
        "subthema": subthema,
        "msg": message,
        "missing_cols": missing_cols or [],
    }


def violation_key(violation: dict[str, Any]) -> str:
    """
    Geef een stabiele sleutel voor negeren/selecteren van een issue.

    In oude sessies kan `ignored_errors` nog object-id's bevatten. De UI houdt
    daar rekening mee; deze functie levert voor nieuwe meldingen altijd een
    rule-code + object-id op zodat meerdere issues op één object apart kunnen
    bestaan.
    """
    issue_key = violation.get("issue_key")
    if issue_key:
        return str(issue_key)

    return f"{violation.get('rule_code', 'ISSUE')}:{violation.get('id', '')}"


def category_counts(violations: list[dict[str, Any]]) -> dict[str, int]:
    """Tel het aantal meldingen per issuecategorie."""
    counts: dict[str, int] = {}

    for violation in violations:
        category = str(violation.get("category") or "Overig")
        counts[category] = counts.get(category, 0) + 1

    return counts


def check_rules(gdf: gpd.GeoDataFrame, graph: nx.Graph | None = None) -> list[dict[str, Any]]:
    """
    Controleer objecten op datakwaliteitsissues.

    Regels:
    1. onderhoudsprojectplichtige objecten zonder Onderhoudsproject;
    2. uitgezonderde objecten die tóch een Onderhoudsproject hebben;
    3. secundaire objecten die los zweven van hoofdobjecten;
    4. objecten met projectnaam die geïsoleerd liggen t.o.v. dat project.

    De uitzonderingslogica volgt het werkproces Grijs: bepaalde subthema's en
    objecten met oorspronkelijke BGT-data krijgen geen onderhoudsproject.
    Iedere melding krijgt vanaf v0.9 ook een categorie en rule-code, zodat de
    werklijst niet meer één grote ongesorteerde hoop wordt.
    """
    violations: list[dict[str, Any]] = []

    if gdf is None or gdf.empty:
        return violations

    if "subthema_clean" not in gdf.columns:
        return violations

    if "Onderhoudsproject" not in gdf.columns:
        gdf = gdf.copy()
        gdf["Onderhoudsproject"] = ""

    backbone_clean = _clean_list(BACKBONE_TYPES)
    project_empty = gdf["Onderhoudsproject"].apply(is_project_value_empty)
    is_exempt = _exemption_mask(gdf)

    # 1. Administratieve controle: object hoort een project te hebben, maar mist dit.
    missing_mask = ~is_exempt & project_empty

    for idx, row in gdf[missing_mask].iterrows():
        violations.append(
            make_violation(
                severity="error",
                category=CATEGORY_MISSING_PROJECT,
                rule_code="MISSING_PROJECT",
                object_id=idx,
                subthema=row.get("subthema", ""),
                message="Mist verplicht onderhoudsproject",
                missing_cols=["Onderhoudsproject"],
            )
        )

    # 2. Liggingcontrole: metrering is gevuld, maar niet als getal te lezen.
    # We melden bewust alleen niet-lege ongeldige waarden. Lege liggingvelden
    # kunnen later als aparte volledigheidsregel worden toegevoegd, maar zouden
    # in sommige exports nu te veel ruis geven.
    if "Metrering" in gdf.columns:
        for idx, row in gdf.iterrows():
            raw_metrering = row.get("Metrering", "")
            if is_empty_value(raw_metrering):
                continue

            parsed_hm = parse_hm_sort(raw_metrering)
            if parsed_hm >= 90000:
                violations.append(
                    make_violation(
                        severity="warning",
                        category=CATEGORY_LOCATION_DATA,
                        rule_code="INVALID_METRERING",
                        object_id=idx,
                        subthema=row.get("subthema", ""),
                        message=(
                            "Metrering is gevuld, maar kan niet als hectometrering worden gelezen: "
                            f"'{clean_display_value(raw_metrering)}'"
                        ),
                        missing_cols=["Metrering"],
                    )
                )

    # 3. Administratieve controle: object is uitgezonderd, maar heeft tóch een project.
    # Dit sluit aan op het werkprocesfilter 'objecten die onterecht een onderhoudsproject hebben'.
    wrong_project_mask = is_exempt & ~project_empty

    for idx, row in gdf[wrong_project_mask].iterrows():
        reason = maintenance_project_exemption_reason(row)
        message = "Heeft onderhoudsproject terwijl dit object is uitgezonderd"
        if reason:
            message = f"{message}: {reason}"

        violations.append(
            make_violation(
                severity="warning",
                category=CATEGORY_WRONG_PROJECT,
                rule_code="EXEMPT_HAS_PROJECT",
                object_id=idx,
                subthema=row.get("subthema", ""),
                message=message,
                missing_cols=["Onderhoudsproject"],
            )
        )

    # 4. Ruimtelijke controles.
    if graph is None:
        return violations

    # Precompute rijwaarden die in de graf-loop vaak worden geraadpleegd.
    # Dit voorkomt duizenden relatief dure gdf.loc-aanroepen op grotere wegen.
    index_set = set(gdf.index)
    sub_clean_by_id = gdf["subthema_clean"].to_dict()
    subthema_by_id = gdf["subthema"].to_dict() if "subthema" in gdf.columns else {}
    project_by_id = {
        idx: "" if is_project_value_empty(value) else str(value).strip()
        for idx, value in gdf["Onderhoudsproject"].items()
    }
    exempt_by_id = {idx: bool(value) for idx, value in is_exempt.items()}

    for node_id in graph.nodes:
        if node_id not in index_set:
            continue

        sub = sub_clean_by_id.get(node_id, "")

        if sub in backbone_clean or exempt_by_id.get(node_id, False):
            continue

        connected_to_backbone = False
        for neighbor in graph.neighbors(node_id):
            if neighbor not in index_set:
                continue

            if sub_clean_by_id.get(neighbor, "") in backbone_clean:
                connected_to_backbone = True
                break

        if not connected_to_backbone:
            violations.append(
                make_violation(
                    severity="warning",
                    category=CATEGORY_TOPOLOGY,
                    rule_code="FLOATING_SECONDARY",
                    object_id=node_id,
                    subthema=subthema_by_id.get(node_id, ""),
                    message="Zwevend secundair object: grenst nergens aan een hoofdroute (Rijbaan/Fiets/etc).",
                )
            )

    # Uitgezonderde objecten met projectnaam hebben hierboven al een gerichte
    # melding gekregen. We slaan ze hier over om dubbele, verwarrende meldingen te voorkomen.
    project_ids = [
        idx
        for idx, project_name in project_by_id.items()
        if project_name and not exempt_by_id.get(idx, False)
    ]

    for idx in project_ids:
        if idx not in graph:
            continue

        project_name = project_by_id.get(idx, "")
        my_sub = sub_clean_by_id.get(idx, "")

        try:
            neighbors = list(graph.neighbors(idx))
        except Exception:
            neighbors = []

        match_found = False
        for neighbor in neighbors:
            if neighbor not in index_set:
                continue

            if project_by_id.get(neighbor, "") == project_name:
                match_found = True
                break

        if not match_found:
            violations.append(
                make_violation(
                    severity="warning",
                    category=CATEGORY_PROJECT_CONSISTENCY,
                    rule_code="ISOLATED_PROJECT",
                    object_id=idx,
                    subthema=subthema_by_id.get(idx, ""),
                    message=f"Geïsoleerd t.o.v. project '{project_name}'. Geen directe buren met dit project.",
                    missing_cols=["Onderhoudsproject"],
                )
            )
            continue

        if my_sub not in backbone_clean:
            connected_to_project_backbone = False

            for neighbor in neighbors:
                if neighbor not in index_set:
                    continue

                if (
                    project_by_id.get(neighbor, "") == project_name
                    and sub_clean_by_id.get(neighbor, "") in backbone_clean
                ):
                    connected_to_project_backbone = True
                    break

            if not connected_to_project_backbone:
                violations.append(
                    make_violation(
                        severity="info",
                        category=CATEGORY_PROJECT_CONSISTENCY,
                        rule_code="NO_DIRECT_PROJECT_BACKBONE",
                        object_id=idx,
                        subthema=subthema_by_id.get(idx, ""),
                        message=(
                            f"Verbonden met '{project_name}', maar raakt niet direct "
                            "de hoofdrijbaan/fietspad van dit project."
                        ),
                    )
                )

    return violations
