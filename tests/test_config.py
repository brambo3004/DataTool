import json

from iasset_tool.config import (
    DEFAULT_MAINTENANCE_RULES,
    load_maintenance_rules,
    maintenance_rules_summary,
)
from iasset_tool.maintenance_control import _project_category_family


def test_load_maintenance_rules_merges_external_json(tmp_path):
    """Een lokaal beheerbestand mag standaardregels gericht overschrijven."""
    config_file = tmp_path / "onderhoudsregels.json"
    config_file.write_text(
        json.dumps(
            {
                "domeinregels": {
                    "subthema_exceptions": ["testobject"],
                    "project_category_families": {"TEST": ["TST", "TSTR"]},
                }
            }
        ),
        encoding="utf-8",
    )

    rules, warnings, source = load_maintenance_rules(config_file)

    assert source == "configuratiebestand"
    assert warnings == []
    assert rules["domeinregels"]["subthema_exceptions"] == ["testobject"]
    # Niet overschreven delen blijven beschikbaar uit de ingebouwde standaard.
    assert "backbone_types" in rules["domeinregels"]
    assert rules["domeinregels"]["project_category_families"]["TEST"] == ["TST", "TSTR"]


def test_load_maintenance_rules_uses_safe_fallback_for_invalid_json(tmp_path):
    """Een kapot configuratiebestand mag de app nooit laten crashen."""
    config_file = tmp_path / "onderhoudsregels.json"
    config_file.write_text("{dit is geen geldige json", encoding="utf-8")

    rules, warnings, source = load_maintenance_rules(config_file)

    assert source == "ingebouwde standaard"
    assert warnings
    assert rules["domeinregels"]["backbone_types"] == DEFAULT_MAINTENANCE_RULES["domeinregels"]["backbone_types"]


def test_maintenance_rules_summary_exposes_loaded_rule_counts():
    """De zijbalk kan compacte beheerinformatie tonen zonder de regels te muteren."""
    summary = maintenance_rules_summary()

    assert summary["primaire_subthemas"] >= 5
    assert summary["uitzonderingssubthemas"] >= 1
    assert "bron" in summary


def test_project_category_family_uses_configured_families():
    """De onderhoudscontrole gebruikt de configureerbare categoriefamilies."""
    assert _project_category_family("HRBR") == "HRB"
    assert _project_category_family("PWL") == "PW"
    assert _project_category_family("FPR") == "FP"
