\
from iasset_tool.utils import clean_display_value, parse_hm_sort, normalize_text


def test_clean_display_value_removes_trailing_dot_zero_for_numbers():
    assert clean_display_value("2005.0") == "2005"


def test_clean_display_value_keeps_project_name():
    assert clean_display_value("N351-HRB-01.2-03.4") == "N351-HRB-01.2-03.4"


def test_parse_hm_sort_accepts_comma():
    assert parse_hm_sort("12,3") == 12.3


def test_normalize_text():
    assert normalize_text(" Rijstrook ") == "rijstrook"
