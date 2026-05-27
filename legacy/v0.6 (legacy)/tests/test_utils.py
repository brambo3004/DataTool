import warnings

from iasset_tool.utils import clean_display_value, normalize_text, parse_date_info, parse_hm_sort, sanitize_filename


def test_clean_display_value_removes_trailing_dot_zero_for_numbers():
    assert clean_display_value("2005.0") == "2005"


def test_clean_display_value_keeps_project_name():
    assert clean_display_value("N351-HRB-01.2-03.4") == "N351-HRB-01.2-03.4"


def test_parse_hm_sort_accepts_comma():
    assert parse_hm_sort("12,3") == 12.3


def test_normalize_text():
    assert normalize_text(" Rijstrook ") == "rijstrook"


def test_parse_date_info_accepts_compact_timestamp_without_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert parse_date_info("20260512095736") == (2026, 5)

    assert not caught


def test_parse_date_info_accepts_compact_date():
    assert parse_date_info("20260512") == (2026, 5)


def test_sanitize_filename_removes_unsafe_characters():
    assert sanitize_filename("Jaar deklaag: N359/HRB") == "Jaar_deklaag_N359_HRB"
