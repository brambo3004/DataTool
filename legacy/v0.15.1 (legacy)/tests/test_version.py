from iasset_tool import __version__
from iasset_tool.config import APP_VERSION


def test_visible_app_version_matches_package_version():
    """Het zichtbare versienummer moet gelijk blijven aan de packageversie."""
    assert APP_VERSION == f"v{__version__}"
