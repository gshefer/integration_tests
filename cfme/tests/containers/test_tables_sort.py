import pytest
from utils import testgen
from utils.version import current_version
from cfme.web_ui import toolbar, SortTable
from cfme.containers.project import Project
from cfme.containers.replicator import Replicator
from cfme.containers.service import Service
from cfme.containers.route import Route
from cfme.containers.provider import ContainersProvider
from utils.appliance.implementations.ui import navigate_to
from utils.blockers import BZ


pytestmark = [
    pytest.mark.uncollectif(
        lambda: current_version() < "5.6"),
    pytest.mark.usefixtures('setup_provider'),
    pytest.mark.tier(1)]
pytest_generate_tests = testgen.generate([ContainersProvider], scope='function')


TEST_OBJECTS = [
    pytest.mark.polarion('CMP-9924')(ContainersProvider),
    pytest.mark.polarion('CMP-9925')(Project),
    pytest.mark.polarion('CMP-9926')(Route),
    pytest.mark.polarion('CMP-9927')(Service),
    pytest.mark.polarion('CMP-9928')(Replicator)
]


@pytest.mark.meta(blockers=[
    BZ(1392413, unblock=lambda cls: cls != ContainersProvider),
    BZ(1409360, unblock=lambda cls: cls != ContainersProvider)
])
@pytest.mark.parametrize('cls', TEST_OBJECTS)
def test_tables_sort(cls):

    pytest.skip('This test is currently skipped due to an issue in the testing framework:'
                ' https://github.com/ManageIQ/integration_tests/issues/4052')

    navigate_to(cls, 'All')
    toolbar.select('List View')
    # NOTE: We must re-instantiate here table
    # in order to prevent StaleElementException or UsingSharedTables
    sort_tbl = SortTable(table_locator="//div[@id='list_grid']//table")
    header_texts = [header.text for header in sort_tbl.headers]
    for col, header_text in enumerate(header_texts):

        if not header_text:
            continue

        # Checking both orders
        sort_tbl.sort_by(header_text, 'ascending')
        rows_ascending = [r[col].text for r in sort_tbl.rows()]
        sort_tbl.sort_by(header_text, 'descending')
        rows_descending = [r[col].text for r in sort_tbl.rows()]

        assert rows_ascending[::-1] == rows_descending
