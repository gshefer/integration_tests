import pytest
from humanfriendly import parse_size, tokenize

from utils import testgen
from utils.log import logger
from utils.units import CHARGEBACK_HEADER_NAMES, parse_number
from cfme.containers.provider import ContainersProvider
from cfme.intelligence.chargeback import assignments
from cfme.intelligence.reports.reports import CustomReport
from cfme.web_ui import flash
from cfme.fixtures.chargeback_rate import new_chargeback_rate_base


pytestmark = [
    pytest.mark.meta(
        server_roles='+ems_metrics_coordinator +ems_metrics_collector +ems_metrics_processor'),
    pytest.mark.usefixtures('setup_provider_modscope')
]
pytest_generate_tests = testgen.generate([ContainersProvider], scope='module')


# We cannot calculate the accurate value because the prices in the reports
# appears in a lower precision (floored). Hence we're using this accuracy coefficient:
TEST_MATCH_ACCURACY = 0.01

rate_interval_factor_lut = {'Hourly': 24, 'Daily': 7, 'Weekly': 4.29, 'Monthly': 1}


@pytest.mark.yield_fixture(scope='function')
def new_chargeback_hourly_fixed_compute_rate(appliance):
    # used for test_project_chargeback_new_fixed_rate
    cb_rate = new_chargeback_rate_base(appliance, 'Hourly', True)
    yield cb_rate
    cb_rate.delete()


@pytest.yield_fixture(scope='module')
def assign_compute_custom_rate(new_chargeback_rate, provider):
    """Assign custom Compute rate for project
    Args:
        :py:class:`ComputeRate` chargeback_rate: The chargeback rate object
        :py:class:`ContainersProvider` provider: The containers provider
    """
    asignment = assignments.Assign(
        assign_to="Selected Containers Providers",
        selections={
            provider.name: new_chargeback_rate.description
        })
    logger.info('ASSIGNING CUSTOM COMPUTE RATE FOR PROJECT CHARGEBACK')
    asignment.computeassign()

    yield new_chargeback_rate

    asignment = assignments.Assign(
        assign_to="Selected Containers Providers",
        selections={
            provider.name: "Default"
        })
    asignment.computeassign()


@pytest.yield_fixture(scope='module')
def new_chargeback_report(provider, assign_compute_custom_rate):
    """Generating a new chargeback report
    Args:
        :py:class:`ContainersProvider` provider: The Containers Provider
        :py:class:`ComputeRate` assign_compute_custom_rate: The chargeback rate object.
    Yield:
        :py:class:`CustomReport` report: The chargeback report.
        :py:class:`ComputeRate` assign_compute_custom_rate: The chargeback rate object.
    """
    title = 'report_project_' + assign_compute_custom_rate.description
    data = {
        'menu_name': title,
        'title': title,
        'base_report_on': 'Chargeback for Projects',
        'report_fields': ['Archived', 'Chargeback Rates', 'Fixed Compute Metric',
                          'Cpu Cores Used Cost', 'Cpu Cores Used Metric',
                          'Network I/O Used', 'Network I/O Used Cost',
                          'Fixed Compute Cost 1', 'Fixed Compute Cost 2',
                          'Memory Used', 'Memory Used Cost',
                          'Provider Name', 'Fixed Total Cost', 'Total Cost'],
        'filter_show_costs': 'Project',
        'provider': provider.name,
        'project': 'All Container Projects'
    }
    data['menu_name'] = title
    data['title'] = title
    data['provider'] = provider.name
    rate_interval = assign_compute_custom_rate['Used CPU Cores']['per_time']
    if rate_interval == 'Hourly':
        data['interval'] = 'Day'
        data['interval_end'] = 'Yesterday'
        data['interval_size'] = '1 Day'
    elif rate_interval == 'Daily':
        data['interval'] = 'Week',
        data['interval_end'] = 'Last Week'
        data['interval_size'] = '1 Week'
    elif rate_interval in ('Weekly', 'Monthly'):
        data['interval'] = 'Month',
        data['interval_end'] = 'Last Month'
        data['interval_size'] = '1 Month'
    else:
        raise Exception('Unsupported rate interval: "{}"; available options: '
                        '(Hourly/Daily/Weekly/Monthly)')
    report = CustomReport(is_candu=True, **data)
    report.create()

    logger.info('QUEUING CUSTOM CHARGEBACK REPORT FOR CONTAINER PROJECT')
    report.queue(wait_for_finish=True)

    yield report, assign_compute_custom_rate

    report.delete()


@pytest.mark.long_running_env
@pytest.mark.parametrize('rate_key', CHARGEBACK_HEADER_NAMES.keys())
def test_chargeback_cost(new_chargeback_report, rate_key, soft_assert):
    """Test rate costs.
    Args:
        :py:type: tuple of (`CustomReport`, `ComputeRate`) new_chargeback_report:
                  The chargeback report and the assigned rate.
        :py:type:`str` rate_key: The rate key as it appear in the CHARGEBACK_HEADER_NAMES keys.
        :var soft_assert: soft_assert fixture.
    """
    report_headers = CHARGEBACK_HEADER_NAMES[rate_key]
    rate_interval = assign_compute_custom_rate['Used CPU Cores']['per_time']
    new_chargeback_report, cb_rate = new_chargeback_report
    interval_factor = rate_interval_factor_lut[rate_interval]

    found_some_project_to_test = False
    for proj in new_chargeback_report.data:
        for row in proj.rows:

            if row['Chargeback Rates'].lower() != cb_rate.description.lower():
                continue
            found_some_project_to_test = True

            fixed_rate = float(cb_rate.fields[report_headers.rate_name]['fixed_rate'])
            variable_rate = float(cb_rate.fields[report_headers.rate_name].get('variable_rate', 0))

            if rate_key == 'Memory':
                size_, unit_ = tokenize(row[report_headers.metric_name].upper())
                metric = round(parse_size(str(size_) + unit_, binary=True) / 1048576.0, 2)
            else:
                metric = parse_number(row[report_headers.metric_name])

            expected_value = round(interval_factor * variable_rate * metric +
                                   interval_factor * fixed_rate, 2)
            found_value = round(parse_number(row[report_headers.cost_name]), 2)

            match_threshold = TEST_MATCH_ACCURACY * expected_value
            soft_assert(
                abs(found_value - expected_value) <= match_threshold,
                'Chargeback Project mismatch: Project: "{}"; rate: "{}"; '
                'Expected price range: {} - {}; Found: {};'
                .format(row['Project Name'], report_headers.cost_name,
                        expected_value - match_threshold,
                        expected_value + match_threshold, found_value))

    assert found_some_project_to_test, \
        'Could not find an project with the assigned rate: {}'.format(cb_rate.description)


@pytest.mark.skip('This test is skipped due to a framework issue: '
                  'https://github.com/ManageIQ/integration_tests/issues/5027')
@pytest.mark.polarion('CMP-10164')
def test_project_chargeback_new_fixed_rate(new_chargeback_hourly_fixed_compute_rate):
    flash.assert_success_message('Chargeback Rate "{}" was added'
                                 .format(new_chargeback_hourly_fixed_compute_rate.description))


@pytest.mark.skip('This test is skipped due to a framework issue: '
                  'https://github.com/ManageIQ/integration_tests/issues/5027')
@pytest.mark.polarion('CMP-10165')
def test_project_chargeback_assign_compute_custom_rate(assign_compute_custom_rate):
    flash.assert_success_message('Rate Assignments saved')
