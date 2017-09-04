# -*- coding: utf-8 -*-
import pytest
from humanfriendly import parse_size, tokenize

from utils import testgen
from utils.log import logger
from utils.units import CHARGEBACK_HEADER_NAMES, parse_number
from cfme.containers.provider import ContainersProvider
from cfme.intelligence.chargeback import assignments
from cfme.intelligence.reports.reports import CustomReport


pytestmark = [
    pytest.mark.meta(
        server_roles='+ems_metrics_coordinator +ems_metrics_collector +ems_metrics_processor'),
    pytest.mark.usefixtures('setup_provider_modscope'),
    pytest.mark.long_running_env
]
pytest_generate_tests = testgen.generate([ContainersProvider], scope='module')


# We cannot calculate the accurate value because the prices in the reports
# appears in a lower precision (floored). Hence we're using this accuracy coefficient:
TEST_MATCH_ACCURACY = 0.01

rate_interval_factor_lut = {'Hourly': 24, 'Daily': 7, 'Weekly': 4.29, 'Monthly': 1}


def assign_compute_custom_rate(chargeback_rate, provider):
    """Assign custom Compute rate for Labeled Container Images
    Args:
        :py:class:`ComputeRate` chargeback_rate: The chargeback rate object
        :py:class:`ContainersProvider` provider: The containers provider
    """
    asignment = assignments.Assign(
        assign_to="Labeled Container Images",
        docker_labels="architecture",
        selections={
            'x86_64': chargeback_rate.description
        })
    logger.info('ASSIGNING COMPUTE RATE FOR LABELED CONTAINER IMAGES')
    asignment.computeassign()
    logger.info('Rate - {}: {}'.format(chargeback_rate.description,
                                       chargeback_rate.fields))

    return chargeback_rate


def revert_to_default_rate(provider):
    """Reverting the assigned compute rate to default rate
    Args:
        :py:class:`ContainersProvider` provider: the Containers Provider to be selected for the rate
    """
    asignment = assignments.Assign(
        assign_to="Selected Containers Providers",
        selections={
            provider.name: "Default"
        })
    asignment.computeassign()


def gen_report_base(provider, rate_desc, rate_interval):
    """Base function for report generation
    Args:
        :py:class:`ContainersProvider` provider: The Containers Provider
        :py:type:`str` rate_desc: The rate description as it appears in the report
        :py:type:`str` rate_interval: The rate interval, (Hourly/Daily/Weekly/Monthly)
    """
    title = 'report_image_' + rate_desc
    data = {
        'base_report_on': 'Chargeback Container Images',
        'report_fields': ['Archived', 'Chargeback Rates', 'Fixed Compute Metric',
                          'Cpu Cores Used Cost', 'Cpu Cores Used Metric',
                          'Network I/O Used', 'Network I/O Used Cost',
                          'Fixed Compute Cost 1', 'Fixed Compute Cost 2',
                          'Memory Used', 'Memory Used Cost',
                          'Provider Name', 'Fixed Total Cost', 'Total Cost'],
        'filter_show_costs': 'Container Image'
    }
    data['menu_name'] = title
    data['title'] = title
    data['provider'] = provider.name
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

    logger.info('QUEUING CUSTOM CHARGEBACK REPORT FOR CONTAINER IMAGE')
    report.queue(wait_for_finish=True)

    return report


@pytest.yield_fixture(scope='module')
def assign_compute_hourly_fixed_rate(new_chargeback_hourly_fixed_compute_rate, provider):
    # Assign hourly fixed compute rate
    assign_compute_custom_rate(new_chargeback_hourly_fixed_compute_rate, provider)
    yield new_chargeback_hourly_fixed_compute_rate.description
    revert_to_default_rate(provider)


@pytest.yield_fixture(scope='module')
def assign_compute_daily_fixed_rate(new_chargeback_daily_fixed_compute_rate, provider):
    # Assign daily fixed compute rate
    assign_compute_custom_rate(new_chargeback_daily_fixed_compute_rate, provider)
    yield new_chargeback_daily_fixed_compute_rate.description
    revert_to_default_rate(provider)


@pytest.yield_fixture(scope='module')
def assign_compute_weekly_fixed_rate(new_chargeback_weekly_fixed_compute_rate, provider):
    # Assign weekly fixed compute rate
    assign_compute_custom_rate(new_chargeback_weekly_fixed_compute_rate, provider)
    yield new_chargeback_weekly_fixed_compute_rate.description
    revert_to_default_rate(provider)


@pytest.yield_fixture(scope='module')
def assign_compute_monthly_fixed_rate(new_chargeback_monthly_fixed_compute_rate, provider):
    # Assign monthly fixed compute rate
    assign_compute_custom_rate(new_chargeback_monthly_fixed_compute_rate, provider)
    yield new_chargeback_monthly_fixed_compute_rate.description
    revert_to_default_rate(provider)


@pytest.yield_fixture(scope='module')
def assign_compute_hourly_variable_rate(new_chargeback_hourly_variable_compute_rate, provider):
    # Assign hourly variable compute rate
    assign_compute_custom_rate(new_chargeback_hourly_variable_compute_rate, provider)
    yield new_chargeback_hourly_variable_compute_rate.description
    revert_to_default_rate(provider)


@pytest.yield_fixture(scope='module')
def assign_compute_daily_variable_rate(new_chargeback_daily_variable_compute_rate, provider):
    # Assign daily variable compute rate
    assign_compute_custom_rate(new_chargeback_daily_variable_compute_rate, provider)
    yield new_chargeback_daily_variable_compute_rate.description
    revert_to_default_rate(provider)


@pytest.yield_fixture(scope='module')
def assign_compute_weekly_variable_rate(new_chargeback_weekly_variable_compute_rate, provider):
    # Assign weekly variable compute rate
    assign_compute_custom_rate(new_chargeback_weekly_variable_compute_rate, provider)
    yield new_chargeback_weekly_variable_compute_rate.description
    revert_to_default_rate(provider)


@pytest.yield_fixture(scope='module')
def assign_compute_monthly_variable_rate(new_chargeback_monthly_variable_compute_rate, provider):
    # Assign monthly variable compute rate
    assign_compute_custom_rate(new_chargeback_monthly_variable_compute_rate, provider)
    yield new_chargeback_monthly_variable_compute_rate.description
    revert_to_default_rate(provider)


@pytest.yield_fixture(scope='module')
def chargeback_report_for_hourly_fixed_rate(assign_compute_hourly_fixed_rate, provider):
    # Chargeback report for hourly fixed compute rate
    report = gen_report_base(provider, assign_compute_hourly_fixed_rate, 'Hourly')
    yield list(report.get_saved_reports()[0].data)
    report.delete()


@pytest.yield_fixture(scope='module')
def chargeback_report_for_hourly_variable_rate(assign_compute_hourly_variable_rate, provider):
    # Chargeback report for hourly variable compute rate
    report = gen_report_base(provider, assign_compute_hourly_variable_rate, 'Hourly')
    yield list(report.get_saved_reports()[0].data)
    report.delete()


@pytest.yield_fixture(scope='module')
def chargeback_report_for_daily_fixed_rate(assign_compute_daily_fixed_rate, provider):
    # Chargeback report for daily fixed compute rate
    report = gen_report_base(provider, assign_compute_daily_fixed_rate, 'Daily')
    yield list(report.get_saved_reports()[0].data)
    report.delete()


@pytest.yield_fixture(scope='module')
def chargeback_report_for_daily_variable_rate(assign_compute_daily_variable_rate, provider):
    # Chargeback report for daily variable compute rate
    report = gen_report_base(provider, assign_compute_daily_variable_rate, 'Daily')
    yield list(report.get_saved_reports()[0].data)
    report.delete()


@pytest.yield_fixture(scope='module')
def chargeback_report_for_weekly_fixed_rate(assign_compute_weekly_fixed_rate, provider):
    # Chargeback report for weekly fixed compute rate
    report = gen_report_base(provider, assign_compute_weekly_fixed_rate, 'Weekly')
    yield list(report.get_saved_reports()[0].data)
    report.delete()


@pytest.yield_fixture(scope='module')
def chargeback_report_for_weekly_variable_rate(assign_compute_weekly_variable_rate, provider):
    # Chargeback report for weekly variable compute rate
    report = gen_report_base(provider, assign_compute_weekly_variable_rate, 'Weekly')
    yield list(report.get_saved_reports()[0].data)
    report.delete()


@pytest.yield_fixture(scope='module')
def chargeback_report_for_monthly_fixed_rate(assign_compute_monthly_fixed_rate, provider):
    # Chargeback report for monthly fixed compute rate
    report = gen_report_base(provider, assign_compute_monthly_fixed_rate, 'Monthly')
    yield list(report.get_saved_reports()[0].data)
    report.delete()


@pytest.yield_fixture(scope='module')
def chargeback_report_for_monthly_variable_rate(assign_compute_monthly_variable_rate, provider):
    # Chargeback report for monthly variable compute rate
    report = gen_report_base(provider, assign_compute_monthly_variable_rate, 'Monthly')
    yield list(report.get_saved_reports()[0].data)
    report.delete()


def abstract_test_chargeback_cost(report_data, cb_rate, rate_key, rate_interval, soft_assert):

    """This is an abstract test function for test fixed rate costs.
    It's comparing the expected value that calculated by the rate
    to the value in the chargeback report
    Args:
        :py:type:`list` report_data: The report data (rows as list).
        :py:class:`ComputeRate` cb_rate: The chargeback rate object.
        :py:type:`str` rate_key: The rate key as it appear in the CHARGEBACK_HEADER_NAMES keys.
        :py:type:`str` rate_interval: The rate interval, (Hourly/Daily/Weekly/Monthly).
        :var soft_assert: soft_assert fixture.
    """
    report_headers = CHARGEBACK_HEADER_NAMES[rate_key]
    interval_factor = rate_interval_factor_lut[rate_interval]

    found_some_image_to_test = False
    for img in report_data:
        for row in img.rows:

            if row['Chargeback Rates'].lower() != cb_rate.description.lower():
                continue
            found_some_image_to_test = True

            fixed_rate = float(cb_rate.fields[report_headers.rate_name]['fixed_rate'])
            variable_rate = float(cb_rate.fields[report_headers.rate_name]['variable_rate'])

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
                'Chargeback image mismatch: Image: "{}"; rate: "{}"; '
                'Expected price range: {} - {}; Found: {};'
                .format(row['Image Name'], report_headers.cost_name,
                        expected_value - match_threshold,
                        expected_value + match_threshold, found_value))

    assert found_some_image_to_test, \
        'Could not find an image with the assigned rate: {}'.format(cb_rate.description)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-10496')
def test_image_chargeback_fixed_rate_1_hourly_fixed_rate(
        chargeback_report_for_hourly_fixed_rate, new_chargeback_hourly_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_hourly_fixed_rate,
                                  new_chargeback_hourly_fixed_compute_rate,
                                  'Fixed1', 'Hourly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-10503')
def test_image_chargeback_fixed_rate_2_hourly_fixed_rate(
        chargeback_report_for_hourly_fixed_rate, new_chargeback_hourly_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_hourly_fixed_rate,
                                  new_chargeback_hourly_fixed_compute_rate,
                                  'Fixed2', 'Hourly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-10489')
def test_image_chargeback_cpu_cores_hourly_fixed_rate(
        chargeback_report_for_hourly_fixed_rate, new_chargeback_hourly_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_hourly_fixed_rate,
                                  new_chargeback_hourly_fixed_compute_rate,
                                  'CpuCores', 'Hourly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-10481')
def test_image_chargeback_memory_used_hourly_fixed_rate(
        chargeback_report_for_hourly_fixed_rate, new_chargeback_hourly_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_hourly_fixed_rate,
                                  new_chargeback_hourly_fixed_compute_rate,
                                  'Memory', 'Hourly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-10510')
def test_image_chargeback_network_io_hourly_fixed_rate(
        chargeback_report_for_hourly_fixed_rate, new_chargeback_hourly_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_hourly_fixed_rate,
                                  new_chargeback_hourly_fixed_compute_rate,
                                  'Network', 'Hourly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_fixed_rate_1_daily_fixed_rate(
        chargeback_report_for_daily_fixed_rate, new_chargeback_daily_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_daily_fixed_rate,
                                  new_chargeback_daily_fixed_compute_rate,
                                  'Fixed1', 'Daily', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_fixed_rate_2_daily_fixed_rate(
        chargeback_report_for_daily_fixed_rate, new_chargeback_daily_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_daily_fixed_rate,
                                  new_chargeback_daily_fixed_compute_rate,
                                  'Fixed2', 'Daily', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_cpu_cores_daily_fixed_rate(
        chargeback_report_for_daily_fixed_rate, new_chargeback_daily_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_daily_fixed_rate,
                                  new_chargeback_daily_fixed_compute_rate,
                                  'CpuCores', 'Daily', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_memory_used_daily_fixed_rate(
        chargeback_report_for_daily_fixed_rate, new_chargeback_daily_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_daily_fixed_rate,
                                  new_chargeback_daily_fixed_compute_rate,
                                  'Memory', 'Daily', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_network_io_daily_fixed_rate(
        chargeback_report_for_daily_fixed_rate, new_chargeback_daily_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_daily_fixed_rate,
                                  new_chargeback_daily_fixed_compute_rate,
                                  'Network', 'Daily', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_fixed_rate_1_weekly_fixed_rate(
        chargeback_report_for_weekly_fixed_rate, new_chargeback_weekly_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_weekly_fixed_rate,
                                  new_chargeback_weekly_fixed_compute_rate,
                                  'Fixed1', 'Weekly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_fixed_rate_2_weekly_fixed_rate(
        chargeback_report_for_weekly_fixed_rate, new_chargeback_weekly_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_weekly_fixed_rate,
                                  new_chargeback_weekly_fixed_compute_rate,
                                  'Fixed2', 'Weekly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_cpu_cores_weekly_fixed_rate(
        chargeback_report_for_weekly_fixed_rate, new_chargeback_weekly_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_weekly_fixed_rate,
                                  new_chargeback_weekly_fixed_compute_rate,
                                  'CpuCores', 'Weekly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_memory_used_weekly_fixed_rate(
        chargeback_report_for_weekly_fixed_rate, new_chargeback_weekly_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_weekly_fixed_rate,
                                  new_chargeback_weekly_fixed_compute_rate,
                                  'Memory', 'Weekly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_network_io_weekly_fixed_rate(
        chargeback_report_for_weekly_fixed_rate, new_chargeback_weekly_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_weekly_fixed_rate,
                                  new_chargeback_weekly_fixed_compute_rate,
                                  'Network', 'Weekly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_fixed_rate_1_monthly_fixed_rate(
        chargeback_report_for_monthly_fixed_rate, new_chargeback_monthly_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_monthly_fixed_rate,
                                  new_chargeback_monthly_fixed_compute_rate,
                                  'Fixed1', 'Monthly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_fixed_rate_2_monthly_fixed_rate(
        chargeback_report_for_monthly_fixed_rate, new_chargeback_monthly_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_monthly_fixed_rate,
                                  new_chargeback_monthly_fixed_compute_rate,
                                  'Fixed2', 'Monthly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_cpu_cores_monthly_fixed_rate(
        chargeback_report_for_monthly_fixed_rate, new_chargeback_monthly_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_monthly_fixed_rate,
                                  new_chargeback_monthly_fixed_compute_rate,
                                  'CpuCores', 'Monthly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_memory_used_monthly_fixed_rate(
        chargeback_report_for_monthly_fixed_rate, new_chargeback_monthly_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_monthly_fixed_rate,
                                  new_chargeback_monthly_fixed_compute_rate,
                                  'Memory', 'Monthly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_network_io_monthly_fixed_rate(
        chargeback_report_for_monthly_fixed_rate, new_chargeback_monthly_fixed_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_monthly_fixed_rate,
                                  new_chargeback_monthly_fixed_compute_rate,
                                  'Network', 'Monthly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-10482')
def test_image_chargeback_memory_used_hourly_variable_rate(
        chargeback_report_for_hourly_variable_rate, new_chargeback_hourly_variable_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(
        chargeback_report_for_hourly_variable_rate, new_chargeback_hourly_variable_compute_rate,
        'Memory', 'Hourly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-10511')
def test_image_chargeback_network_used_hourly_variable_rate(
        chargeback_report_for_hourly_variable_rate, new_chargeback_hourly_variable_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_hourly_variable_rate,
                                  new_chargeback_hourly_variable_compute_rate,
                                  'Network', 'Hourly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_cpu_cores_used_hourly_variable_rate(
        chargeback_report_for_hourly_variable_rate, new_chargeback_hourly_variable_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_hourly_variable_rate,
                                  new_chargeback_hourly_variable_compute_rate,
                                  'CpuCores', 'Hourly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_memory_used_daily_variable_rate(
        chargeback_report_for_daily_variable_rate, new_chargeback_daily_variable_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(
        chargeback_report_for_daily_variable_rate, new_chargeback_daily_variable_compute_rate,
        'Memory', 'Daily', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_network_used_daily_variable_rate(
        chargeback_report_for_daily_variable_rate, new_chargeback_daily_variable_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_daily_variable_rate,
                                  new_chargeback_daily_variable_compute_rate,
                                  'Network', 'Daily', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_cpu_cores_used_daily_variable_rate(
        chargeback_report_for_daily_variable_rate, new_chargeback_daily_variable_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_daily_variable_rate,
                                  new_chargeback_daily_variable_compute_rate,
                                  'CpuCores', 'Daily', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_memory_used_weekly_variable_rate(
        chargeback_report_for_weekly_variable_rate, new_chargeback_weekly_variable_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(
        chargeback_report_for_weekly_variable_rate, new_chargeback_weekly_variable_compute_rate,
        'Memory', 'Weekly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_network_used_weekly_variable_rate(
        chargeback_report_for_weekly_variable_rate, new_chargeback_weekly_variable_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_weekly_variable_rate,
                                  new_chargeback_weekly_variable_compute_rate,
                                  'Network', 'Weekly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_cpu_cores_used_weekly_variable_rate(
        chargeback_report_for_weekly_variable_rate, new_chargeback_weekly_variable_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_weekly_variable_rate,
                                  new_chargeback_weekly_variable_compute_rate,
                                  'CpuCores', 'Weekly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_memory_used_monthly_variable_rate(
        chargeback_report_for_monthly_variable_rate, new_chargeback_monthly_variable_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(
        chargeback_report_for_monthly_variable_rate, new_chargeback_monthly_variable_compute_rate,
        'Memory', 'Monthly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_network_used_monthly_variable_rate(
        chargeback_report_for_monthly_variable_rate, new_chargeback_monthly_variable_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_monthly_variable_rate,
                                  new_chargeback_monthly_variable_compute_rate,
                                  'Network', 'Monthly', soft_assert)


@pytest.mark.long_running_env
@pytest.mark.polarion('CMP-<TBD>')
def test_image_chargeback_cpu_cores_used_monthly_variable_rate(
        chargeback_report_for_monthly_variable_rate, new_chargeback_monthly_variable_compute_rate,
        soft_assert):
    abstract_test_chargeback_cost(chargeback_report_for_monthly_variable_rate,
                                  new_chargeback_monthly_variable_compute_rate,
                                  'CpuCores', 'Monthly', soft_assert)
