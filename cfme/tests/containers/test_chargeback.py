# -*- coding: utf-8 -*-
from humanfriendly import parse_size, tokenize
import pytest

from cfme.containers.provider import ContainersProvider
from cfme.intelligence.chargeback import assignments
from cfme.intelligence.reports.reports import CustomReport
from utils import testgen
from utils.log import logger
from utils.units import CHARGEBACK_HEADER_NAMES, parse_number


pytestmark = [
    pytest.mark.meta(
        server_roles='+ems_metrics_coordinator +ems_metrics_collector +ems_metrics_processor'),
    pytest.mark.usefixtures('setup_provider_modscope'),
]
pytest_generate_tests = testgen.generate([ContainersProvider], scope='module')


# We cannot calculate the accurate value because the prices in the reports
# appears in a lower precision (floored). Hence we're using this accuracy coefficient:
TEST_MATCH_ACCURACY = 0.01

rate_interval_factor_lut = {'Hourly': 24, 'Daily': 7, 'Weekly': 4.29, 'Monthly': 1}


obj_types = ['Image', 'Project']
fixed_rates = ['Fixed1', 'Fixed2', 'CpuCores', 'Memory', 'Network']
variable_rates = ['CpuCores', 'Memory', 'Network']
rates = set(fixed_rates + variable_rates)
intervals = ['Hourly', 'Daily', 'Weekly', 'Monthly']


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


def gen_report_base(obj_type, provider, rate_desc, rate_interval):
    """Base function for report generation
    Args:
        :py:type:`str` obj_type: Object being tested; only 'Project' and 'Image' are supported
        :py:class:`ContainersProvider` provider: The Containers Provider
        :py:type:`str` rate_desc: The rate description as it appears in the report
        :py:type:`str` rate_interval: The rate interval, (Hourly/Daily/Weekly/Monthly)
    """
    title = 'report_{}_{}'.fromat(obj_type.lower(), rate_desc)
    if obj_type == 'Project':
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
    elif obj_type == 'Image':
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
    else:
        raise Exception("Unknown object type: {}".format(obj_type))

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

    logger.info('QUEUING CUSTOM CHARGEBACK REPORT FOR CONTAINER {}'.format(obj_type.upper()))
    report.queue(wait_for_finish=True)

    return report


def assign_custom_compute_rate(obj_type, chargeback_rate, provider):
    """Assign custom Compute rate for Labeled Container Images
    Args:
        :py:type:`str` obj_type: Object being tested; only 'Project' and 'Image' are supported
        :py:class:`ComputeRate` chargeback_rate: The chargeback rate object
        :py:class:`ContainersProvider` provider: The containers provider
    """
    if obj_type == 'Image':
        asignment = assignments.Assign(
            assign_to="Labeled Container Images",
            docker_labels="architecture",
            selections={
                'x86_64': chargeback_rate.description
            })
        logger.info('ASSIGNING COMPUTE RATE FOR LABELED CONTAINER IMAGES')
    elif obj_type == 'Project':
        asignment = assignments.Assign(
            assign_to="Selected Containers Providers",
            selections={
                provider.name: chargeback_rate.description
            })
        logger.info('ASSIGNING CUSTOM COMPUTE RATE FOR PROJECT CHARGEBACK')
    else:
        raise Exception("Unknown object type: {}".format(obj_type))

    asignment.computeassign()
    logger.info('Rate - {}: {}'.format(chargeback_rate.description,
                                       chargeback_rate.fields))

    return chargeback_rate


def abstract_test_chargeback_cost(
        obj_type, report_data, cb_rate, rate_key, rate_interval, soft_assert):

    """This is an abstract test function for testing rate costs.
    It's comparing the expected value that calculated by the rate
    to the value in the chargeback report
    Args:
        :py:type:`str` obj_type: Object being tested; only 'Project' and 'Image' are supported
        :py:type:`list` report_data: The report data (rows as list).
        :py:class:`ComputeRate` cb_rate: The chargeback rate object.
        :py:type:`str` rate_key: The rate key as it appear in the CHARGEBACK_HEADER_NAMES keys.
        :py:type:`str` rate_interval: The rate interval, (Hourly/Daily/Weekly/Monthly).
        :var soft_assert: soft_assert fixture.
    """

    report_headers = CHARGEBACK_HEADER_NAMES[rate_key]
    interval_factor = rate_interval_factor_lut[rate_interval]

    found_something_to_test = False
    for obj in report_data:
        for row in obj.rows:

            if row['Chargeback Rates'].lower() != cb_rate.description.lower():
                continue
            found_something_to_test = True

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
                'Chargeback {} mismatch: {}: "{}"; rate: "{}"; '
                'Expected price range: {} - {}; Found: {};'
                .format(obj_type, obj_type, row['{} Name'.format(obj_type)],
                        report_headers.cost_name,
                        expected_value - match_threshold,
                        expected_value + match_threshold, found_value))

    assert found_something_to_test, \
        'Could not find {} with the assigned rate: {}'.format(obj_type, cb_rate.description)


@pytest.yield_fixture()
def assign_compute_rate(obj_type, compute_rate, rate_type, rate, interval, provider):
    assign_custom_compute_rate(obj_type, compute_rate, provider)
    yield compute_rate.description
    revert_to_default_rate(provider)


@pytest.yield_fixture()
def chargeback_report(
        obj_type, assign_compute_rate, rate_type, rate, interval, provider):
    report = gen_report_base(obj_type, provider, assign_compute_rate, interval)
    yield list(report.get_saved_reports()[0].data)
    report.delete()


# =========== WHY DOES THIS EXIST JUST FOR PROJECTS AND NOT IMAGES ??
# I didn't touch it for that reason ^

# This is incorrect; we are testing object creation within a fixture; this test will never FAIL
# it can only PASS or ERROR because of that...
# @pytest.mark.polarion('CMP-10164')
# def test_project_chargeback_new_fixed_rate(new_chargeback_hourly_fixed_compute_rate):
#    flash.assert_success_message('Chargeback Rate "{}" was added'
#                                 .format(new_chargeback_hourly_fixed_compute_rate.description))
#
#
# Same here
# @pytest.mark.polarion('CMP-10165')
# def test_project_chargeback_assign_compute_custom_rate(assign_compute_custom_rate):
#    flash.assert_success_message('Rate Assignments saved')
#
#
# Same here
# @pytest.mark.long_running_env
# @pytest.mark.polarion('CMP-10166')
# def test_project_chargeback_report_fixed_rate(chargeback_report_for_hourly_fixed_rate):
#    assert chargeback_report_for_hourly_fixed_rate, 'Error in produced report, No records found'
#
# ===========


@pytest.mark.parametrize('obj_type', obj_types)
@pytest.mark.parametrize('rate', rates)
@pytest.mark.parametrize('interval', intervals)
@pytest.mark.parametrize('rate_type', ['fixed', 'variable'])
@pytest.mark.uncollectif(
    lambda rate_type, rate:
        (rate_type == 'variable' and rate not in variable_rates) or
        (rate_type == 'fixed' and rate not in fixed_rates)
)
@pytest.mark.long_running_env
# @pytest.mark.skip('This test is skipped due to a framework issue: '
#                   'https://github.com/ManageIQ/integration_tests/issues/5027')
def test_chargeback_rate(
        obj_type, chargeback_report, compute_rate, rate_type, rate, interval, soft_assert):
    abstract_test_chargeback_cost(
        obj_type, chargeback_report, compute_rate, rate, interval, soft_assert)
