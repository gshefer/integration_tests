# -*- coding: utf-8 -*-
import pytest
import fauxfactory

from cfme.intelligence.chargeback import rates


def new_chargeback_rate(appliance, interval, variable_rate=1):
    """Create a new Chargeback compute rate
    Args:
        :py:class:`IPAppliance` appliance: The appliance
        :py:type:`str` rate_interval: The rate interval, (Hourly/Daily/Weekly/Monthly)
        :py:type:`int` or `float' variable_rate: The variable rate to apply (for all the metrics)
    """
    description = 'custom_rate_' + fauxfactory.gen_alphanumeric()
    data = {
        'Used CPU Cores': {'per_time': interval,
                           'fixed_rate': 1,
                           'variable_rate': variable_rate},
        'Fixed Compute Cost 1': {'per_time': interval,
                                 'fixed_rate': 1},
        'Fixed Compute Cost 2': {'per_time': interval,
                                 'fixed_rate': 1},
        'Used Memory': {'per_time': interval,
                        'fixed_rate': 1,
                        'variable_rate': variable_rate},
        'Used Network I/O': {'per_time': interval,
                             'fixed_rate': 1,
                             'variable_rate': variable_rate}
    }
    ccb = rates.ComputeRate(description, fields=data, appliance=appliance)
    ccb.create()
    return ccb


@pytest.yield_fixture(scope='session')
def new_chargeback_hourly_fixed_compute_rate(appliance):
    ccb = new_chargeback_rate(appliance, 'Hourly', variable_rate=0)
    yield ccb
    ccb.delete()


@pytest.yield_fixture(scope='session')
def new_chargeback_hourly_variable_compute_rate(appliance):
    ccb = new_chargeback_rate(appliance, 'Hourly', variable_rate=1)
    yield ccb
    ccb.delete()


@pytest.yield_fixture(scope='session')
def new_chargeback_daily_fixed_compute_rate(appliance):
    ccb = new_chargeback_rate(appliance, 'Daily', variable_rate=0)
    yield ccb
    ccb.delete()


@pytest.yield_fixture(scope='session')
def new_chargeback_daily_variable_compute_rate(appliance):
    ccb = new_chargeback_rate(appliance, 'Daily', variable_rate=1)
    yield ccb
    ccb.delete()


@pytest.yield_fixture(scope='session')
def new_chargeback_weekly_fixed_compute_rate(appliance):
    ccb = new_chargeback_rate(appliance, 'Weekly', variable_rate=0)
    yield ccb
    ccb.delete()


@pytest.yield_fixture(scope='session')
def new_chargeback_weekly_variable_compute_rate(appliance):
    ccb = new_chargeback_rate(appliance, 'Weekly', variable_rate=1)
    yield ccb
    ccb.delete()


@pytest.yield_fixture(scope='session')
def new_chargeback_monthly_fixed_compute_rate(appliance):
    ccb = new_chargeback_rate(appliance, 'Monthly', variable_rate=0)
    yield ccb
    ccb.delete()


@pytest.yield_fixture(scope='session')
def new_chargeback_monthly_variable_compute_rate(appliance):
    ccb = new_chargeback_rate(appliance, 'Monthly', variable_rate=1)
    yield ccb
    ccb.delete()
