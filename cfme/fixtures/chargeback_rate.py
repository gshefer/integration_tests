# -*- coding: utf-8 -*-
import pytest
import fauxfactory

from cfme.intelligence.chargeback import rates


def new_chargeback_rate_base(appliance, rate_interval, fixed):
    """Create a new Chargeback compute rate
    Args:
        :py:class:`IPAppliance` appliance: The appliance
        :py:type:`str` rate_interval: The rate interval, (Hourly/Daily/Weekly/Monthly)
        :py:type:`bool' fixed: Whether to use only fixed rate (for all the metrics)
    """
    description = 'custom_rate_' + fauxfactory.gen_alphanumeric()
    data = {
        'Used CPU Cores': {'per_time': rate_interval,
                           'fixed_rate': 1,
                           'variable_rate': int(fixed)},
        'Fixed Compute Cost 1': {'per_time': rate_interval,
                                 'fixed_rate': 1},
        'Fixed Compute Cost 2': {'per_time': rate_interval,
                                 'fixed_rate': 1},
        'Used Memory': {'per_time': rate_interval,
                        'fixed_rate': 1,
                        'variable_rate': int(fixed)},
        'Used Network I/O': {'per_time': rate_interval,
                             'fixed_rate': 1,
                             'variable_rate': int(fixed)}
    }
    ccb = rates.ComputeRate(description, fields=data, appliance=appliance)
    ccb.create()
    return ccb


@pytest.fixture(scope='session', params=[True, False])
def is_fixed_rate(request):
    return request.param


@pytest.yield_fixture(scope='session', params=['Hourly', 'Daily', 'Weekly', 'Monthly'])
def new_chargeback_rate(appliance, request, is_fixed_rate):
    ccb = new_chargeback_rate_base(appliance, request.param, is_fixed_rate)
    yield ccb
    ccb.delete()
