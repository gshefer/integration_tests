# -*- coding: utf-8 -*-
import pytest
import fauxfactory

from cfme.intelligence.chargeback import rates


@pytest.yield_fixture()
def compute_rate(appliance, rate_type, rate, interval):
    variable_rate = 1 if rate_type == 'variable' else 0
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
    if not ccb.exists:
        ccb.create()
    yield ccb


# We can also have a session scoped fixture for clean up at the very end
# HERE; if we really care that much, TODO
