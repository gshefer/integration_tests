import multiprocessing as mp

import pytest

from manageiq_client.api import ManageIQClient as MiqApi

from cfme import test_requirements
from cfme.rest.gen_data import automation_requests_data as _automation_requests_data
from cfme.rest.gen_data import a_provider as _a_provider
from cfme.rest.gen_data import vm as _vm
from utils.wait import wait_for
from utils.version import current_version
from utils.blockers import BZ


pytestmark = [test_requirements.rest]


@pytest.fixture(scope='module')
def a_provider(request):
    return _a_provider(request)


@pytest.fixture(scope='module')
def vm(request, a_provider, rest_api_modscope):
    return _vm(request, a_provider, rest_api_modscope)


def wait_for_requests(requests):
    def _finished():
        for request in requests:
            request.reload()
            if request.request_state != 'finished':
                return False
        return True

    wait_for(_finished, num_sec=600, delay=5, message="automation_requests finished")


def gen_pending_requests(collection, rest_api, vm, requests=False):
    requests_data = _automation_requests_data(vm, approve=False, requests_collection=requests)
    response = collection.action.create(*requests_data[:2])
    assert rest_api.response.status_code == 200
    assert len(response) == 2
    for resource in response:
        assert resource.request_state == 'pending'
    return response


def create_requests(collection, rest_api, automation_requests_data, multiple):
    if multiple:
        requests = collection.action.create(*automation_requests_data)
    else:
        requests = collection.action.create(
            automation_requests_data[0])
    assert rest_api.response.status_code == 200

    wait_for_requests(requests)

    for request in requests:
        assert request.approval_state == 'approved'
        resource = collection.get(id=request.id)
        assert resource.type == 'AutomationRequest'


def create_pending_requests(collection, rest_api, requests_pending):
    for request in requests_pending:
        resource = collection.get(id=request.id)
        assert rest_api.response.status_code == 200
        assert resource.type == 'AutomationRequest'


def approve_requests(collection, rest_api, requests_pending, from_detail):
    if from_detail:
        for request in requests_pending:
            request.action.approve(reason="I said so")
    else:
        collection.action.approve(
            reason="I said so", *requests_pending)
    assert rest_api.response.status_code == 200

    wait_for_requests(requests_pending)

    for request in requests_pending:
        request.reload()
        assert request.approval_state == 'approved'


def deny_requests(collection, rest_api, requests_pending, from_detail):
    if from_detail:
        for request in requests_pending:
            request.action.deny(reason="I said so")
    else:
        collection.action.deny(
            reason="I said so", *requests_pending)
    assert rest_api.response.status_code == 200

    wait_for_requests(requests_pending)

    for request in requests_pending:
        request.reload()
        assert request.approval_state == 'denied'


class TestAutomationRequestsRESTAPI(object):
    """Tests using /api/automation_requests."""

    @pytest.fixture(scope='function')
    def collection(self, rest_api):
        return rest_api.collections.automation_requests

    @pytest.fixture(scope='function')
    def automation_requests_data(self, vm):
        return _automation_requests_data(vm)

    @pytest.fixture(scope='function')
    def requests_pending(self, rest_api, vm):
        return gen_pending_requests(rest_api.collections.automation_requests, rest_api, vm)

    @pytest.mark.tier(3)
    @pytest.mark.parametrize(
        'multiple', [False, True],
        ids=['one_request', 'multiple_requests'])
    def test_create_requests(self, collection, rest_api, automation_requests_data, multiple):
        """Test adding the automation request using /api/automation_requests.

        Metadata:
            test_flag: rest, requests
        """
        create_requests(collection, rest_api, automation_requests_data, multiple)

    @pytest.mark.tier(3)
    def test_create_pending_requests(self, rest_api, requests_pending, collection):
        """Tests creating pending requests using /api/automation_requests.

        Metadata:
            test_flag: rest, requests
        """
        create_pending_requests(collection, rest_api, requests_pending)

    @pytest.mark.tier(3)
    @pytest.mark.parametrize(
        'from_detail', [True, False],
        ids=['from_detail', 'from_collection'])
    def test_approve_requests(self, collection, rest_api, requests_pending, from_detail):
        """Tests approving automation requests using /api/automation_requests.

        Metadata:
            test_flag: rest, requests
        """
        approve_requests(collection, rest_api, requests_pending, from_detail)

    @pytest.mark.tier(3)
    @pytest.mark.parametrize(
        'from_detail', [True, False],
        ids=['from_detail', 'from_collection'])
    def test_deny_requests(self, collection, rest_api, requests_pending, from_detail):
        """Tests denying automation requests using /api/automation_requests.

        Metadata:
            test_flag: rest, requests
        """
        deny_requests(collection, rest_api, requests_pending, from_detail)


class TestAutomationRequestsCommonRESTAPI(object):
    """Tests using /api/requests (common collection for all requests types)."""

    @pytest.fixture(scope='function')
    def collection(self, rest_api):
        return rest_api.collections.requests

    @pytest.fixture(scope='function')
    def automation_requests_data(self, vm):
        return _automation_requests_data(vm, requests_collection=True)

    @pytest.fixture(scope='function')
    def requests_pending(self, rest_api, vm):
        return gen_pending_requests(rest_api.collections.requests, rest_api, vm, requests=True)

    @pytest.mark.uncollectif(lambda: current_version() < '5.7')
    @pytest.mark.tier(3)
    @pytest.mark.parametrize(
        'multiple', [False, True],
        ids=['one_request', 'multiple_requests'])
    def test_create_requests(self, collection, rest_api, automation_requests_data, multiple):
        """Test adding the automation request using /api/requests.

        Metadata:
            test_flag: rest, requests
        """
        create_requests(collection, rest_api, automation_requests_data, multiple)

    @pytest.mark.uncollectif(lambda: current_version() < '5.7')
    @pytest.mark.tier(3)
    def test_create_pending_requests(self, collection, rest_api, requests_pending):
        """Tests creating pending requests using /api/requests.

        Metadata:
            test_flag: rest, requests
        """
        create_pending_requests(collection, rest_api, requests_pending)

    @pytest.mark.uncollectif(lambda: current_version() < '5.7')
    @pytest.mark.tier(3)
    @pytest.mark.parametrize(
        'from_detail', [True, False],
        ids=['from_detail', 'from_collection'])
    def test_approve_requests(self, collection, rest_api, requests_pending, from_detail):
        """Tests approving automation requests using /api/requests.

        Metadata:
            test_flag: rest, requests
        """
        approve_requests(collection, rest_api, requests_pending, from_detail)

    @pytest.mark.uncollectif(lambda: current_version() < '5.7')
    @pytest.mark.tier(3)
    @pytest.mark.parametrize(
        'from_detail', [True, False],
        ids=['from_detail', 'from_collection'])
    def test_deny_requests(self, collection, rest_api, requests_pending, from_detail):
        """Tests denying automation requests using /api/requests.

        Metadata:
            test_flag: rest, requests
        """
        deny_requests(collection, rest_api, requests_pending, from_detail)

    @pytest.mark.uncollectif(lambda: current_version() < '5.7')
    @pytest.mark.tier(3)
    @pytest.mark.parametrize(
        'from_detail', [True, False],
        ids=['from_detail', 'from_collection'])
    def test_edit_requests(self, collection, rest_api, requests_pending, from_detail):
        """Tests editing requests using /api/requests.

        Metadata:
            test_flag: rest, requests
        """
        body = {'options': {'arbitrary_key_allowed': 'test_rest'}}

        if from_detail:
            if BZ('1418331', forced_streams=['5.7', 'upstream']).blocks:
                pytest.skip("Affected by BZ1418331, cannot test.")
            for request in requests_pending:
                request.action.edit(**body)
                assert rest_api.response.status_code == 200
        else:
            identifiers = []
            for i, resource in enumerate(requests_pending):
                loc = ({'id': resource.id}, {'href': '{}/{}'.format(collection._href, resource.id)})
                identifiers.append(loc[i % 2])
            collection.action.edit(*identifiers, **body)
            assert rest_api.response.status_code == 200

        for request in requests_pending:
            request.reload()
            assert request.options['arbitrary_key_allowed'] == 'test_rest'

    @pytest.mark.uncollectif(lambda: current_version() < '5.7')
    def test_create_requests_parallel(self, rest_api):
        """Create automation requests in parallel.

        Metadata:
            test_flag: rest, requests
        """
        output = mp.Queue()
        entry_point = rest_api._entry_point
        auth = rest_api._auth

        def _gen_automation_requests(output):
            api = MiqApi(entry_point, auth, verify_ssl=False)
            requests_data = _automation_requests_data(
                'nonexistent_vm', requests_collection=True, approve=False)
            api.collections.requests.action.create(*requests_data[:2])
            result = (api.response.status_code, api.response.json())
            output.put(result)

        processes = [
            mp.Process(target=_gen_automation_requests, args=(output,))
            for _ in range(4)]

        for proc in processes:
            proc.start()

        # wait for all processes to finish
        for proc in processes:
            proc.join()

        for proc in processes:
            status, response = output.get()
            assert status == 200
            for result in response['results']:
                assert result['request_type'] == 'automation'
