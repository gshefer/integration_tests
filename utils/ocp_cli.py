from utils.conf import credentials
from utils.log import logger
from utils.ssh import SSHClient


class OcpCli(object):
    """This class provides CLI functionality for Openshift provider.
    """
    def __init__(self, provider):

        provider_cfme_data = provider.get_yaml_data()
        self.hostname = provider_cfme_data['hostname']
        creds = provider_cfme_data.get('ssh_creds', None)

        if not creds:
            raise
        if isinstance(creds, dict):
            self.username = creds.get('username', None)
            self.password = creds.get('password', None)
        else:
            self.username = credentials[creds].get('username', None)
            self.password = credentials[creds].get('password', None)

        self.ssh_client = SSHClient()
        self.ssh_client.load_system_host_keys()
        self.ssh_client.connect(self.hostname, username=self.username,
                                password=self.password, look_for_keys=True)
        self._command_counter = 0
        self.log_line_limit = 500

    def run_command(self, *args, **kwargs):
        logger.info('{} - Running SSH Command#{} : {}'
                    .format(self.hostname, self._command_counter, args[0]))
        results = self.ssh_client.run_command(*args, **kwargs)
        results_short = results[:max((self.log_line_limit, len(results)))]
        if results.success:
            logger.info('{} - Command#{} - Succeed: {}'
                        .format(self.hostname, self._command_counter, results_short))
        else:
            logger.warning('{} - Command#{} - Failed: {}'
                           .format(self.hostname, self._command_counter, results_short))
        self._command_counter += 1
        return results

    def close(self):
        self.ssh_client.close()
