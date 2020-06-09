import logging

from bentoml.yatai.proto.repository_pb2 import BentoUri
from e2e_tests.cli_operations import delete_bento
from e2e_tests.yatai_server.utils import (
    get_bento_service,
    run_bento_service_prediction,
    local_yatai_server,
    BentoServiceForYataiTest,
)

logger = logging.getLogger('bentoml.test')


def test_yatai_server_with_postgres_and_local_storage(postgres_db_container_url):
    with local_yatai_server(postgres_db_container_url):
        logger.info('Saving bento service')
        svc = BentoServiceForYataiTest()
        svc.save()
        bento_tag = f'{svc.name}:{svc.version}'
        logger.info('BentoService saved')

        logger.info("Display bentoservice info")
        get_svc_result = get_bento_service(svc.name, svc.version)
        logger.info(get_svc_result)
        assert (
            get_svc_result.bento.uri.type == BentoUri.LOCAL
        ), 'BentoService storage type mismatched, expect LOCAL'

        logger.info('Validate BentoService prediction result')
        run_result = run_bento_service_prediction(bento_tag, '[]')
        logger.info(run_result)
        assert 'cat' in run_result, 'Unexpected BentoService prediction result'

        logger.info(f'Deleting saved bundle {bento_tag}')
        delete_svc_result = delete_bento(bento_tag)
        assert f"{bento_tag} deleted" in delete_svc_result
