import pytest

from bentoml.yatai.client import YataiClient
from tests.bento_service_examples.example_bento_service import ExampleBentoService


class TestModel(object):
    def predict_dataframe(self, df):
        return df["col1"].sum()

    def predict_image(self, input_data):
        assert input_data is not None
        return input_data.shape

    def predict_json(self, input_data):
        assert input_data is not None
        return {"ok": True}


@pytest.fixture()
def bento_service():
    """Create a new TestBentoService
    """
    test_model = TestModel()

    # When the TestBentoService got saved and loaded again in the test, the two class
    # attribute below got set to the loaded BentoService class. Resetting it here so it
    # does not effect other tests
    ExampleBentoService._bento_service_bundle_path = None
    ExampleBentoService._bento_service_bundle_version = None

    test_svc = ExampleBentoService()
    test_svc.pack('model', test_model)
    return test_svc


@pytest.fixture()
def bento_bundle_path(bento_service):  # pylint:disable=redefined-outer-name
    """Create a new TestBentoService, saved it to tmpdir, and return full saved_path
    """
    saved_path = bento_service.save()
    yield saved_path
    yc = YataiClient()
    yc.repository.dangerously_delete_bento(bento_service.name, bento_service.version)
