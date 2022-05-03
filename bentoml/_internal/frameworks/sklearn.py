from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

import bentoml
from bentoml import Tag
from bentoml.models import Model
from bentoml.models import ModelContext
from bentoml.models import ModelOptions
from bentoml.exceptions import NotFound
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..models import PKL_EXT
from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from ..models.model import ModelSignature
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from sklearn.pipeline import Pipeline

    from .. import external_typing as ext
    from ..models import ModelStore

try:
    import joblib
    from joblib import parallel_backend
except ImportError:  # pragma: no cover
    try:
        from sklearn.utils._joblib import joblib
        from sklearn.utils._joblib import parallel_backend
    except ImportError:
        raise MissingDependencyException(
            """sklearn is required in order to use the module `bentoml.sklearn`, install
             sklearn with `pip install sklearn`. For more information, refer to
             https://scikit-learn.org/stable/install.html
            """
        )

MODULE_NAME = "bentoml.sklearn"

logger = logging.getLogger(__name__)


@inject
def load_model(
    tag: t.Union[str, Tag],
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> t.Union["BaseEstimator", "Pipeline"]:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`Union[BaseEstimator, Pipeline]`: an instance of :obj:`sklearn` model from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml

        sklearn = bentoml.sklearn.load_model('my_model:latest')
    """
    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )
    model_file = model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}")

    return joblib.load(model_file)


def save_model(
    name: str,
    model: BaseEstimator | Pipeline,
    *,
    signatures: dict[str, ModelSignature] | None = None,
    labels: t.Dict[str, str] | None = None,
    custom_objects: t.Dict[str, t.Any] | None = None,
    metadata: t.Dict[str, t.Any] | None = None,
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (:code:`Union[BaseEstimator, Pipeline]`):
            Instance of model to be saved.
        signatures (:code: `Dict[str, bool | BatchDimType | AnyType | tuple[AnyType]]`)
            Methods to expose for running inference on the target model. Signatures are
             used for creating Runner instances when serving model with bentoml.Service
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (:code:`Dict[str, Any]]`, `optional`, default to :code:`None`):
            user-defined additional python objects to be saved alongside the model,
             e.g. a tokenizer instance, preprocessor function, model configuration json
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is
        the user-defined model's name, and a generated `version`.

    Examples:

    .. code-block:: python

        import bentoml

        from sklearn.datasets import load_iris
        from sklearn.neighbors import KNeighborsClassifier

        model = KNeighborsClassifier()
        iris = load_iris()
        X = iris.data[:, :4]
        Y = iris.target
        model.fit(X, Y)

        tag = bentoml.sklearn.save_model('kneighbors', model)

        # load the model back:
        loaded = bentoml.sklearn.load_model("kneighbors:latest")
        # or:
        loaded = bentoml.sklearn.load_model(tag)
    """  # noqa
    context = ModelContext(
        framework_name="sklearn",
        framework_versions={"scikit-learn": get_pkg_version("scikit-learn")},
    )
    if signatures is None:
        logger.info(
            'Using default model signature `{"predict": {"batchable": False}}` for sklearn model'
        )
        signatures = {"predict": ModelSignature(batchable=False)}

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        labels=labels,
        custom_objects=custom_objects,
        metadata=metadata,
        context=context,
        signatures=signatures,
    ) as _model:

        joblib.dump(model, _model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}"))

        return _model.tag


def get(tag_like: str | Tag) -> Model:
    model = bentoml.models.get(tag_like)
    if model.info.module != MODULE_NAME:
        raise NotFound(
            f'Model "{tag_like}" saved with module "{MODULE_NAME}" is not found'
        )
    return model


def get_runnable(model: Model):
    ...
