from __future__ import annotations

import json
import typing as t
import logging
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

from .base import IODescriptor
from .json import MIME_TYPE_JSON
from ..types import LazyType
from ..utils.http import set_cookies
from ...exceptions import BadInput
from ...exceptions import BentoMLException
from ...exceptions import InternalServerError

if TYPE_CHECKING:
    import numpy as np

    from .. import external_typing as ext
    from ..context import InferenceApiContext as Context

logger = logging.getLogger(__name__)


def _is_matched_shape(
    left: t.Optional[t.Tuple[int, ...]],
    right: t.Optional[t.Tuple[int, ...]],
) -> bool:  # pragma: no cover
    if (left is None) or (right is None):
        return False

    if len(left) != len(right):
        return False

    for i, j in zip(left, right):
        if i == -1 or j == -1:
            continue
        if i == j:
            continue
        return False
    return True


class NumpyNdarray(IODescriptor["ext.NpNDArray"]):
    """
    :code:`NumpyNdarray` defines API specification for the inputs/outputs of a Service, where
    either inputs will be converted to or outputs will be converted from type
    :code:`numpy.ndarray` as specified in your API function signature.

    Sample implementation of a sklearn service:

    .. code-block:: python

        # sklearn_svc.py
        import bentoml
        from bentoml.io import NumpyNdarray
        import bentoml.sklearn

        runner = bentoml.sklearn.get("sklearn_model_clf").to_runner()

        svc = bentoml.Service("iris-classifier", runners=[runner])

        @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
        def predict(input_arr):
            res = runner.run(input_arr)
            return res

    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

        % bentoml serve ./sklearn_svc.py:svc --auto-reload

        (Press CTRL+C to quit)
        [INFO] Starting BentoML API server in development mode with auto-reload enabled
        [INFO] Serving BentoML Service "iris-classifier" defined in "sklearn_svc.py"
        [INFO] API Server running on http://0.0.0.0:3000

    Users can then send requests to the newly started services with any client:

    .. tabs::

    .. code-block:: bash

        % curl -X POST -H "Content-Type: application/json" --data '[[5,4,3,2]]' http://0.0.0.0:3000/predict

        [1]%

    Args:
        dtype (:code:`numpy.typings.DTypeLike`, `optional`, default to :code:`None`):
            Data Type users wish to convert their inputs/outputs to. Refers to `arrays dtypes <https://numpy.org/doc/stable/reference/arrays.dtypes.html>`_ for more information.
        enforce_dtype (:code:`bool`, `optional`, default to :code:`False`):
            Whether to enforce a certain data type. if :code:`enforce_dtype=True` then :code:`dtype` must be specified.
        shape (:code:`Tuple[int, ...]`, `optional`, default to :code:`None`):
            Given shape that an array will be converted to. For example:

            .. code-block:: python

                from bentoml.io import NumpyNdarray

                @svc.api(input=NumpyNdarray(shape=(2,2), enforce_shape=False), output=NumpyNdarray())
                def predict(input_array: np.ndarray) -> np.ndarray:
                    # input_array will be reshaped to (2,2)
                    result = await runner.run(input_array)

            When `enforce_shape=True` is provided, BentoML will raise an exception if
            the input array received does not match the `shape` provided.

        enforce_shape (:code:`bool`, `optional`, default to :code:`False`):
            Whether to enforce a certain shape. If `enforce_shape=True` then `shape`
            must be specified.

    Returns:
        :obj:`~bentoml._internal.io_descriptors.IODescriptor`: IO Descriptor that :code:`np.ndarray`.
    """

    def __init__(
        self,
        dtype: t.Optional[t.Union[str, "np.dtype[t.Any]"]] = None,
        enforce_dtype: bool = False,
        shape: t.Optional[t.Tuple[int, ...]] = None,
        enforce_shape: bool = False,
    ):
        import numpy as np

        if dtype is not None and not isinstance(dtype, np.dtype):
            # Convert from primitive type or type string, e.g.:
            # np.dtype(float)
            # np.dtype("float64")
            try:
                dtype = np.dtype(dtype)
            except TypeError as e:
                raise BentoMLException(f'NumpyNdarray: Invalid dtype "{dtype}": {e}')

        self._dtype = dtype
        self._shape = shape
        self._enforce_dtype = enforce_dtype
        self._enforce_shape = enforce_shape

    def _infer_types(self) -> str:  # pragma: no cover
        if self._dtype is not None:
            name = self._dtype.name
            if name.startswith("int") or name.startswith("uint"):
                var_type = "integer"
            elif name.startswith("float") or name.startswith("complex"):
                var_type = "number"
            else:
                var_type = "object"
        else:
            var_type = "object"
        return var_type

    def _items_schema(self) -> t.Dict[str, t.Any]:
        if self._shape is not None:
            if len(self._shape) > 1:
                return {"type": "array", "items": {"type": self._infer_types()}}
            return {"type": self._infer_types()}
        return {}

    def input_type(self) -> LazyType["ext.NpNDArray"]:
        return LazyType("numpy", "ndarray")

    def openapi_schema_type(self) -> t.Dict[str, t.Any]:
        return {"type": "array", "items": self._items_schema()}

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""
        return {MIME_TYPE_JSON: {"schema": self.openapi_schema_type()}}

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""
        return {MIME_TYPE_JSON: {"schema": self.openapi_schema_type()}}

    def _verify_ndarray(
        self,
        obj: "ext.NpNDArray",
        exception_cls: t.Type[Exception] = BadInput,
    ) -> "ext.NpNDArray":
        import numpy as np

        if self._dtype is not None and self._dtype != obj.dtype:
            # ‘same_kind’ means only safe casts or casts within a kind, like float64
            # to float32, are allowed.
            if np.can_cast(obj.dtype, self._dtype, casting="same_kind"):
                obj = obj.astype(self._dtype, casting="same_kind")  # type: ignore
            else:
                msg = f'{self.__class__.__name__}: Expecting ndarray of dtype "{self._dtype}", but "{obj.dtype}" was received.'
                if self._enforce_dtype:
                    raise exception_cls(msg)
                else:
                    logger.warning(msg)

        if self._shape is not None and not _is_matched_shape(self._shape, obj.shape):
            msg = f'{self.__class__.__name__}: Expecting ndarray of shape "{self._shape}", but "{obj.shape}" was received.'
            if self._enforce_shape:
                raise exception_cls(msg)
            try:
                obj = obj.reshape(self._shape)
            except ValueError as e:
                logger.warning(f"{msg} Failed to reshape: {e}.")

        return obj

    async def from_http_request(self, request: Request) -> "ext.NpNDArray":
        """
        Process incoming requests and convert incoming
         objects to `numpy.ndarray`

        Args:
            request (`starlette.requests.Requests`):
                Incoming Requests
        Returns:
            a `numpy.ndarray` object. This can then be used
             inside users defined logics.
        """
        import numpy as np

        obj = await request.json()
        res: "ext.NpNDArray"
        try:
            res = np.array(obj, dtype=self._dtype)  # type: ignore[arg-type]
        except ValueError:
            res = np.array(obj)  # type: ignore[arg-type]
        res = self._verify_ndarray(res, BadInput)
        return res

    async def to_http_response(self, obj: ext.NpNDArray, ctx: Context | None = None):
        """
        Process given objects and convert it to HTTP response.

        Args:
            obj (`np.ndarray`):
                `np.ndarray` that will be serialized to JSON
        Returns:
            HTTP Response of type `starlette.responses.Response`. This can
             be accessed via cURL or any external web traffic.
        """
        obj = self._verify_ndarray(obj, InternalServerError)
        if ctx is not None:
            res = Response(
                json.dumps(obj.tolist()),
                media_type=MIME_TYPE_JSON,
                headers=ctx.response.metadata,  # type: ignore (bad starlette types)
                status_code=ctx.response.status_code,
            )
            set_cookies(res, ctx.response.cookies)
            return res
        else:
            return Response(json.dumps(obj.tolist()), media_type=MIME_TYPE_JSON)

    @classmethod
    def from_sample(
        cls,
        sample_input: "ext.NpNDArray",
        enforce_dtype: bool = True,
        enforce_shape: bool = True,
    ) -> "NumpyNdarray":
        """
        Create a NumpyNdarray IO Descriptor from given inputs.

        Args:
            sample_input (:code:`np.ndarray`): Given sample np.ndarray data
            enforce_dtype (;code:`bool`, `optional`, default to :code:`True`):
                Enforce a certain data type. :code:`dtype` must be specified at function
                signature. If you don't want to enforce a specific dtype then change
                :code:`enforce_dtype=False`.
            enforce_shape (:code:`bool`, `optional`, default to :code:`False`):
                Enforce a certain shape. :code:`shape` must be specified at function
                signature. If you don't want to enforce a specific shape then change
                :code:`enforce_shape=False`.

        Returns:
            :obj:`~bentoml._internal.io_descriptors.NumpyNdarray`: :code:`NumpyNdarray` IODescriptor from given users inputs.

        Example:

        .. code-block:: python

            import numpy as np
            from bentoml.io import NumpyNdarray
            arr = [[1,2,3]]
            inp = NumpyNdarray.from_sample(arr)

            ...
            @svc.api(input=inp, output=NumpyNdarray())
            def predict() -> np.ndarray:...
        """
        return cls(
            dtype=sample_input.dtype,
            shape=sample_input.shape,
            enforce_dtype=enforce_dtype,
            enforce_shape=enforce_shape,
        )
