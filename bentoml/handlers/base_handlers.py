# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import pandas as pd
import numpy as np

from bentoml.exceptions import InvalidArgument


class BentoHandler:
    """Handler in BentoML is the layer between a user API request and
    the input to user's API function.
    """

    HTTP_METHODS = ["POST", "GET"]

    def handle_request(self, request, func):
        """Handles an HTTP request, convert it into corresponding data
        format that user API function is expecting, and return API
        function result as the HTTP response to client

        :param request: Flask request object
        :param func: user API function
        """
        raise NotImplementedError

    def handle_cli(self, args, func):
        """Handles an CLI command call, convert CLI arguments into
        corresponding data format that user API function is expecting, and
        prints the API function result to console output

        :param args: CLI arguments
        :param func: user API function
        """
        raise NotImplementedError

    def handle_aws_lambda_event(self, event, func):
        """Handles a Lambda event, convert event dict into corresponding
        data format that user API function is expecting, and use API
        function result as response

        :param event: A dict containing AWS lambda event information
        :param func: user API function
        """
        raise NotImplementedError

    @property
    def request_schema(self):
        return {"application/json": {"schema": {"type": "object"}}}

    @property
    def pip_dependencies(self):
        return []


def get_output_str(result, output_format, output_orient="records"):
    if output_format == "str":
        return str(result)
    elif output_format == "json":
        if isinstance(result, pd.DataFrame):
            return result.to_json(orient=output_orient)
        elif isinstance(result, np.ndarray):
            return json.dumps(result.tolist())
        else:
            try:
                return json.dumps(result)
            except (TypeError, OverflowError):
                # when result is not JSON serializable
                return json.dumps(str(result))
    else:
        raise InvalidArgument("Output format {} is not supported".format(output_format))


class NestedDecoder:

    @staticmethod
    def B64_DECODER(obj):
        import base64
        B64_KEY = 'b64'
        if isinstance(obj, dict) and B64_KEY in obj:
            return base64.b64decode(obj[B64_KEY])
        else:
            return obj

    @staticmethod
    def TF_TENSOR_DECODER(obj):
        import tensorflow as tf
        import numpy as np
        if isinstance(obj, tf.Tensor):
            if tf.__version__.startswith("2."):
                ndarray = obj.numpy()
            elif tf.__version__.startswith("1."):
                with tf.compat.v1.Session():
                    ndarray = obj.numpy()
            else:
                raise NotImplementedError()
            if isinstance(ndarray, np.ndarray):
                if ndarray.dtype == np.dtype(object):
                    ndarray = ndarray.astype(str)
                return ndarray.tolist()
            else:
                if isinstance(ndarray, bytes):
                    ndarray = ndarray.decode("utf-8")
                return ndarray
        else:
            return obj

    @staticmethod
    def NDARRAY_DECODER(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    DECODERS = (
        B64_DECODER,
        TF_TENSOR_DECODER,
        NDARRAY_DECODER,
    )

    def __init__(self, enabled_decoders):
        self.enabled_decoders = enabled_decoders

    def __call__(self, obj):
        for decoder in self.enabled_decoders:
            obj = decoder(obj)

        if isinstance(obj, dict):
            return {k: self(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self(v) for v in obj]
        else:
            return obj
