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

import os
import logging
from collections import OrderedDict

from six.moves.configparser import ConfigParser

from bentoml.exceptions import BentoMLConfigException

logger = logging.getLogger(__name__)


class BentoConfigParser(ConfigParser):
    """ BentoML configuration parser

    :param default_config string - serve as default value when conf key not presented in
        environment var or user local config file
    """

    def __init__(self, default_config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bentoml_defaults = ConfigParser(*args, **kwargs)
        if default_config is not None:
            self.bentoml_defaults.read_string(default_config)

    @staticmethod
    def _env_var_name(section, key):
        return "BENTOML__{}__{}".format(section.upper(), key.upper())

    def get(self, section, key, **kwargs):
        """ A simple hierachical config access, priority order:
            1. environment var
            2. user config file
            3. bentoml default config file
        """
        section = str(section).lower()
        key = str(key).lower()

        env_var = self._env_var_name(section, key)
        if env_var in os.environ:
            return os.environ[env_var]

        if super().has_option(section, key):
            return super().get(section, key, **kwargs)

        if self.bentoml_defaults.has_option(section, key):
            return self.bentoml_defaults.get(section, key, **kwargs)
        else:
            raise BentoMLConfigException(
                "section/key '{}/{}' not found in BentoML config".format(section, key)
            )

    def as_dict(self, display_source=False):
        cfg = {}

        def add_config(cfg, source, config):
            for section in config:
                cfg.setdefault(section, OrderedDict())
                for k, val in config.items(section=section, raw=False):
                    if display_source:
                        cfg[section][k] = (val, source)
                    else:
                        cfg[section][k] = val

        add_config(cfg, "default", self.bentoml_defaults)
        add_config(cfg, "bentoml.cfg", self)

        for ev in os.environ:
            if ev.startswith("BENTOML__"):
                _, section, key = ev.split("__")
                val = os.environ[ev]
                if display_source:
                    val = (val, "env var")
                cfg.setdefault(section.lower(), OrderedDict()).update(
                    {key.lower(): val}
                )

        return cfg

    def __repr__(self):
        return "<BentoML config: {}>".format(str(self.as_dict()))
