import re
import os
import uuid
from datetime import datetime

from abc import abstractmethod
from bentoml.utils import Path
from bentoml.utils.module_helper import copy_module_and_local_dependencies
from bentoml.service_env import BentoServiceEnv
from bentoml.service import SingleModelBentoService
from bentoml.artifacts import ArtifactCollection
from bentoml.loader import load_bentoml_config
from bentoml.version import __version__ as BENTOML_VERSION

BENTO_MODEL_SETUP_PY_TEMPLATE = """\
import os
import pip
import logging
import pkg_resources
import setuptools

def _parse_requirements(file_path):
    pip_ver = pkg_resources.get_distribution('pip').version
    pip_version = list(map(int, pip_ver.split('.')[:2]))
    if pip_version >= [6, 0]:
        raw = pip.req.parse_requirements(file_path,
                                         session=pip.download.PipSession())
    else:
        raw = pip.req.parse_requirements(file_path)
    return [str(i.req) for i in raw]

try:
    install_reqs = _parse_requirements("requirements.txt")
except Exception:
    logging.warning('Fail load requirements file, so using default ones.')
    install_reqs = []

setuptools.setup(
    name='{name}',
    version='{version}',
    description="BentoML generated model module",
    long_description=\"\"\"{long_description}\"\"\",
    long_description_content_type="text/markdown",
    url="https://github.com/atalaya-io/BentoML",
    packages=setuptools.find_packages(),
    install_requires=install_reqs,
    include_package_data=True,
    package_data={{
        '{name}': ['bentoml.yml', 'artifacts/*']
    }},
    # TODO: add console entry_point for using model from cli
    # entry_points={{
    #
    # }}
)
"""

MANIFEST_IN_TEMPLATE = """\
include {model_name}/bentoml.yml
graft {model_name}/artifacts
"""

# TODO: add setup.sh and run
BENTO_SERVER_SINGLE_MODEL_DOCKERFILE_TEMPLATE = """\
FROM continuumio/miniconda3

# copy over model files
COPY . /model

# create conda env
RUN conda env create -f /model/environment.yml

# Pull the environment name out of the environment.yml
RUN echo "source activate $(head-1/model/environment.yml | cut -d ' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /model/environment.yml | cut -d ' ' -f2)/bin:$PATH

RUN conda install pip && pip install -r /model/requirements

# Run bento server with path to model
CMD ["/bin/bash", "-c", "bentoml serve --model-path=/model"]
"""

INIT_PY_TEMPLATE = """\
import os
from {model_name}.{module_name} import {model_name}

_module_path, _ = os.path.split(__file__)
{model_name}._bento_module_path = _module_path
"""

BENTOML_CONFIG_YAML_TEMPLATE = """\
model_name: {model_name}
model_version: {model_version}
bentoml_version: {bentoml_version}
module_name: {module_name}
module_file: {module_file}
created_at: {created_at}
"""

DEFAULT_MODEL_DESCRIPTION = """\
# BentoML(bentoml.ai) generated model archive
"""


def _validate_version_str(version_str):
    """
    Validate that version str starts with char, has length >= 4
    """
    regex = r"^[a-zA-Z][a-zA-Z0-9_]{3,}"
    if re.match(regex, version_str) is None:
        raise ValueError('Invalid version format: "{}"'.format(version_str))


def _generate_new_version_str():
    """
    Generate a version string in the format of YYYY_MM_DD_RandomHash
    """
    time_obj = datetime.now()
    date_string = time_obj.strftime('%Y_%m_%d')
    random_hash = uuid.uuid4().hex[:8]

    return date_string + '_' + random_hash


class BentoModel(SingleModelBentoService):
    """
    BentoModel is the base abstraction for defining how a ML model can
    be save to or load from a file directory in BentoArchive format
    """

    # User may override this in their Model definition class, to specify
    # pip version of the genreated python module
    _model_package_version = "1.0.0"

    # User may override this if they don't want the generated model to
    # have the same name as their Python model class name
    _model_name = None

    # This is overwritten when user install exported bento model as a
    # pip package, in that case, #load method will load from the installed
    # python package location
    _bento_module_path = None

    def __init__(self, *args, **kwargs):
        self.env = BentoServiceEnv()
        self.artifacts = ArtifactCollection()
        self.config(self.artifacts, self.env)
        self._version = None  # Will be set after #load or #save
        if args or kwargs:
            self.artifacts.pack(*args, **kwargs)

        if 'module_path' in kwargs:
            self._module_path = kwargs['module_path']

        super(BentoModel, self).__init__()

    @property
    def name(self):
        if self.__class__._model_name is not None:
            # TODO: verify self.__class__._model_name format, can't have space in it
            #  and can be valid folder name
            return self.__class__._model_name
        else:
            return self.__class__.__name__

    @property
    def version(self):
        return self._version

    def save(self, base_path, version=None, project_base_dir=None, copy_entire_project=False):
        Path(os.path.join(base_path), self.name).mkdir(parents=True, exist_ok=True)

        if version is not None:
            _validate_version_str(version)
        else:
            version = _generate_new_version_str()
        self._version = version

        # Update path to subfolder in the form of 'base/model_name/version/'
        path = os.path.join(base_path, self.name, version)
        if os.path.exists(path):
            raise ValueError("Version {version} in Path: {path} already "
                             "exist.".format(version=version, path=base_path))

        os.mkdir(path)
        module_base_path = os.path.join(path, self.name)
        os.mkdir(module_base_path)

        # write README.md with user model's docstring
        if self.__class__.__doc__:
            model_description = self.__class__.__doc__.strip()
        else:
            model_description = DEFAULT_MODEL_DESCRIPTION
        with open(os.path.join(path, 'README.md'), 'w') as f:
            f.write(model_description)

        # save all model artifacts to 'base_path/name/artifacts/' directory
        self.artifacts.save(module_base_path)

        # write conda environment, requirement.txt
        self.env.save(path)

        # copy over all custom model code
        module_name, module_file = copy_module_and_local_dependencies(self.__class__.__module__,
                                                                      os.path.join(path, self.name),
                                                                      project_base_dir,
                                                                      copy_entire_project)

        # create __init__.py
        with open(os.path.join(path, self.name, '__init__.py'), "w") as f:
            f.write(INIT_PY_TEMPLATE.format(model_name=self.name, module_name=module_name))

        # write setup.py, make exported model pip installable
        setup_py_content = BENTO_MODEL_SETUP_PY_TEMPLATE.format(
            name=self.name, version=self.__class__._model_package_version,
            long_description=model_description)
        with open(os.path.join(path, 'setup.py'), 'w') as f:
            f.write(setup_py_content)

        with open(os.path.join(path, 'MANIFEST.in'), 'w') as f:
            f.write(MANIFEST_IN_TEMPLATE.format(model_name=self.name))

        # write Dockerfile
        with open(os.path.join(path, 'Dockerfile'), 'w') as f:
            f.write(BENTO_SERVER_SINGLE_MODEL_DOCKERFILE_TEMPLATE)

        # write bentoml.yml
        bentoml_yml_content = BENTOML_CONFIG_YAML_TEMPLATE.format(
            model_name=self.name, bentoml_version=BENTOML_VERSION, model_version=version,
            module_name=module_name, module_file=module_file, created_at=str(datetime.now()))
        with open(os.path.join(path, 'bentoml.yml'), 'w') as f:
            f.write(bentoml_yml_content)
        # Also write bentoml.yml to module base path to make it accessible
        # as package data after pip installed as a python package
        with open(os.path.join(module_base_path, 'bentoml.yml'), 'w') as f:
            f.write(bentoml_yml_content)

        return path

    def load(self, path=None):
        # TODO: add model.env.verify() to check dependencies and python version etc

        if self.__class__._bento_module_path is not None:
            # When calling load from pip installled bento model, use installed
            # python package for loading and the same path for '/artifacts'
            # TODO: warn user when path is None here
            path = self.__class__._bento_module_path
            artifacts_path = path
        else:
            # When calling load on generated archive directory, look for /artifacts
            # directory under module sub-directory
            # TODO: assert path is not None
            artifacts_path = os.path.join(path, self.name)

        bentoml_config = load_bentoml_config(path)

        self.artifacts.load(artifacts_path)
        self._model_name = bentoml_config['model_name']
        self._version = bentoml_config['model_version']
        return self

    @abstractmethod
    def config(self, artifacts, env):
        pass

    @abstractmethod
    def predict(self, data):
        pass
