from ntpath import join
import os
import mlflow
from mlflow.pyfunc import model
from mlflow.types.schema import Schema
from mlflow.types.schema import TensorSpec

from pydantic import BaseModel
import numpy as np
import pandas as pd
from fastapi import HTTPException, Depends
from datetime import datetime

import dill as pickle
import importlib

import shutil
import pathlib


class BaseHandler:

    dtype_sample = {
        "float64": 1.234,
        "float32": 1.234,
        "int": 1,
        "int64": 1,
        "int32": 1,
        "str": "A",
        "object": "?"
    }

    def _setup_model(self):
        from mlflow.pyfunc import _download_artifact_from_uri
        import subprocess
        import glob

        # create a logfile
        setup_logfile = open(os.path.join(
            self.work_folder, f"{os.path.basename(self.model_folder)}_setup_log.txt"), "a")

        setup_logfile.write(f"Start download {self.model_folder}\n")
        setup_logfile.flush()
        _download_artifact_from_uri(
            artifact_uri=self.model_version_source,
            output_path=self.work_folder)
        setup_logfile.write(f"End download {self.model_folder}\n")
        setup_logfile.flush()

        result_env = subprocess.run(
            ["python", "-m",  "venv", "env"], stdout=setup_logfile, stderr=setup_logfile, cwd=self.model_folder)
        setup_logfile.flush()

        env_pip = os.path.join(self.model_folder, "./env/Scripts/pip")

        # get the requirements of the model
        req_file = os.path.join(self.model_folder, "requirements.txt")
        if os.path.exists(req_file):
            with open(req_file, "r") as f:
                req = f.readlines()
        else:
            req = mlflow.pyfunc.get_default_pip_requirements()

        # remove the directly given libs from the requirements
        for lib_folder in glob.glob(os.path.join(self.model_folder, "code/*")):
            lib_name = os.path.basename(lib_folder)
            req = [r for r in req if lib_name not in r]
        req = [r.rstrip("\n") for r in req]

        # log stripped requirements
        setup_logfile.write("Install req:\n")
        for r in req:
            setup_logfile.write(f"{r}\n")
        setup_logfile.flush()

        # install the requirements
        result_pip = subprocess.run(
            [env_pip, "install", *req], stdout=setup_logfile, stderr=setup_logfile, cwd=self.model_folder)
        setup_logfile.flush()

        # install direct libs
        for lib_folder in glob.glob(os.path.join(self.model_folder, "code/*")):

            # install requirements of the libs
            lib_req_file = os.path.join(lib_folder, "requirements.txt")
            if os.path.exists(lib_req_file):
                result_pip = subprocess.run(
                    [env_pip,
                     "install", "-r",
                     lib_req_file],
                    stdout=setup_logfile, stderr=setup_logfile, cwd=self.model_folder)
            setup_logfile.flush()

            # install the libs
            result_lib = subprocess.run([
                os.path.join(self.model_folder, "./env/Scripts/python"),
                "./setup.py",
                "build",
                "install"], stdout=setup_logfile, stderr=setup_logfile,
                cwd=lib_folder)
            setup_logfile.flush()

        # close the logging
        setup_logfile.close()

    def _get_free_port(self, host='127.0.0.1'):
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((host, 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def _start_server(self):
        from time import sleep

        # create a logfile
        serve_logfile = open(os.path.join(
            self.work_folder, f"{os.path.basename(self.model_folder)}_servelog.txt"), "a")

        import subprocess
        cmd = f"""
        {os.path.join(self.model_folder, './env/Scripts/activate')}
        {os.path.join(self.model_folder, './env/Scripts/mlflow.exe')} models serve -m . --no-conda --port {self.port}
        
        """

        process = subprocess.Popen(
            """cmd.exe""", cwd=self.model_folder,
            stdin=subprocess.PIPE, stdout=serve_logfile, stderr=serve_logfile,
            text=True, shell=True
        )

        try:
            process.communicate(cmd, timeout=15)
        except:
            pass

        serve_logfile.flush()


    def _health(self):

        import requests

        

        res = requests.post(f"http://localhost:{self.port}/invocations", json=inp)

        print(res)


    def _get_input_example(self):
        import json
        try:
            with open(os.path.join(self.model_folder, "input_example.json"), "r") as f:
                inp = json.load(f)

            return inp
        except:
            return {}





    def __init__(self, server,  m, model_version):
        self.server = server
        self.m = m
        self.name = m.name

        self.model_version_source = (model_version.source)
        self.run_id = model_version.run_id
        self.work_folder = os.path.join(server.full_cache_dir, self.run_id)
        self.model_folder = os.path.join(
            self.work_folder, model_version.source.split("/")[-1])
        pathlib.Path(self.work_folder).mkdir(parents=True, exist_ok=True)

        if os.path.exists(self.model_folder) == False:
            self._setup_model()

        self.port = self._get_free_port()
        self._start_server()

        from mlflow.models.model import Model as _Model
        metadata = _Model.load(os.path.join(self.model_folder, mlflow.pyfunc.MLMODEL_FILE_NAME))

        
        #self._health()



        try:
            input_schema = metadata.get_input_schema()
        except:
            input_schema = None

        try:
            output_schema = metadata.get_output_schema()
        except:
            output_schema = None

        if not input_schema:
            input_schema = Schema([])
        if not output_schema:
            output_schema = Schema([])

        self.input_example_data = self._get_input_example()


        self.version = model_version.version
        self.source = model_version.source

        self.input_schema = input_schema if input_schema else {"inputs": []}
        self.output_schema = output_schema if output_schema else {"inputs": []}

        self.output_schema._inputs.append(
            TensorSpec(np.dtype("int"), [1], "x__version"))
        self.output_schema._inputs.append(
            TensorSpec(np.dtype("str"), [1], "x__mlflow_id"))

        self.description = m.description
        timestamp = model_version.creation_timestamp/1000
        self.creation = datetime.fromtimestamp(
            timestamp).strftime('%Y-%m-%d %H:%M')

        self.long_description = f"""{m.description}\n\n"""
        try:
            if len(input_schema.inputs) > 0:
                self.long_description += f"<b>Input Schema:</b> {self.get_schema_string(input_schema)} <br/>\n"
        except:
            pass
        try:
            if len(output_schema.inputs) > 0:
                self.long_description += f"<b>Output Schema:</b> {self.get_schema_string(output_schema)}<br/>\n"
        except:
            pass

        self.long_description += f"""
<b>Version: </b> {self.get_version_link(m.name, model_version)}<br/>
<b>Run: </b> {self.get_experiment_link(m.name, model_version)}<br/>
<b>Creation: </b> {self.creation}
        """

        self.update_schema_classes()
        self.register_route()

    def update_schema_classes(self):

        if self.input_schema:
            input_schema_class = type(
                self.name+"-input",
                (BaseModel, ),
                {el["name"]:
                 self.input_example_data[el["name"]]
                 if el["name"] in self.input_example_data else
                 self.get_example_el(el) for el in self.input_schema.to_dict() if "name" in el})
        else:
            input_schema_class = None

        try:
            np_input = self.numpy_input(
                self.input_example_data, self.input_schema)
            output_example_data = self.model.predict(np_input)
            output_example_data = {k: v.tolist()
                                   for k, v in output_example_data.items()}
        except:
            output_example_data = {}

        if self.output_schema:
            output_schema_class = type(
                self.name+"-output",
                (BaseModel, ),
                {el["name"]:
                 output_example_data[el["name"]]
                 if el["name"] in output_example_data else
                 self.get_example_el(el) for el in self.output_schema.to_dict()})
        else:
            output_schema_class = None

        self.input_schema_class = input_schema_class() if input_schema_class else None
        self.output_schema_class = output_schema_class() if output_schema_class else None
        self.input_schema_class_type = input_schema_class
        self.output_schema_class_type = output_schema_class

    def register_route(self):
        if self.input_schema_class is None or len(self.input_schema.inputs) == 0:
            # no input create get interface
            if len(self.server.config.token) > 0:
                @self.server.app.get(
                    self.server.config.basepath+'/'+self.name,
                    description=self.long_description,
                    name=self.name, tags=["Models"],
                    response_model=self.output_schema_class_type
                )
                async def func(token: str = Depends(self.server.security)):
                    self.server.check_token(token)
                    return self.apply_model(None)
            else:
                @self.server.app.get(
                    self.server.config.basepath+'/'+self.name,
                    description=self.long_description,
                    name=self.name, tags=["Models"],
                    response_model=self.output_schema_class_type
                )
                async def func():
                    return self.apply_model(None)
        else:
            # create post interface
            if len(self.server.config.token) > 0:
                @self.server.app.post(
                    self.server.config.basepath+'/'+self.name,
                    description=self.long_description,
                    name=self.name, tags=["Models"],
                    response_model=self.output_schema_class_type
                )
                async def func(
                    data: self.input_schema_class_type,
                    token: str = Depends(self.server.security)
                ):
                    self.server.check_token(token)
                    return self.apply_model(data)
            else:
                @self.server.app.post(
                    self.server.config.basepath+'/'+self.name,
                    description=self.long_description,
                    name=self.name, tags=["Models"],
                    response_model=self.output_schema_class_type
                )
                async def func(data: self.input_schema_class_type):
                    return self.apply_model(data)

    def apply_model(self, data):
        # create a numpy input array
        if self.input_schema_class is None or len(self.input_schema.inputs) == 0:
            np_input = np.array([])
        else:
            try:
                np_input = self.numpy_input(data.__dict__, self.input_schema)
            except Exception as ex:
                raise self.get_error_message("Parse input error", ex)

        try:
            model_output = self.model.predict(np_input)
        except Exception as ex:
            raise self.get_error_message("Model prediction error", ex)

        try:
            output = self.parse_output(model_output)
        except Exception as ex:
            raise self.get_error_message("Parse output error", ex)

        output.update(
            {"x__version": [int(self.version)], "x__mlflow_id": [self.run_id]})

        return output

    def get_version_link(self, name, model_version):
        return f"{model_version.version}"

    def get_experiment_link(self, name, model_version):
        return f"{model_version.run_id}"

    def get_nested(self, dtype, shape):
        if len(shape) == 1:
            return [self.dtype_sample[dtype]]*shape[0]
        else:
            return [self.get_nested(dtype, shape[1:])]*max(1, shape[0])

    def get_example_el(self, el):
        if el["type"] == 'tensor':
            return self.get_nested(**el["tensor-spec"])
        return None

    def numpy_input(self, data, input_cfg):
        types = {el["name"]: el["tensor-spec"]["dtype"]
                 for el in input_cfg.to_dict() if "name" in el}
        return {
            key: np.array(val).astype(types[key])
            for key, val in data.items()
        }

    def get_error_message(self, loc, ex):
        self.server.logger.error(ex)
        return HTTPException(status_code=442, detail=[
            {
                "loc": [loc],
                "msg": str(ex),
                "type": str(type(ex))
            }
        ])

    def parse_output(self, data):
        if isinstance(data, pd.DataFrame):
            return data.to_dict(orient="list")
        return {
            key: val.tolist() if isinstance(val, (np.ndarray, np.generic)) else val
            for key, val in data.items()
        }

    def get_schema_string(self, schema):
        return "<ul><li>" + \
            '</li><li>'.join([
                '<b>'+s.name+'</b>: ' +
                str(s).replace('\''+s.name+'\':', '')
                for s in schema.inputs]) + \
            '</li></ul>'

    def info(self):
        return {
            "name": self.name,
            "version": self.version,
            "latest_versions": self.m.latest_versions,
            "input": self.input_schema.to_dict(),
            "output": self.output_schema.to_dict(),
            "description": self.description,
            "creation": self.creation
        }

    def save(self, dirname):
        """
        store the model to a directory
        """
        pass
        # save special references
        server = self.server
        model = self.model
        input_schema_class = self.input_schema_class
        output_schema_class = self.output_schema_class
        input_schema_class_type = self.input_schema_class_type
        output_schema_class_type = self.output_schema_class_type

        # null class references
        self.server = None
        self.model = None
        self.input_schema_class = None
        self.output_schema_class = None
        self.input_schema_class_type = None
        self.output_schema_class_type = None

        # create clean folder
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)

        # copy the mlflow model to a local folder
        extra_model_dir = os.path.join(dirname, "mlflow")
        pathlib.Path(extra_model_dir).mkdir(parents=True, exist_ok=True)
        self.extra_model_dir = _pyfunc_save_model_to_cache(
            self.model_version_source, extra_model_dir)

        # save the metadata
        filename = os.path.join(dirname, "meta.pkl")
        with open(filename, "wb") as f:
            pickle.dump(self, f)

        # reset class references
        self.server = server
        self.model = model
        self.input_schema_class = input_schema_class
        self.output_schema_class = output_schema_class
        self.input_schema_class_type = input_schema_class_type
        self.output_schema_class_type = output_schema_class_type


def load(dirname, server):
    """
    create a basehandler from cache
    """
    filename = os.path.join(dirname, "meta.pkl")
    with open(filename, "rb") as f:
        output = pickle.load(f)

    output.model = _pyfunc_load_model_from_cache(output.extra_model_dir)
    output.server = server
    output.update_schema_classes()
    output.register_route()

    return output


def _pyfunc_save_model_to_cache(model_uri, output_path):
    """
    download the artifact into the cache folder
    """

    local_path = mlflow.pyfunc._download_artifact_from_uri(artifact_uri=model_uri,
                                                           output_path=output_path)
    return local_path


def _pyfunc_load_model_from_cache(local_path):
    """
    load the pyfunc model directly from cache
    """
    from mlflow.models import Model
    from mlflow.pyfunc import PyFuncModel
    from mlflow.models.model import MLMODEL_FILE_NAME
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
    FLAVOR_NAME = "python_function"
    DATA = "data"
    MAIN = "loader_module"
    CODE = "code"

    model_meta = Model.load(os.path.join(local_path, MLMODEL_FILE_NAME))

    conf = model_meta.flavors.get(FLAVOR_NAME)
    if conf is None:
        raise MlflowException(
            'Model does not have the "{flavor_name}" flavor'.format(
                flavor_name=FLAVOR_NAME),
            RESOURCE_DOES_NOT_EXIST,
        )

    data_path = os.path.join(local_path, conf[DATA]) if (
        DATA in conf) else local_path
    if CODE in conf and conf[CODE]:
        code_path = os.path.join(local_path, conf[CODE])
        mlflow.pyfunc.utils._add_code_to_system_path(code_path=code_path)
    model_impl = importlib.import_module(conf[MAIN])._load_pyfunc(data_path)
    return PyFuncModel(model_meta=model_meta, model_impl=model_impl)
