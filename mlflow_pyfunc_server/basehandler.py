import os
import shutil
import mlflow
from mlflow.types.schema import Schema
from mlflow.types.schema import TensorSpec
from mlflow.exceptions import MlflowException as _MlflowException
from pydantic import BaseModel as _BaseModel
import numpy as np
import pandas as pd
from fastapi import HTTPException, Depends
from datetime import datetime


import time
import pathlib

import requests
import subprocess


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

    def __init__(self, server,  m, model_version):
        # parent server
        self.server = server

        # the local mlflow server
        self.__serve_logfile = None
        self.__serve_proc = None
        self.m = m
        self.name = m.name
        self.model_version = model_version
        self.model_version_source = (model_version.source)
        self.run_id = model_version.run_id
        self.work_folder = os.path.join(server.full_cache_dir, self.run_id)
        self.model_folder = os.path.join(
            self.work_folder, model_version.source.split("/")[-1])
        pathlib.Path(self.work_folder).mkdir(parents=True, exist_ok=True)

        if os.path.exists(self.model_folder) == False:
            try:
                self._setup_model()
            except Exception as ex:
                shutil.rmtree(self.model_folder)
                raise ex

        self.port = self._get_free_port()
        self._start_server()

        from mlflow.models.model import Model as _Model
        metadata = _Model.load(os.path.join(
            self.model_folder, mlflow.pyfunc.MLMODEL_FILE_NAME))

        health = False
        checkcount = False
        while health == False and checkcount < 5:
            health = self.health()
            checkcount += 1
            time.sleep(1.0)

        if health == False:
            self._stop_server()
            raise Exception("Model not working with example input!")

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
                self.long_description += f"<b>Input Schema:</b> {self._get_schema_string(input_schema)} <br/>\n"
        except:
            pass
        try:
            if len(output_schema.inputs) > 0:
                self.long_description += f"<b>Output Schema:</b> {self._get_schema_string(output_schema)}<br/>\n"
        except:
            pass

        self.long_description += f"""
<b>Version: </b> {self._get_version_link(m.name, model_version)}<br/>
<b>Run: </b> {self._get_experiment_link(m.name, model_version)}<br/>
<b>Creation: </b> {self.creation}
        """

        self._update_schema_classes()

    def _setup_model(self):
        """
        Setup the local environment to serve the mlflow model
        """
        from mlflow.pyfunc import _download_artifact_from_uri
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

        if result_env.returncode != 0:
            raise Exception("Unable to setup env")

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

        if result_pip.returncode != 0:
            raise Exception("Unable to install requirements.txt")

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
                if result_pip.returncode != 0:
                    raise Exception(
                        f"Unable to install requirements of library {os.path.basename(lib_folder)}")
                setup_logfile.flush()

            # install the libs
            result_lib = subprocess.run([
                os.path.join(self.model_folder, "./env/Scripts/python"),
                "./setup.py",
                "build",
                "install"], stdout=setup_logfile, stderr=setup_logfile,
                cwd=lib_folder)
            setup_logfile.flush()

            if result_lib.returncode != 0:
                raise Exception(
                    f"Unable to install library {os.path.basename(lib_folder)}")

        # close the logging
        setup_logfile.close()

    def _get_free_port(self, host='127.0.0.1'):
        """
        use socket to search for a free port
        """
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((host, 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def _start_server(self):
        """
        start the mlflow server
        """
        # create a logfile
        filename = os.path.join(self.work_folder,
                                f"{os.path.basename(self.model_folder)}_servelog_{self.port}.txt")

        import logging
        import logging.handlers as handlers

        self.__serve_logfile = logging.getLogger(str(self.port))
        self.__serve_logfile.setLevel(logging.INFO)

        # setup the logging format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logHandler = handlers.RotatingFileHandler(
            filename, mode="a", maxBytes=10000000, backupCount=2,
        )

        logHandler.setLevel(logging.INFO)
        logHandler.setFormatter(formatter)

        self.__serve_logfile.addHandler(logHandler)

        cmd = f"""
        {os.path.join(self.model_folder, './env/Scripts/activate')}
        {os.path.join(self.model_folder, './env/Scripts/mlflow.exe')} models serve -m . --no-conda -w 1 --port {self.port}
        
        """

        self.__serve_proc = subprocess.Popen(
            """cmd.exe""", cwd=self.model_folder,
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            text=True,
            shell=True
        )

        self.__serve_proc.stdin.write(cmd)
        self.__serve_proc.stdin.flush()

    def _stop_server(self):
        """
        stop the mlflow server
        """
        try:
            self.__serve_proc.kill()
        except:
            pass
        try:
            self.__serve_logfile.error("Killed server")
        except:
            pass

    def health(self):
        """
        Internal healthcheck.

        Uses the input example to compute a result.
        """
        inp = self._get_input_example()

        try:
            res = self._predict(inp)
            return res.ok
        except:
            return False

    def _predict(self, inp):
        """
        Just call the created server to compute the result
        """
        self.__serve_logfile.info(inp)
        res = requests.post(
            f"http://localhost:{self.port}/invocations", json=inp)
        if res.ok:
            self.__serve_logfile.info("OK")
        else:
            self.__serve_logfile.info(res.content)
        return res

    def _get_input_example(self):
        """
        Usually a mlflow model has an input_example.json.
        This is used to prepare the webgui + health checks.
        """
        import json
        try:
            with open(os.path.join(self.model_folder, "input_example.json"), "r") as f:
                inp = json.load(f)

            return inp
        except:
            return {}

    def _update_schema_classes(self):
        """
        extract schema information
        """

        if self.input_schema:
            input_schema_class = type(
                self.name+"-input",
                (_BaseModel, ),
                {el["name"]:
                 self.input_example_data[el["name"]]
                 if el["name"] in self.input_example_data else
                 self._get_example_el(el) for el in self.input_schema.to_dict() if "name" in el})
        else:
            input_schema_class = None

        try:
            input_example = self._get_input_example()
            res = self._predict(input_example)
            if res.ok:
                output_example_data = res.json()
                output_example_data.update(
                    {"x__version": [int(self.version)],
                     "x__mlflow_id": [self.run_id]}
                )
            else:
                output_example_data = {}
        except:
            output_example_data = {}

        if self.output_schema:
            output_schema_class = type(
                self.name+"-output",
                (_BaseModel, ),
                {el["name"]:
                 output_example_data[el["name"]]
                 if el["name"] in output_example_data else
                 self._get_example_el(el) for el in self.output_schema.to_dict()})
        else:
            output_schema_class = None

        self.input_schema_class = input_schema_class() if input_schema_class else None
        self.output_schema_class = output_schema_class() if output_schema_class else None
        self.input_schema_class_type = input_schema_class
        self.output_schema_class_type = output_schema_class

    def register_route(self):
        """
        setup the parent server route
        """
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
        """
        compute the model result
        """
        # create a input array
        try:
            input_array = {
                key: val
                for key, val in data.__dict__.items()
            }
        except Exception as ex:
            raise self._get_error_message("Model input error", ex)

        try:
            res = self._predict({"inputs": input_array})
        except Exception as ex:
            raise self._get_error_message("Model call error", ex)

        if res.ok:
            model_output = res.json()
            model_output.update(
                {
                    "x__version": [int(self.version)],
                    "x__mlflow_id": [self.run_id]
                }
            )
        else:
            raise self._get_error_message(
                "Model prediction error", _MlflowException(res.json()["message"]))

        try:
            output = self._parse_output(model_output)
        except Exception as ex:
            raise self._get_error_message("Parse output error", ex)

        return output

    def _get_version_link(self, name, model_version):
        return f"{model_version.version}"

    def _get_experiment_link(self, name, model_version):
        return f"{model_version.run_id}"

    def _get_nested(self, dtype, shape):
        if len(shape) == 1:
            return [self.dtype_sample[dtype]]*shape[0]
        else:
            return [self._get_nested(dtype, shape[1:])]*max(1, shape[0])

    def _get_example_el(self, el):
        if el["type"] == 'tensor':
            return self._get_nested(**el["tensor-spec"])
        return None

    def _get_error_message(self, loc, ex):
        self.server.logger.error(ex)
        return HTTPException(status_code=442, detail=[
            {
                "loc": [loc],
                "msg": str(ex),
                "type": str(type(ex))
            }
        ])

    def _parse_output(self, data):
        if isinstance(data, pd.DataFrame):
            return data.to_dict(orient="list")
        return {
            key: val.tolist() if isinstance(val, (np.ndarray, np.generic)) else val
            for key, val in data.items()
        }

    def _get_schema_string(self, schema):
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
