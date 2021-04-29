import os
import mlflow
from mlflow.types.schema import Schema
from pydantic import BaseModel
import numpy as np
import pandas as pd
from fastapi import HTTPException, Depends
from datetime import datetime


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
        self.server = server
        self.m = m

        try:
            model = mlflow.pyfunc.load_model(model_version.source)
        except:
            model = None

        try:
            input_schema = model.metadata.get_input_schema()
        except:
            input_schema = None

        try:
            output_schema = model.metadata.get_output_schema()
        except:
            output_schema = None

        if not input_schema:
            input_schema = Schema([])
        if not output_schema:
            output_schema = Schema([])

        try:
            res = self.server.load_artifact(
                model_version.run_id,
                os.path.join(
                    model.metadata.artifact_path,
                    model.metadata.saved_input_example_info[
                        'artifact_path']
                ))
            input_example_data = res.json()['inputs']
        except:
            input_example_data = {}

        if input_schema:
            input_schema_class = type(
                m.name+"-input",
                (BaseModel, ),
                {el["name"]:
                 input_example_data[el["name"]]
                 if el["name"] in input_example_data else
                 self.get_example_el(el) for el in input_schema.to_dict()})
        else:
            input_schema_class = None

        try:
            np_input = self.numpy_input(input_example_data, input_schema)
            output_example_data = model.predict(np_input)
            output_example_data = {k: v.tolist()
                                   for k, v in output_example_data.items()}
        except:
            output_example_data = {}

        if output_schema:
            output_schema_class = type(
                m.name+"-output",
                (BaseModel, ),
                {el["name"]:
                 output_example_data[el["name"]]
                 if el["name"] in output_example_data else
                 self.get_example_el(el) for el in output_schema.to_dict()})
        else:
            output_schema_class = None

        self.version = model_version.version
        self.source = model_version.source
        self.run_id = model_version.run_id
        self.model = model
        self.input_schema = input_schema if input_schema else {"inputs": []}
        self.output_schema = output_schema if output_schema else {"inputs": []}
        self.input_schema_class = input_schema_class() if input_schema_class else None
        self.output_schema_class = output_schema_class() if output_schema_class else None
        self.description = m.description
        timestamp = model_version.creation_timestamp/1000
        self.creation = datetime.fromtimestamp(
            timestamp).strftime('%Y-%m-%d %H:%M')

        long_description = f"""{m.description}

<b>Input Schema:</b> {self.get_schema_string(input_schema)} <br/>
<b>Output Schema:</b> {self.get_schema_string(output_schema)}<br/>
<b>Version: </b> {self.get_version_link(m.name, model_version)}<br/>
<b>Run: </b> {self.get_experiment_link(m.name, model_version)}<br/>
<b>Creation: </b> {self.creation}
        """

        if len(self.server.config.token) > 0:
            @self.server.app.post(
                self.server.config.basepath+'/'+m.name,
                description=long_description,
                name=m.name, tags=["Models"],
                response_model=output_schema_class)
            async def func(
                data: input_schema_class,
                token: str = Depends(self.server.security)
            ):
                self.server.check_token(token)
                return self.apply_model(data)
        else:
            @self.server.app.post(
                self.server.config.basepath+'/'+m.name,
                description=long_description,
                name=m.name, tags=["Models"],
                response_model=output_schema_class
            )
            async def func(data: input_schema_class):
                return self.apply_model(data)

    def apply_model(self, data):
        try:
            np_input = self.numpy_input(data.__dict__, self.input_schema)
        except Exception as ex:
            raise self.get_error_message("Parse input error", ex)

        try:
            output = self.model.predict(np_input)
        except Exception as ex:
            raise self.get_error_message("Model prediction error", ex)

        try:
            output = self.parse_output(output)
        except Exception as ex:
            raise self.get_error_message("Parse output error", ex)

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
                 for el in input_cfg.to_dict()}
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
            "name": self.m.name,
            "version": self.version,
            "latest_versions": self.m.latest_versions,
            "input": self.input_schema,
            "output": self.output_schema,
            "description": self.description,
            "creation": self.creation
        }
