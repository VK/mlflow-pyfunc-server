import os

from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import mlflow


from apscheduler.schedulers.background import BackgroundScheduler
import datetime


# FastAPI app
from fastapi import FastAPI, Request, HTTPException, Depends
import fastapi
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer
import re
from starlette.routing import RedirectResponse
from starlette.routing import Route as starlette_Route
import urllib
from typing import Dict, List, Any

# internal requests for artifact
import requests

# logging
from fastapi.logger import logger
import logging

from .config import p as cfg
from .basehandler import BaseHandler

__version__ = "0.1.1"

class Server:

    def __init__(self):
        self.config = cfg.parse_known_args()[0]

        # connect to mlflow
        if self.config.mlflow_noverify:
            os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        mlflow.set_tracking_uri(self.config.mlflow)
        if self.config.mlflow_token:
            os.environ["MLFLOW_TRACKING_TOKEN"] = self.config.mlflow_token
        self.client = MlflowClient()

        # init scheduler
        self.scheduler = None
        self.init_scheduler()

        # init FastAPI app
        tags_metadata = [
            {
                "name": "Metadata",
                "description": "Get meta information of the models.",
            },
            {
                "name": "Models",
                "description": "Predict with the models.",
            }
        ]
        self.app = FastAPI(
            redoc_url=None, tags_metadata=tags_metadata, root_path=self.config.basepath)
        self.security = HTTPBearer()

        def custom_openapi():
            if self.app.openapi_schema:
                return self.app.openapi_schema
            openapi_schema = get_openapi(
                title="MLflow PyFunc Server",
                version=__version__,
                description="Serve MLflow PyFunc models",
                routes=self.app.routes,
            )
            self.app.openapi_schema = openapi_schema
            return self.app.openapi_schema
        self.app.openapi = custom_openapi

        # move all routes to the basepath
        if self.config.basepath != "":
            basepath_re = "^" + self.config.basepath.replace("/", "\\/")+"\\/"
            for r in self.app.routes:
                if not isinstance(r, fastapi.routing.APIRoute):
                    r.path_regex = re.compile(r.path_regex.pattern.replace(
                        '^/', basepath_re), re.UNICODE)

        # make docs entrypage
        async def redirect_to_docs(data):
            response = RedirectResponse(url=self.config.basepath+'/docs')
            return response
        if self.config.basepath != "":
            self.app.routes.append(starlette_Route(
                self.config.basepath, redirect_to_docs))
        self.app.routes.append(starlette_Route(
            self.config.basepath+"/", redirect_to_docs))

        # create model list endpoint
        @self.app.get(
            self.config.basepath+'/models',
            tags=["Metadata"],
            description="Get a list of all available models.",
            response_model=List[str]
        )
        async def models():
            return [m for m in self.model_dict]

        # get more information about a model
        @self.app.get(
            self.config.basepath+'/modelinfo/{model}',
            tags=["Metadata"],
            description="Get basic information of a model.",
            response_model=Dict[str, Any]
        )
        async def modelinfo(model: str):
            if model in self.model_dict:
                return self.model_dict[model].info()
            return {}

        # create an error dict endpont
        @self.app.get(
            self.config.basepath+'/errors',
            tags=["Metadata"],
            description="Get a dict of all errors.",
            response_model=Dict[str, Any]
        )
        async def errors():
            return self.error_dict

        # trigger a model refresh
        @self.app.get(
            self.config.basepath+'/refresh',
            tags=["Metadata"],
            description="Trigger the server to update the models."
        )
        async def refresh():
            self.init_scheduler()
            return "OK. Please be patient :)"

        # create model artifact endpoints

        # create the dictionary with all the models
        self.model_dict = {}
        # create a dictionary with all errors
        self.error_dict = {}

        # init logger
        gunicorn_logger = logging.getLogger('gunicorn.error')
        logger.handlers = gunicorn_logger.handlers
        logger.setLevel(gunicorn_logger.level)
        self.logger = logger

    def check_token(self, token):
        """
        Check the request token
        """
        if len(self.config.token) == 0:
            return True
        if token.credentials in self.config.token:
            return True
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    def get_version(self, m):
        prod_models = [
            mm for mm in m.latest_versions if mm.current_stage == "Production"]
        stage_models = [
            mm for mm in m.latest_versions if mm.current_stage == "Staging"]

        if self.config.staging and len(stage_models) > 0:
            return stage_models[0]
        if len(prod_models) > 0:
            return prod_models[0]
        return m.latest_versions[0]

    def update_models(self):
        logger.info(f"Update models")

        all_models = []
        try:
            all_models = self.client.list_registered_models()
            if "server" in self.error_dict:
                del self.error_dict["server"]
        except Exception as ex:
            self.error_dict["server"] = str(ex)

        for m in all_models:

            # get model information
            name = urllib.parse.quote_plus(m.name)

            # get the best version
            model_version = self.get_version(m)

            # if the currently loaded model is already ok
            if name in self.model_dict and model_version.run_id == self.model_dict[name].run_id:
                continue

            # check if the type tag fits
            if len(self.config.tags) > 0 and all([not tt in m.tags.keys() for tt in self.config.tags]):
                continue

            logger.info(f"Update model {name}")

            # delete old route
            for idx2 in [idx for idx, r in enumerate(self.app.routes) if r.path == self.config.basepath+"/"+name]:
                del self.app.routes[idx2]

            # create new handler
            try:
                newmodel = BaseHandler(self, m, model_version)
                self.model_dict[name] = newmodel

                if name in self.error_dict:
                    del self.error_dict[name]
            except Exception as ex:
                self.error_dict["name"] = str(ex)

        self.app.openapi_schema = None

    def load_artifact(self, run_id, artifact_path):
        """ load an artifact from server, used for input_example
        """
        params = {"path": artifact_path, "run_uuid": run_id}
        res = requests.get(self.config.mlflow+"/get-artifact", params=params)
        return res

    def init_scheduler(self):
        """ init and restart the scheduler
        """
        if self.scheduler:
            try:
                self.scheduler.shutdown()
            except:
                pass

        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(func=self.update_models,
                               trigger="interval", seconds=self.config.timer)
        self.scheduler.add_job(func=self.update_models,
                               trigger="date",
                               run_date=datetime.datetime.now()
                               + datetime.timedelta(seconds=10))
        self.scheduler.start()
