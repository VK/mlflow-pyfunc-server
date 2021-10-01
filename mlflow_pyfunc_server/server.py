import pickle
import sys
import os
import pathlib
import shutil


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
import traceback


from .config import p as cfg
from .basehandler import BaseHandler
#from .basehandler import load as load_BaseHandler
import atexit
import glob

__version__ = "0.2.01"
_eureka_client = None


@atexit.register
def cleanup():
    global _eureka_client
    if _eureka_client:
        _eureka_client.stop()


class Server:

    def __init__(self, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = cfg.parse_known_args()[0]

        # create the dictionary with all the models
        self.model_dict = {}
        # create a dictionary with all errors
        self.error_dict = {}

        # init logger
        gunicorn_logger = logging.getLogger('gunicorn.error')
        logger.handlers = gunicorn_logger.handlers
        logger.setLevel(gunicorn_logger.level)
        self.logger = logger
        logger.info(f"Start server version {__version__}")

        # connect to mlflow
        if "databricks" in self.config.mlflow:
            os.environ['DATABRICKS_HOST'] = self.config.mlflow
            os.environ['DATABRICKS_TOKEN'] = self.config.mlflow_token
        if self.config.mlflow_noverify:
            os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        mlflow.set_tracking_uri(self.config.mlflow)
        if self.config.mlflow_token:
            os.environ["MLFLOW_TRACKING_TOKEN"] = self.config.mlflow_token
        self.client = MlflowClient()

        # register at eureka
        if self.config.eureka_server:
            try:
                import py_eureka_client.eureka_client as eureka_client
                global _eureka_client
                if _eureka_client:
                    print("Eureka client already started")
                else:

                    _eureka_client = eureka_client.EurekaClient(
                        eureka_server=self.config.eureka_server,
                        app_name=self.config.app_name,
                        instance_port=self.config.host_port if self.config.host_port else self.config.port,
                        instance_host=self.config.host_name,
                        region=self.config.eureka_region,
                        zone=self.config.eureka_zone,
                    )
                    _eureka_client.start()

            except Exception as ex:
                self.error_dict["eureka"] = {
                    "message": str(ex),
                }

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
            try:
                openapi_schema = get_openapi(
                    title=self.config.title,
                    version=__version__,
                    description=self.config.description,
                    routes=self.app.routes,
                )

                self.app.openapi_schema = openapi_schema
            except:
                pid = os.getpid()
                os.kill(pid, 9)

            return self.app.openapi_schema
        self.app.openapi = custom_openapi

        # move all routes to the basepath
        if self.config.basepath != "":
            basepath_re = self.config.basepath.replace("/", "\\/")+"\\/"
            for r in self.app.routes:
                if not isinstance(r, fastapi.routing.APIRoute):
                    r.path_regex = re.compile(r.path_regex.pattern.replace(
                        '\\/', basepath_re, 1), re.UNICODE)

        # make docs entrypage

        async def redirect_to_docs(data):
            response = RedirectResponse(url=self.config.basepath+'/docs')
            return response
        if self.config.basepath != "":
            self.app.routes.append(starlette_Route(
                self.config.basepath, redirect_to_docs))
        self.app.routes.append(starlette_Route(
            self.config.basepath+"/", redirect_to_docs))
        self.app.routes.append(starlette_Route(
            self.config.basepath+"/info", redirect_to_docs))

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
            response_model=Dict[str, Any],
            include_in_schema=True
        )
        async def errors():
            return self.error_dict

        # trigger a model refresh
        @self.app.get(
            self.config.basepath+'/refresh',
            tags=["Metadata"],
            description="Trigger the server to update the models.",
            include_in_schema=False
        )
        async def refresh():
            self.init_scheduler()
            return "OK. Please be patient :)"

        # dummy route for the  eureka health call
        if self.config.eureka_server:
            @self.app.get(
                self.config.basepath+'/health',
                tags=["Metadata"],
                description="Eureka health call.",
                include_in_schema=False
            )
            async def health():
                return {}

            @self.app.post(
                self.config.basepath+'/health',
                tags=["Metadata"],
                description="Eureka health call.",
                include_in_schema=False
            )
            async def health2():
                return {}

        # check caching
        self.full_cache_dir = os.path.join(
            pathlib.Path().absolute(), self.config.cachedir)
        logger.info(f"Use cachedir: {self.full_cache_dir}")
        pathlib.Path(self.full_cache_dir).mkdir(
            parents=True, exist_ok=True)
        self.update_models_from_cache()

        # init schedulerr
        self.scheduler = None
        self.init_scheduler()

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

    def update_models_from_cache(self):
        logger.info(f"Update models from cache")
        # load all available cached models
        meta_file = os.path.join(self.full_cache_dir, "meta.pkl")
        if os.path.exists(meta_file):
            with open(meta_file, "rb") as f:
                old_config = pickle.load(f)
                for name, m in old_config.items():
                    try:
                        newmodel = BaseHandler(
                            self, m["model"], m["model_version"])
                        newmodel.register_route()
                        self.model_dict[name] = newmodel
                    except Exception as ex:
                        self.error_dict[name] = {
                            "message": str(ex),
                            "type": "Exception"
                        }

        self.app.openapi_schema = None

    def update_models(self):
        logger.info(f"Update models")

        # get a list of the latest models
        all_models = []
        try:
            all_models = self.client.list_registered_models()
            if "server" in self.error_dict:
                del self.error_dict["server"]
        except Exception as ex:
            self.error_dict["server"] = {
                "message": str(ex),
                "type": "Exception"
            }

        for m in all_models:

            # get model information
            name = urllib.parse.quote_plus(m.name)

            # get the best version
            model_version = self.get_version(m)

            # if the currently loaded model is already ok
            if name in self.model_dict and model_version.run_id == self.model_dict[name].run_id:
                # check if the model is still working
                if self.model_dict[name].health():
                    continue

            # check if the type tag fits
            if len(self.config.tags) > 0 and all([not tt in m.tags.keys() for tt in self.config.tags]):
                continue

            logger.info(f"Update model {name}")

            # create new handler
            try:
                newmodel = BaseHandler(self, m, model_version)

                # delete old route
                for idx2 in [idx for idx, r in enumerate(self.app.routes) if r.path == self.config.basepath+"/"+name]:
                    del self.app.routes[idx2]

                # register new route
                newmodel.register_route()

                # keep old model
                if name in self.model_dict:
                    oldmodel = self.model_dict[name]
                else:
                    oldmodel = None

                self.model_dict[name] = newmodel

                if name in self.error_dict:
                    del self.error_dict[name]

                with open(os.path.join(self.full_cache_dir, "meta.pkl"), "wb") as f:
                    pickle.dump({
                        key: {
                            "model": val.m,
                            "model_version": val.model_version
                        }
                        for key, val in self.model_dict.items()
                    }, f)

                # finally stop the old mlflow server
                if oldmodel is not None:
                    oldmodel._stop_server()
                    del oldmodel

            except Exception as ex:
                self.error_dict[name] = {
                    "message": str(ex),
                    "type": "Exception"
                }

        # cleanup non registerd models

        # cleanup cache
        for folder in glob.glob(os.path.join(pathlib.Path().absolute(), self.config.cachedir, "*")):
            if os.path.isdir(folder):
                run_id = os.path.basename(folder)
                usefull = any(
                    [any([v.run_id == run_id for v in m.latest_versions]) for m in all_models])
                if usefull == False:
                    dir_path = pathlib.Path(folder)
                    try:
                        shutil.rmtree(dir_path)
                    except Exception:
                        pass

        self.app.openapi_schema = None

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
                               + datetime.timedelta(seconds=2))
        self.scheduler.start()
