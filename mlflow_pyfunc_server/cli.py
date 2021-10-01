#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mlflow_pyfunc_server
import uvicorn


def cli():
    config = mlflow_pyfunc_server.config.parse_known_args()[0]

    uvicorn.run("mlflow_pyfunc_server.serverapp:app",
                port=config.port,
                host=config.host,
                workers=config.workers,
                )


if __name__ == '__main__':
    cli()
