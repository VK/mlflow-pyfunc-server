# MLflow PyFunc Server

A simple server to publish several mlflow pyfunc models quickly in the private network.

Settings can be defined in a config file (./mlflow-pyfunc-server.cfg, ~/.mlflow-pyfunc-server.cfg) or at startup:
```
mlflow-pyfunc-server --host 0.0.0.0 -w 2 --mlflow https://mlflow:1234
```
![Screenshot](/media/screenshot.png?raw=true "Optional Title")

## Develop
```
$ py -m venv env
$ .\env\Scripts\activate
$ pip install -r requirements.txt
```
