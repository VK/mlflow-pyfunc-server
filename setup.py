import os
from setuptools import setup, find_packages
import mlflow_pyfunc_server


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='mlflow_pyfunc_server',
    url="https://github.com/VK/mlflow-pyfunc-server",
    version=mlflow_pyfunc_server.__version__,
    packages=find_packages(),
    license = "MIT",
    keywords = "mlflow pyfunc",
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    install_requires=[
        'mlflow',
        'uvicorn',
        'fastapi',
        'configargparse',
        'apscheduler'
    ],
    entry_points="""
        [console_scripts]
        mlflow-pyfunc-server=mlflow_pyfunc_server.cli:cli
    """
)
