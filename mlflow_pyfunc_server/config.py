import configargparse

p = configargparse.ArgParser(default_config_files=[
                             './mlflow-pyfunc-server.cfg', '~/.mlflow-pyfunc-server.cfg'])

# basic configuration of the server
p.add_argument('--host', type=str, default='localhost', help="hostname")
p.add_argument('-p', '--port', type=int, default=5000, help="port")
p.add_argument('-b', '--basepath', type=str, default='',
               help="basepath of the server")
p.add_argument('-w', '--workers', type=int,
               default=1, help="number of workers")
p.add_argument('-t', '--timer', type=int, default=600,
               help="seconds between calls to refresh the mlflow models")


# model selection
p.add_argument('-s', '--staging', default=False, action="store_true",
               help="flag if the staging model is prefered")
p.add_argument('--tags', type=str, default=[],
               help="a list of allowed tags", action="append")

# secure server calls by a token
p.add_argument('--token', type=str, default=[],
               help="a list of allowed tokens", action="append")

# parameters to connect to a mlflow server
p.add_argument('--mlflow', type=str,
               default="http://localhost:4040", help="mlflow server location")
p.add_argument('--mlflow-token', type=str,
               default=None, help="mlflow server token")
p.add_argument('--mlflow-noverify',  default=False,
               action="store_true", help="ignore ssl cert problems")

# local caching
# p.add_argument('--cachedir', type=str, default='./cache',
#                help="folder to keep the local cache")
