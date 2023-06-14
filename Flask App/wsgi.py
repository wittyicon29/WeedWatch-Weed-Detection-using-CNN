#Waitress Server to deploy the model to the server with port 5000

from waitress import serve
from local_deployment_app import app

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)
