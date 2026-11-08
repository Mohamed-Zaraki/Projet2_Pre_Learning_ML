import os
from flask import Flask
from app.config import Config

def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config.from_object(Config)

    from app.controllers.main_controller import main_bp
    app.register_blueprint(main_bp)

    return app
