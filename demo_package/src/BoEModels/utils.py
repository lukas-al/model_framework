"""Utilities for the demo project"""

import logging
import sys
from typing import Any, Dict, Optional
import mlflow

# Configure global logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class BasicModel(mlflow.pyfunc.PythonModel):
    """
    Basic Model Wrapper class for demo
    """

    def __init__(self, model_config: Dict[str, str]):
        logger.info(msg="BasicModelWrapper instantiated")

        # For experimentation - set the backend uri, tracking uri, etc.
        backend_uri = r"N:\CECD\10. Personal\Lukas Alemu\Future FAME\01. Future Forecast\ml_framework\mlflow"
        mlflow.set_tracking_uri(  # Tracking URI would be a DB workspace - so not set manually in production
            # uri="http://127.0.0.1:8888"
            uri="file:/" + backend_uri
        )

        # Bind the model config to the class instance
        for k, v in model_config.items():
            setattr(self, k, v)

    def register_model(self):
        """Register model"""
        logger.info(msg="Registering model")
        self.model_version = mlflow.register_model(
            model_uri=self.model_name,
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(msg=f"model version registered: {self.model_version}")

    def set_experiment(self, experiment_tags: Optional[Dict[str, Any]] = None):
        """Set the experiment, using config values"""
        self.experiment = mlflow.set_experiment(
            experiment_name=self.experiment_name,
        )
        logger.info(msg=f"Succesffully set experiment: {self.experiment.experiment_id}")

    def log_parameters(self, parameters: Dict[str, Any]):
        """Log parameters to the specific run"""

        if hasattr(self, "run"):
            logger.info(msg=f"Logging parameters to run: {self.run}")
            mlflow.log_params(params=parameters, run_id=self.run.run_id)

        else:
            raise RuntimeError("Need to set run before logging parameters")

    def log_judgement(self):
        """Log judgements as artifacts to the MLFlow instance"""
        raise NotImplementedError

    def set_tags(self):
        """Update the model tags"""
        raise NotImplementedError

    def set_metadata(self):
        """Set additional metadata as required"""
        raise NotImplementedError

    def _fit(self):
        """Replace with model's specific fit method"""
        raise NotImplementedError

    def _predict(self):
        """Replace with model's specific prediction method"""
        raise NotImplementedError

    def predict(self, context, model_input):
        """MLFlow's default entry point for models - adapt as necessary"""
        # Must adhere to this API https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html?highlight=mlflow%20pyfunc#pyfunc-inference-api
        # Can add extra arguments to the input if the model has multiple prediction modes
        raise NotImplementedError
