"""Basic model for testing purposes"""

import logging
from typing import Dict, Optional, Any, Union
import sys
import mlflow

# Configure global logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

DEMO_CONFIG = {
    'experiment_name': 'LUKAS_TEST_1',
    'model_name': 'TEST1',
    'model_author': 'Lukas Alemu',
    'model_owner': 'Lukas Alemu',
    'model_description': """Lukas testing model""",
    'model_parameters': {
        'param1': 1,
        'param2': 2,
        'param3': True,
    },
    'code_paths': None
}


class BasicModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Basic Model Wrapper class
    """
    def __init__(self, model_config: Dict[str, str]):
        logger.info(msg="BasicModelWrapper instantiated")

        # Bind the metadata
        self.experiment_name = model_config['experiment_name']
        self.model_name = model_config['model_name']
        self.model_author = model_config['model_author']
        self.model_owner = model_config['model_owner']
        self.model_description = model_config['model_description']
        # Bind model parameters
        self.model_params = {
            **model_config['model_parameters']
        }
        # Add any paths to directories with code relevant to this model
        self.code_paths = model_config['code_paths']

      
    def _fit(self):
        """Place the model's specific fit method here"""
        raise NotImplementedError

    def _predict(self):
        """Place the model's specific prediction method here"""
        raise NotImplementedError

    def predict(self):
        """MLFlow's required entry point for models"""
        # Must adhere to this API https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html?highlight=mlflow%20pyfunc#pyfunc-inference-api
        # For example...:
        self._fit()
        self._predict()
        raise NotImplementedError

    
        

if __name__ == "__main__":
    model = BasicModelWrapper(DEMO_CONFIG)
