"""Wrapper for the octave Dynare model"""

import subprocess
from typing import Optional, Dict
from dataclasses import dataclass
from BoEModels.utils import BasicModel

@dataclass
class ModelOutput:
    """Class representing the outputs from the dynare model"""
    M_: Dict  # Structure containing various information about the model.
    options_: Dict  # Structure contains the values of the various options used by Dynare during the computation.
    oo_: Dict  # Structure containing the various results of the computations.
    dataset_: Dict  # A dseries object containing the data used for estimation.
    oo_recursive_: Dict  # Cell array containing the oo_ structures obtained when estimating the model for the different samples when performing recursive estimation and forecasting. The oo_ structure obtained for the sample ranging to the i -th observation is saved in the i -th field. The fields for non-estimated endpoints are empty.
    results_: Dict  # Structure containing results from the model run


class DynareBKK(BasicModel):
    """
    Wrapper class for dynare BKK demo model
    """

    def _predict(self, params: Optional[Dict] = None) -> ModelOutput:
        """
        Run the BKK model.

        A lot of the configuration is done in the BKK.mod file in Dynare/Octave
        """
        #  A lot of this needs to be done via shell / terminal commands.
        # We will need to clean the config before reading it - and probably not use "shell=True"
        # We will need to pipe the stdout through to the python caller, and perhaps some kind of progress bar...
        # We also need to ensure that the dynare matlab subfolder is added to our octave path.
        self.model_path = r"bkk_model/bkk.mod"
        self.script_path = "test.o"

        results = subprocess.run(
            args=[
                "octave",
                "test.o"
            ],
        )

        results = self.format_results(results)
        results["results_"] = {
            "Val1": str(
                        open("bkk_model/bkk/Output/BKK_results.mat", "r")
                    )
        }
        
        model_outputs = ModelOutput(**results)

        return model_outputs

    def format_results(self, res):
        """Format results as returned by Dynare"""
        return {
            "M_": {"Val1": "Lorem Ipsum"},
            "options_": {"Val1": "Lorem Ipsum"},
            "oo_": {"Val1": "Lorem Ipsum"},
            "dataset_": {"Val1": "Lorem Ipsum"},
            "oo_recursive_": {"Val1": "Lorem Ipsum"},
        }

    def _fit(self):
        return NotImplementedError


if __name__ == "__main__":
    print('Running:')
    
    test = DynareBKK(model_config={'foo': "bar"})
    model_outputs = test._predict()
    print(model_outputs)    
