"""
module containing the config related methods.
"""
"""
This file contains the methods to parsh the config files and read it.
"""

import os
from configparser import ConfigParser
from dataclasses import dataclass

from constants import config


class ConfigParserHelper:
    """
    Helper class to make job easier for config parser methods.
    """
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.config = ConfigParser()

    def load(self) -> None:
        """
        Load the config file into the config parsher object.
        """
        self.config.read(self.file_path)

    def has_section(self, section: str) -> bool:
        """
        Check the config file has the given section or not.
        """
        if section in self.config.sections():
            return True
        return False

    def get_config_items(self, section_name: str) -> dict[str, str] | None:
        """
        Get the parameter values under the section name.
        """
        if self.has_section(section_name):
            return dict(self.config.items(section_name))
        return None


class PreprocessConfig(ConfigParserHelper):
    """
    Config parser for the logistic regression config file.
    """
    def __init__(self):
        file_path: str = os.path.join(config.CONFIG_FILE_DIRECTORY, config.PREPROCESS_CONFIG_NAME)
        super().__init__(file_path)
        self.load()


@dataclass
class LogisticRegressionConfig:
    """
    Model parameters for logistic regression.
    """
    C: float = 0.006739078892927555
    max_iter: int = 483
    solver: str = "liblinear"
    penalty: str = "l2"
    class_weight: str = "balanced"


@dataclass
class SVCConfig:
    """
    Model parameters for SVM classification.
    """
    C: float = 0.7536299183832335
    kernel: str =  'sigmoid'
    gamma: float = 0.38769439007402395
    degree: int = 5
    class_weight: str ="balanced"


@dataclass
class RandomForestConfig:
    """
    Model parameters for Random Forest classification.
    """
    n_estimators: int =  75
    max_depth: int = 3
    min_samples_split:int = 9
    min_samples_leaf:int =  10
    max_features: str = "sqrt"
    random_state: int = 42
    class_weight: str = "balanced"


@dataclass
class NueralNetworkConfig:
    """
    Model parameters for Nueral Network classification.
    """
    hidden_layers: int = 1
