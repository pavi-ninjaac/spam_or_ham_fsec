"""
module containing the config related methods.
"""
"""
This file contains the methods to parsh the config files and read it.
"""

import os
from configparser import ConfigParser

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


class LogisticRegressionConfig(ConfigParserHelper):
    """
    Config parser for the logistic regression config file.
    """
    def __init__(self):
        file_path: str = os.path.join(config.CONFIG_FILE_DIRECTORY, config.LOGISTIC_REGRESSION_CONFIG_NAME)
        super().__init__(file_path)
        self.load()


class PreprocessConfig(ConfigParserHelper):
    """
    Config parser for the logistic regression config file.
    """
    def __init__(self):
        file_path: str = os.path.join(config.CONFIG_FILE_DIRECTORY, config.PREPROCESS_CONFIG_NAME)
        super().__init__(file_path)
        self.load()
