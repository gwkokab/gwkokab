#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

from configparser import ConfigParser
from typing_extensions import Any

import configargparse


cmd_parser = configargparse.ArgParser(config_file_parser_class=configargparse.ConfigparserConfigFileParser)

cmd_parser.add_argument(
    "-c",
    "--config",
    help="config file path",
)


def parse_config(config_path: str) -> dict[str, Any]:
    """Parse the configuration file.

    This function parses the configuration file and returns the
    configurations as a dictionary.

    :param config_path: path to the configuration file
    :return: dictionary containing the configurations
    """
    config = ConfigParser()
    config.read(config_path)
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for key in config[section]:
            config_dict[section][key] = config[section][key]
        if "model" in section:
            config_dict[section]["params"] = eval(config[section]["params"])
            config_dict[section]["config_vars"] = eval(config[section]["config_vars"])
            config_dict[section]["col_names"] = eval(config[section]["col_names"])
            config_dict[section]["error_type"] = config[section]["error_type"]
            config_dict[section]["error_params"] = eval(config[section].get("error_params", r"{}"))

    config_dict["general"]["size"] = int(config["general"]["size"])
    config_dict["general"]["error_size"] = int(config["general"]["error_size"])
    config_dict["general"]["num_realizations"] = int(config["general"]["num_realizations"])
    config_dict["general"]["extra_size"] = int(config["general"].get("extra_size", "1500"))
    config_dict["general"]["extra_error_size"] = int(config["general"].get("extra_error_size", "1000"))

    if "plots" in config:
        config_dict["plots"] = {}
        config_dict["plots"]["injs"] = eval(config["plots"].get("injs", "None"))
        config_dict["plots"]["posts"] = eval(config["plots"].get("posts", "None"))

    return config_dict
