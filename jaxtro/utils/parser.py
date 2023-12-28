# Copyright 2023 The Jaxtro Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from configparser import ConfigParser

import configargparse

cmd_parser = configargparse.ArgParser(config_file_parser_class=configargparse.ConfigparserConfigFileParser)
cmd_parser.add(
    '-c',
    '--my-config',
    help='config file path',
)


def parse_config(config_path: str) -> dict:
    """Parse config file and return config object."""
    config = ConfigParser()
    config.read(config_path)
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for key in config[section]:
            config_dict[section][key] = config[section][key]
        if 'model' in section:
            config_dict[section]['params'] = eval(config[section]['params'])
            config_dict[section]['config_vars'] = eval(config[section]['config_vars'])
            config_dict[section]['col_names'] = eval(config[section]['col_names'])

    config_dict['general']['size'] = int(config['general']['size'])
    config_dict['general']['error_scale'] = float(config['general']['error_scale'])
    config_dict['general']['error_size'] = int(config['general']['error_size'])

    return config_dict
