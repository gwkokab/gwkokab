#  Copyright 2023 The Jaxtro Authors
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

from .utils import cmd_parser, parse_config, PopulationGenerator


def main():
    args = cmd_parser.parse_args()
    configuration_dict = parse_config(args.my_config)

    general = configuration_dict["general"]
    models = [
        configuration_dict.get("mass_model", None),
        configuration_dict.get("spin_model", None),
    ]

    pg = PopulationGenerator(general=general, models=models)
    pg.generate()
