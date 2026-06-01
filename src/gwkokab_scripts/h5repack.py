# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import argparse
import os
import shutil
import subprocess


def main():
    description = r"""Repack an HDF5 file using `h5repack` if available.

It is equivalent to the following shell script:

```sh
#!/bin/bash

file=$1
temp_file="${file}.tmp"

h5repack [options] "$file" "$temp_file"

if [ $? -eq 0 ]; then
    mv "$temp_file" "$file"
else
    rm -f "$temp_file"
fi
```
"""
    parser = argparse.ArgumentParser(
        description=description,
        epilog="Example usage: gwk_h5repack -f GZIP=9 -f SHUF data.h5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "options",
        nargs=argparse.REMAINDER,
        help="Additional options to pass to h5repack (e.g., '-f GZIP=9 -f SHUF')",
    )
    parser.add_argument("file", help="Path to the HDF5 file to repack")
    args = parser.parse_args()

    filename = args.file

    if not shutil.which("h5repack"):
        raise OSError("'h5repack' command not found in PATH.")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' does not exist.")

    temp_file = f"{filename}.tmp"

    cmd = ["h5repack"] + args.options + [filename, temp_file]

    try:
        subprocess.run(cmd, check=True)
        shutil.move(temp_file, filename)
    except (subprocess.CalledProcessError, OSError) as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise e
