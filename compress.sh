#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

if [ ! -d $1 ]; then
    echo "Directory $1 does not exist."
    exit 2
fi

if [ ! -r $1 ]; then
    echo "Directory $1 is not readable."
    exit 3
fi

current_dir=$(pwd)
for dir in $(ls -d $1/*); do
    cd $dir
    folder_name=$(basename $dir)
    zip -rq $folder_name.zip *
    cd $current_dir
    folder_name=$(basename $dir)
    mv $dir/$folder_name.zip $1
done