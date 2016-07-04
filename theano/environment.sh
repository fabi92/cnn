#!/bin/bash

BASE=$(dirname "$BASH_SOURCE")
echo "$BASE"
DIR=`pwd`
BASEDIR="$DIR/$BASE"

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export PYTHONPATH="$BASEDIR"/tools:"$BASEDIR"/base:"$BASEDIR"/base/layers:"$BASEDIR"/base/nets:$PYTHONPATH