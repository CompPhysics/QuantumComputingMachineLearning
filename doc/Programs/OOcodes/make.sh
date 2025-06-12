#!/bin/sh
set -x

function system {
  "$@"
  if [ $? -ne 0 ]; then
    echo "make.sh: unsuccessful command $@"
    echo "abort!"
    exit 1
  fi
}

if [ $# -eq 0 ]; then
echo 'bash make.sh slides1|slides2'
exit 1
fi

name=$1
rm -f *.tar.gz

opt="--encoding=utf-8"
# Note: Makefile examples contain constructions like ${PROG} which
# looks like Mako constructions, but they are not. Use --no_mako
# to turn off Mako processing.
opt="--no_mako"

rm -f *.aux



# IPython notebook
system doconce format ipynb $name $opt


