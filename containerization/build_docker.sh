#!/bin/bash

git rev-parse --short HEAD > ./git.hash

if [ "$1" = "--experimental" ]; then
    docker build . -t "deepmi/lit:dev" -f ./containerization/Dockerfile_experimental
    exit 0
fi

docker build . -t "deepmi/lit:dev" -f ./containerization/Dockerfile

rm ./git.hash
