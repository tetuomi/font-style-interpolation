#!/bin/bash
docker build \
    --pull \
    --rm \
    -f "Dockerbase-main/Dockerfile" \
    --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg USER=ahi --build-arg PASSWORD=ahi \
    -t \
    tetta:latest "."
