#!/usr/bin/env bash
pip install keras # --user
pip install tensorflow-gpu # --user
pip install h5py # --user
#docker build --no-cache=true -t tf-model-server model_server
docker build -t tf-model-server model_server
