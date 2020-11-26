#!/bin/bash
trap "exit" INT

python3 ddpg_car_racing.py train double 0 && \
python3 ddpg_car_racing.py train double 1 && \
python3 ddpg_car_racing.py train double 2 && \
python3 ddpg_car_racing.py train double 3 && \
python3 ddpg_car_racing.py train double 4 && \
python3 ddpg_car_racing.py train double 5 && \
python3 ddpg_car_racing.py train double 6 && \
python3 ddpg_car_racing.py train double 7