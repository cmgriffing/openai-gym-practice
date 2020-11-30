#!/bin/bash
trap "exit" INT

python3 ddpg_car_racing_a.py train fiddlin 0 && \
python3 ddpg_car_racing_a.py train fiddlin 1 && \
python3 ddpg_car_racing_a.py train fiddlin 2 && \
python3 ddpg_car_racing_a.py train fiddlin 3 && \
python3 ddpg_car_racing_a.py train fiddlin 4 && \
python3 ddpg_car_racing_a.py train fiddlin 5 && \
python3 ddpg_car_racing_a.py train fiddlin 6 && \
python3 ddpg_car_racing_a.py train fiddlin 7
