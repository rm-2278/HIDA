#!/bin/bash
conda activate hieros
pip install -r requirements.txt
python3 -c "
import embodied
from embodied.envs import pinpad
env = pinpad.PinPad('three', length=100)
print('Environment created successfully')
obs = env.step({'action': 0, 'reset': True})
print('Initial observation keys:', obs.keys())
print('First step successful')
"
