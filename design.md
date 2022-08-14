# Wrist vision runtime

### runtime design

main runtime is all high-level function calling, all other intensive calculation is packaged in methods

hyperparameters are all kept in JSON

camera streaming is done with realsense package, processing is done with opencv

modes are determined first in `main`, toggled with call flags
- record
- debug

camera resolution is determined in json

if runtime loses track of target, we use last known position
- in the future we will be using kalman filter