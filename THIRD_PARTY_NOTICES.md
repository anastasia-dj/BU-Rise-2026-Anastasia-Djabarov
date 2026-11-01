This project is based on "Randomized Kinodynamic RRT for a Car-like Robot" by Angelo Espinoza (https://github.com/AngeloEspinoza/kinodynamic-rrt-car-like-robot). environment.py, robot.py, RRT.py, kinodynamic_RRT.py and the images in images/ started from that repository and were modified. Its license notice is reproduced below as required by the MIT License.

MIT License

Copyright (c) [2023] [Angelo Espinoza]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


The tabular Q-learning in qlearning.py has been adapted from Hugging Face Deep RL Course, Unit 2 (https://github.com/huggingface/deep-rl-class), which is under the Apache License, Version 2.0. The code within this repository has been modified as follows: It has been restructured under a Qlearning class and extended with convergence detection, a plot of Q-value history and a custom robot environment.