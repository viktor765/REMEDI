# REMEDI entropy estimation

This is the code for the entropy estimation part of the article REMEDI, containing experiments on the triangle and two moons, ball, and hypercube datasets.

## Instructions

To replicate these experiments, run the file 'experiments/final.py'. This can also be used with command line arguments to limit the number of runs as well as choosing a specific experiment. To summarize the experiments and produce plots, run 'experiments/post_plots.py', with the adequate path inserted. Alternatively, if you do not have the training run logs, you may use the third last cell of the script and change 'load_from_csv' to true, to import a supplied 'avg_big.csv' and 'avg_last_big.csv', after running the initial imports. For the generative experiments, run 'experiments/generative.py'.

The code is tested with Pytorch 2.0.1 and Numpy 1.24.3. See import statements for further requirements.
