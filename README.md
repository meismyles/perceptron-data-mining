# Data Mining Perceptron Algorithm

The main file for this program is ‘perceptron.py’.
All code is contained within this single file.

This program is designed to run with these 4 files:
train.positive (positive train instances)
train.negative (negative train instances)
test.positive (positive test instances)
test.negative (negative test instances)

These should all be within the same directory as ‘perceptron.py’.
If different filenames are to be used the code will need to be updated accordingly.

**Build & Execute Instructions**

To build and execute run command ‘python perceptron.py’ from the command line.

You will be presented with 2 options:
1. Repeatedly train the data for the desired number of iterations and then run a single test iteration.
2. Test the data after each training iteration for the desired number of iterations and then plot a graph of error rate vs number of iterations.

NOTE: Option 2 requires the package ‘matplotlib’ in order to plot the graph.
This often comes pre-packaged within python but if not will require manual installation.
As this is not a requirement of the assignment (since I have already included some plots in my PDF report) I have not included instructions on how to install this.
