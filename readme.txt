
********************************  Only some notes, detail answers are in the pdf file **************************************
SGDLinearRegression
The stochastic gradient descent won't perform well in a step-size of 0.01. Its fluctuation is too large so that its graph doesn't look like a convergence. If I tweaked its learning rate/step-size to 0.001, it works much better. Both graphs for step-size=0.01 and 0.001 are included. 

BatchGDLinearRegression
As instructed by TA's in lab, this algorithm is implemented in a slightly different way as it was on class notes.
The Batch Gradient Descent ends after 1000 epoches, even though it usually takes ~100 epoches to converge.
In class notes, the Batch GD ends if the cost is no longer descreasing within a threshold.

The file plot_comparison_graph.py and some helper functions are only created for ploting graphs. You might need to manually tweak some parameters to plot desiring graphs.

Please check my PDF submission for detail answers to other questions.