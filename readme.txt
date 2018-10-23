bgd的line search每次将a还原吗
bgd的cost变化多少，收敛于多少，一般多少次descent
bgd的runtime比sgd的快很多，结果差很多，是陷在local minimum了吗？
哪里设ms=1


How to report these xxx versus xxx? Draw a graph or list statistics?
Where do you change parameter/hyperparameters when testing our code? In parameters, or in the function itself? Are you going to use the same name?

What is the error reported for question e) f) and Bonus questions? The average error provided in the template code?
Will providing a graph that the error versus epoch/runtime suffice for what asked in question f)
What do you mean by runtime? 
Is this error or just cost?

For error, I used l2-error if not specified in the question.
For batch gradient descent in question f), since I was asked to compare it with a SGD of 1000 epoches, my implementation of batch gradient descent terminates after 1000 epoches, although it only need ~20 descents to reach the minimum point. Its implementation is therefore different with the one on class notes.
