# Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
### **Reference:** 
Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning," in Proceedings of Machine Learning Research, 2016.
[Link](http://proceedings.mlr.press/v48/gal16.html?ref=https://githubhelp.com)

### **Summary:** 

* This paper deals with a new theoretical framework which casts dropout training in neural networks as bayesian inference in Gaussian processes.

### **Proposed approach:**

* 

### **Datasets and other stuff experiments:**

* Dataset -- MNIST

* Tasks -- Classification and regression using MNIST dataset.

* 



### **Notes:**

* Bayesian models require high computational cost but they provide a mathematically grounded framework/tools for reasoning about the model uncertainty.

* Deep learning models can be casted as Bayesian models without chaning the model architecture or the optimization.

* Gaussian Process is a probabilistic model.

* The predicitons obtained from softmax output do not represent the confidence of the DNN model.

* Even with high softmax output the model can still be uncertain about its predictions.

* Softmax function provides estimations with unjustified high confidence for the points far from the training data.

* 


### **Important:**
 
* Many deep learning models uses dropout as a way to avoid overfitting.

* Dropout approximately integrates over the model weights <cite> Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation.

* Infinitely wide neural networks of one hidden layer with distributions placed over their weights converge to Gaussian processes.

* Finite Neural networks with distributions placed over the weights have been studied extensively as Bayesian neural networks. <cite> Neal, R M. Bayesian learning for neural networks. PhD
thesis, University of Toronto, 1995.

* Bayesian neural networks also offer robustness to overfitting but they are hard to infer and require high computational costs.

* To represent the uncertainty the number of model parameters are doubled in these models for the same network thereby increasing the computational cost.

* Bayesian networks also need more time to converge.