# Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
### **Reference:** 
Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning," in Proceedings of Machine Learning Research, 2016.
[Link](http://proceedings.mlr.press/v48/gal16.html?ref=https://githubhelp.com)

### **Summary:** 

* This paper deals with a new theoretical framework which casts dropout training in neural networks as bayesian inference in Gaussian processes.

### **Proposed approach:**

* Authors have shown that a neural network with arbitary depth and non-linearities, with dropout applied before every weight layer is mathematically equivalent to an approximation to the probabilistic deep Gaussian process.

* Authors have used L2 regularisation weighted by some weight decay lambda.

* Except for the last layer, authors have sampled binary variables for every input point and for every network unit in each layer.

* For a probability of pi for layer i, each binary variable represents a value of 1.

* A unit will be dropped for a given input if the corresponding binary value is zero. i.e the value of the neuron output will be set to zero.

* The same values are also used during the backpropogation and the parameters will be updated respectively.

* To approximate the intractable posterior, authors have used a variational distribution over the matrices in which the columns are randomly set to zero.

* This distribution is similar to bernoulli distribution.

* The variational distribution is highly multi model and it induces strong correlations over the rows of the weight matrix.

* To get an unbiased estimate, authors have minimized the KL divergence between the approximate posterior of variational distribution and the posterior of the full deep gaussian process and aprroximated each term by MOnte Carlo integration with a single sample.
    #### **Obtaining model uncertainty**
    * Authors have performed a moment matching and estimated the first two moments of the predictive distribution empirically.
    
    * They have sampled a T sets of vectors of predictions from the Bernoulli distribution and refered the Monte Carlo estimate as MC dropout.

    * This process is also equivalent to performing T stochastic forward passes through the network and averaging the results which is also known as model averaging.
    
    * This process can also be interpreted as an ensemble technique (But it is not exactly an ensemble technique.)

    *  Authors have estimated the predictive log-likelihood by monte carlo integration. This estimate represents how well the model fits the mean and uncertainty.

    * For estimating the predictive mean and predictive uncertainty authors have collected the results of stochastic forward passes though the model but the neural network model is not changed. 

### **Datasets and experiments:**

* Datasets
    * MNIST - Classification.
    * Atmospheric CO2 concentrations - Regression.
    * Reconstucted solar irrdiance dataset - Regression.
    * All the datasets are centered and normalized.

* Tasks -- Classification and regression.

* Authors have compared the uncertainties obtained from different model architectures and non-linearitites.

* Authors claim that using dropout's uncertainty they obtained a considerable improvement in the predictive log-likelihood and RMSE compared to the existing state of the art methods.

* For regression tasks authors have used ReLU and TanH nonlinearities along with a dropout probability of either 0.1 or 0.2

* Author says that in the experiments even if the model has incorrect predictive mean, the increased standard deviation will express the model's uncertainty in for the regression tasks.

* The uncertainty was increasing far from the data for the ReLU models while for the TanH model it remained bounded.

* Models with relu activation function performed better.

* For the classification task authors have trained LeNet convolutional neural network with dropout applied before last fully connected inner-product layer.

* Authors have evaluated the trained model on a continuously rotated image of digit one and they have scattered 100 stochastic forward passes of the softmax input and the softmax output for each of the top classes.


### **Notes:**

* Bayesian models require high computational cost but they provide a mathematically grounded framework/tools for reasoning about the model uncertainty.

* Deep learning models can be casted as Bayesian models without chaning the model architecture or the optimization.

* Gaussian Process is a probabilistic model.

* The predicitons obtained from softmax output do not represent the confidence of the DNN model.

* Even with high softmax output the model can still be uncertain about its predictions.

* Softmax function provides estimations with unjustified high confidence for the points far from the training data.

### **Important:**
 
* Dropout can be used to avoid overfitting of the deep learning models.

* Dropout approximately integrates over the model weights <cite> Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation.

* Infinitely wide neural networks of one hidden layer with distributions placed over their weights converge to Gaussian processes.

* Finite Neural networks with distributions placed over the weights have been studied extensively as Bayesian neural networks. <cite> Neal, R M. Bayesian learning for neural networks. PhD
thesis, University of Toronto, 1995.

* Bayesian neural networks also offer robustness to overfitting but they are hard to infer and require high computational costs.

* To represent the uncertainty the number of model parameters are doubled in these models for the same network thereby increasing the computational cost.

* Bayesian networks also need more time to converge.

* Deep Gaussian process is a powerful tool in statistics and it allows to model distributions over functions.