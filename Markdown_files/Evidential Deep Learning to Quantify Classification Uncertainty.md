# Evidential Deep Learning to Quantify Classification Uncertainty


### **Reference:**
M. Sensoy, L. Kaplan and M. Kandemir, "Evidential Deep Learning to Quantify
Classification Uncertainty," in Advances in Neural Information Processing Systems, 2018. [Link](https://proceedings.mlr.press/v119/van-amersfoort20a.html)

### **Summary**

* In this paper authors have proposed an explicit modelling of Bayesian neural networks using
the theory of subjective logic.

* Authors have approached the problem of uncertainty estimation in the Theory of Evidence
perspective.



### **Proposed approach**

* In this paper, authors have designed a predictive distribution for classification problem by
placing a Dirichlet distribution over the class probabilities and by assigning neural network
outputs to these parameters.

* For this they have used The Dempster–Shafer Theory of Evidence (DST) which is a
generalization of the Bayesian theory to subjective probabilities 

* In this paper, authors have designed a predictive distribution for classification problem by placing a Dirichlet distribution over the class probabilities and by assigning neural network outputs to these parameters.
* In this paper authors have used The Dempster–Shafer Theory of Evidence (DST) which is a generalization of the Bayesian theory to subjective probabilities.

* Authors have replaced the parameter set with the parameters of a Dirichlet density to represent the predictions of the model as a distribution over possible softmax outputs.

* According to the authors, LeNet architecture was trained for MNIST utilizing 20 and 50 filters
of size 5x5 at the first and second convolutional layers, respectively, and 500 hidden units for the fully connected layer. 


### **Datasets and experiments:**

* DNN model - LeNet with ReLU activation function.
* Dataset
    * MNIST - Training
    * notMNIST (letters)-testing
    * CIFAR 10
        * First five training
        * Last five testing
* Loss function- cross entropy loss 
* Experiments
    * L2
    * Dropout
    * Deep Ensemble
    * Bayesian Neural networks
    * Structured variational inference
    * Evidential deep learning



* Authors have replaced the softmax layer with a ReLU activation function so that there will not be any negative outputs.

* These outputs are taken as an evidence vector for the predicted Diricchlet distribution.

* Unlike other methods the EDL doesnt use entropy to measure the uncertainty. It directly quantifies the uncertainty.


### **Notes**
* Bayesian neural networks estimate prediction uncertainty by approximating the moments of the posterior predictive distribution.
* Authors have interpreted the softmax output as the parameter set of categorical distribution.
* GPs do not have deterministic or stochastic model parameters because they are non-parametric models. The variance of GPs' predictions may be estimated in closed form, which is a considerable advantage in uncertainty estimation.

* Dirichlet distribution with zero total evidence corresponds to a uniform distribution and indicates a total uncertainty of one. This can be achieved by Kullback_Leiber(KL) divergence term into the loss function.

* Some of the approaches uses entropy to measure the uncertainty of the predictions.

### **Important**

* Softmax operator converts continuous activations of the output layer to class probabilities.

* These class probabilities of the model can be eventually interpreted as a multinomial distribution.

* Softmax function provides only a point estimate for the class probabilities of a sample and does not provide the associated uncertainty.

* In neural networks minimizing the negative log-likelihood is prefered for the computational convenience which is widely known as the cross-entropy loss.

* Maximum Likelihood Estimation is not capable of inferring the predictive distribution variance.

* The Dempster–Shafer Theory of Evidence (DST) is a generalization of the Bayesian theory to subjective probabilities.

* The outputs of a neural network classifier is a probability assignment over the classes for each sample.

* Dirichlet distribution parameterized over the evidence represents the density of each such probability assignment. Thus it models the second-order probabilities and uncertaiinty.

