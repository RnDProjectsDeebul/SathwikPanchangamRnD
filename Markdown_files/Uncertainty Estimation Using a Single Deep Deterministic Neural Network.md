# Uncertainty Estimation Using a Single Deep Deterministic Neural Network


### **Reference:**
J. V. Amersfoort, L. Smith, Y. W. Teh and Y. Gal, "Uncertainty Estimation Using
a Single Deep Deterministic Neural Network," in Proceedings of Machine Learning Research,
2020. [Link](https://proceedings.neurips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html) 

### **Summary**

* In this paper authors have proposed a deep neural network model which is capable of estimating uncertainty in a single forward pass and they have named
their model as Deterministic Uncertainty Quantification (DUQ).

* One of the approaches in the literature used generative models for finding the out of
distribution data and have shown that by measuring the likelihood under the data
distribution is not sufficient for estimating the uncertainty [cite](Nalisnick, E., Matsukawa, A., Teh, Y. W., Gorur, D., and Lakshminarayanan, B.
Hybrid models with deep and invertible features. arXiv preprint arXiv:1902.02767,
2019b.)

* 
### **Proposed approach**
* Original approaches like Gaussian process and support vector machines were not scaled to high dimensional data due to the lack of good kernel functions.

* Auhtors have shown that reliable uncertainty estimates can be obtained from RBF network by using two-sided Jacobian regularization.

* The proposed method consists of a deep neural network and a set of feature vectors corresponding to different centroids (classes).

* The proposed method incentivises the model to put the features of training data close to particular centroid.

* In this paper authors have used an effect called feature collapse which represents the mapping of out of distribution data to in distribution feature representations.

* For calculating the limit or threshold for this sensitivity, authors have used Lipschitz constant of the model to quantify the upper limit of the sensitivity.

* To counter affect the very low and too high sensitivities authors have used a method in which they have regularized the Jacobian with respect to the given input

* The reason for not considering very low and too high sensitivities is for the
generalization and optimization characteristics.

* As the general RBFs are difficult to optimize due to their saturating loss and instability of the centroids authors have used an approach proposed by van den Oord et al. (2017)
[8] to make the training stable.

* Using the method proposed by van den Oord et al. authors have updated the centroids using an exponential moving average of the feature vectors of the data points assigned
to them and stabilized the training of DUQ.

* The proposed method consists of a deep feature extractor like ResNet without the softmax layer.
• Instead of the softmax layer authors have used a learnable weight matrix per each of
the class present in the data.
• For this, authors have computed the exponentiated distance between the model output
and the class centroids.
• Authors called the weight matrix with the size of centroid size and feature extractor
output size along with length scale hyper parameter as a Radial Basis Function kernel.
• For the loss function authors have considered the sum of binary cross entropy between
each class’s kernel value and one hot encoding of the labels.

* The defined class centroids are updated using the exponential moving average of the
feature vectors of the data points belonging to the particular class.

* For the optimization authors have used stochastic gradient descent with a momentum
between 0.99 and 0.999.

* During training the proposed approach made the centroids pushed away at each
minibatch without converging to a stable point.

* To avoid this, authors have regularized the l2 norm of the parameters, thus restricting
the model to sensible solutions and helped for optimization.


### **Datasets and experiments:**

* DNN
    * Resnet without softmax layer.
    * Resnet-18


* Loss Function - Binary cross entropy loss.

* Optimizer- Stochastic gradient descent with momentum between 0.99 and 0.999.

* Dataset
    * MNIST
    * Fashion MNIST
    * Not MNIST
    * CIFAR
    * SVHN
    * Two moons dataset

* Authors have evaluated the proposed model by comparing the results with the current
best approaches like Ensembles

* Authors claim that they have stabilized the training of Radial biases function networks.

* They have also shown a comparison for out of distribution (OoD) detections for a
number of evaluations which include Fashion MNIST vs MNIST and CIFAR vs
SVHN.

* authors have visualized the performance of the proposed DUQ method on two
moons dataset and have shown that the certainty decreases when the data is far away
from training data.

* Authors have shown good evaluation results for the behaviour of the DUQ in two
dimensions namely with 2 sided-gradient penalty and one-sided gradient penalty using the two moons dataset.

* As an experimental setup author have created a 3 layered deep convolutional neural network and during testing, they have set the Batch Normalization to evaluation mode.

* For tuning the hyper parameters length scale and gradient penalty weight, authors have initially set the gradient penalty weight to zero and have computed the length scale by doing a grid search over the interval (0,1)

* After measuring for 5 runs authors have found that a length scale of 0.1 produced
highest accuracy.

* For selecting the gradient penalty weight authors have used a third dataset (Not MNIST) and evaluated the AUROC and selected the gradient penalty weight values.

* With these hyper parameters the proposed DUQ method outperformed the existing state-of-the-art classification methods except for one.
* Authors claim that LL ratio method which is based on Generative models outperformed the proposed DUQ method but its computational cost is very high when compared the
DUQ while training.

* For visualizing the results authors have plotted a normalized histogram for the kernel distances of CIFAR-10 and SVHN in which most of the CIFAR-10 data is very close to 1 and the SVHN data is uniformly spread out over the range of the distances.


### **Notes**
* Training genertaive models is computationally more expensive than training for clasification.
* Gradient penalties are also called as double backpropogation and this penalty has been used successfully to train Wasserstein GANs to regularize the Lipschitz constant.
### **Important**
* Uncertainty estimation techniques based on Bayesian deep neural networks are hard to infer.

* By regularizing the representation map using gradient penalty the effect of feature collapse can be avoided.

* By using a gradient penalty smoothness is enforced, which limits how quickly the output of a function changes when the input x changes.

* Smoothness is critical for generalisation, especially when utilizing a kernel that is dependent on representation space distances.

* Collapsing features can improve accuracy, but they can also make input points
indistinguishable in the representation space, which reduces our ability to identify out of distributions.

* In practice Deep Ensemble methods outperform Variational Bayesian methods but are
computationally expensive.

* The memory and computation required by the Deep Ensembles increases linearly with number of ensemble elements used at both training and testing.

* In RBF networks uncertainty is given by the distance between model output and the
closest centroid.

* If the model is sensitive to the changes in the input, then one can reliably detect out of distribution data.