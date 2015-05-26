# Machine Learning

## Localized sliced inverse regression

### Introduction

This is a MATLAB function that implements the method proposed in the paper ["Localized Sliced Inverse Regression"](https://stat.duke.edu/~km68/files/Wu-LSIR.pdf) by Wu et.al (2008) from [Department of Statistical Science](http://stat.duke.edu/) of [http://www.duke.edu/](Duke University). It allows for supervised dimension reduction on nonlinear subspaces and alleviates the issue of degeneracy in the classical Sliced Inverse Regression (SIR) method by taking into account of the local structure of the input variables. 

### Inputs and Outputs

The inputs of this function should consist of a covariate matrix X with columns corresponding to observations and rows dimensions (input variables), a response vector Y with length equal the number of columns of X, the number of effective dimension reduction (EDR) dimensions d, a regularization parameter s, and (optionally) a structure type variable that specifies performing regression or classification, the number of slices and the number of nearest neighbors. The function will return a structure variable that contains the estimated EDR directions and some other quantities. Try "help LSIR" in the command line for more details.

### Examples

#### Tai-Chi

In this section we illustrate how dimension reduction is performed by considering a classification problem for the Tai-Chi data. The covariate matrix X has six rows (hence six dimensions). An illustration of the first two dimensions is shown in Figure 1 (a). The third to sixth dimensions are independent random errors. The number of columns (the sample size) is taken to be 1000 (500 for each of the red and blue points). The response Y is simply a vector of -1 and 1 with -1 corresponding to the red points and 1 the blue points.

Now obviously the true EDRs should be (e1, e2) where e1=(1,0,0,0,0,0)' and e2=(0,1,0,0,0,0)'. One can type:

[LS] = LSIR(X, Y, 2, 0, opts);

in the command line with opts.pType = 'c' and opts.numNN = 10. The third argument tells the function to choose 2 EDRs, and the fourth argument sets the regularization parameter to be 0. opts is a structure: opts.pType='c' means performing classification and opts.numNN=10 means the number of nearest neighbors is taken to be 10.

The output LS is a structure, and usually LS.edrs which represents the estimated EDRs is of concern. By typing "plot(LS.edrs(:,1)'*X(:,Y==1) , LS.edrs(:,2)'*X(:,Y==1))" followed by "hold on; plot(LS.edrs(:,1)'*X(:,Y==-1) , LS.edrs(:,2)'*X(:,Y==-1))" one projects the training data X onto the estimated 2 EDRs and visualizes it. The resulting figure is shown in Figure 1 (b). We also form an independent test dataset formatted like X and repeat the project and visualize procedure, and the resulting figure is shown in Figure 1 (c).

![tai chi](https://raw.githubusercontent.com/sayan-duke/machine-learning/master/readme_images/digifig.jpg)

Figure1: The first two dimensions of the Tai-Chi data (a), projection (on the estimated first two EDRs) of the training data (b) and projection (on the estimated first two EDRs) of an independent test data (c).

Full description [here](https://stat.duke.edu/~km68/lsir.htm).


## Bayesian Mixture of Inverse Regression (BMI)

### Introduction

This is a Matlab implementation for the supervised dimension reduction method proposed in [K. Mao etal. (2009)](http://ftp.stat.duke.edu/WorkingPapers/09-08.html) which utilizes Bayesian mixture models in an inverse regression scenario. The code realizes the Markov chain Monte Carlo procedure in that paper and returns posterior draws of quantities such as the effective dimension reduction directions (e.d.r) from which further inference can be made and uncertainty can be measured.

### A Brief Tutorial

Unzip the downloaded file. To start, first make sure that the folder "BMI/" and the sub folders and files are in the search path of Matlab. The input training data should at least consist of a response vector Y, a covariate matrix X with rows observations and columns explanatory/input variables, an integer number specifying the number of e.d.r. directions, and a structure specifying whether the response is continuous or discrete and the corresponding parameters to be tuned. The program will return the posterior mean and posterior draws for the e.d.r. directions. For a further detailed explanation of the inputs and outputs type "help bmi" in the command line.

### Examples

#### Dimension Reduction

We illustrate how dimension reduction can be performed in this section by considering a classification problem for handwritten digits. An illustration of the digit data is shown in Figure 1. Each digit is represented by a 28*28=784 vector that contains the pixel values.

![digifig](https://raw.githubusercontent.com/sayan-duke/machine-learning/readme-images/digifig.jpg)

Figure1: Illustration of handwritten digit data 1-9.
\includegraphics[totalheight=2.5in]{digifig.jpg}
Suppose we want to classify digit "5" and digit "8". We can collect 100 samples for each digit and label digit "5" as response=1 and digit "8" as response=0, so that the covariates X is a 200*784 matrix for pixel values and the response Y is a 200 vector for class labels. We can type in the command line:

[B, Bpost]=bmi(Y, X, 1, res, 'T')

where the fourth argument is a structure with elements:

res.type='d': the response is discrete; res.alpha0=1: the concentration parameter = 1

The fifth argument 'T' tells the program to pre-process the data by a principle component analysis, since in this case the number of input variables is larger than the sample size.

Now Bpost, a 784*1000 matrix, contains the posterior samples for the e.d.r., and B is the orthonormalized version of the mean of Bpost. "mean(Bpost,2)" and "std(Bpost,0,2)" return the posterior mean and standard deviation for this direction, both 784 vectors. We can plot them in a visually friendly way by "imagesc(reshape(mean(Bpost,2),28,28)')" and "imagesc(reshape(std(Bpost,0,2),28,28)')" with the results shown in Figure 2. The red part in the left panel is exactly the region that differentiates digit "5" and "8", hence if we project the original data onto this direction we can immediately perform classification. The right panel indicates small uncertainty.

![digi58topf](https://raw.githubusercontent.com/sayan-duke/machine-learning/readme-images/digi58topf.jpg)

Figure2: The left panel is the posterior mean and the right panel is the posterior standard deviation for the top dimension reduction direction.

Full description [here](https://stat.duke.edu/~km68/BMI.htm).