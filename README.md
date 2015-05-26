**Table of Contents**

- [Machine Learning](#)
	- [Localized sliced inverse regression](#) - LSIR/
		- [Introduction](#)
		- [Inputs and Outputs](#)
		- [Examples](#)
			- [Tai-Chi](#)
	- [Bayesian Mixture of Inverse Regression (BMI)](#) - BMI/
		- [Introduction](#)
		- [A Brief Tutorial](#)
		- [Examples](#)
			- [Dimension Reduction](#)
	- [Bayesian Gradient Learning](#) - gradlearn/
		- [Introduction](#)
		- [A Brief Tutorial](#)
		- [Examples](#)
			- [Dimension Reduction](#)
			- [Graphical Models](#)
	- [Kernel Sliced Inverse Regression (kSIR)](#) - KSIR
		- [Introduction](#)
		- [Inputs and Outputs](#)
		- [A toy Example](#)


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

![digifig](https://raw.githubusercontent.com/sayan-duke/machine-learning/master/readme_images/digifig.jpg)

Figure1: Illustration of handwritten digit data 1-9.
Suppose we want to classify digit "5" and digit "8". We can collect 100 samples for each digit and label digit "5" as response=1 and digit "8" as response=0, so that the covariates X is a 200*784 matrix for pixel values and the response Y is a 200 vector for class labels. We can type in the command line:

[B, Bpost]=bmi(Y, X, 1, res, 'T')

where the fourth argument is a structure with elements:

res.type='d': the response is discrete; res.alpha0=1: the concentration parameter = 1

The fifth argument 'T' tells the program to pre-process the data by a principle component analysis, since in this case the number of input variables is larger than the sample size.

Now Bpost, a 784*1000 matrix, contains the posterior samples for the e.d.r., and B is the orthonormalized version of the mean of Bpost. "mean(Bpost,2)" and "std(Bpost,0,2)" return the posterior mean and standard deviation for this direction, both 784 vectors. We can plot them in a visually friendly way by "imagesc(reshape(mean(Bpost,2),28,28)')" and "imagesc(reshape(std(Bpost,0,2),28,28)')" with the results shown in Figure 2. The red part in the left panel is exactly the region that differentiates digit "5" and "8", hence if we project the original data onto this direction we can immediately perform classification. The right panel indicates small uncertainty.

![digi58topf](https://raw.githubusercontent.com/sayan-duke/machine-learning/master/readme_images/digi58topf.jpg)

Figure2: The left panel is the posterior mean and the right panel is the posterior standard deviation for the top dimension reduction direction.

Full description [here](https://stat.duke.edu/~km68/BMI.htm).
























## Bayesian Gradient Learning

### Introduction

This is a software based on the method proposed in [K. Mao etal. (2008)](http://ftp.stat.duke.edu/WorkingPapers/08-24.html) that does simultaneous dimension reduction and regression as well as inference of graphical models in Bayesian perspective. It realizes the Markov chain Monte Carlo procedure in that paper and returns posterior draws of quantities such as the effective dimension reduction directions and the gradient outer product matrix (GOP), from which further inference can be made and uncertainty can be measured. Both regression and binary classification are available. It runs on Matlab.

### A Brief Tutorial

Unzip the downloaded file. To start, first make sure that the folder "gradlearn/" and the sub folders and files are in the search path of Matlab. The input training data should consist of a response vector Y and a covariate matrix X with rows observations and columns dimensions (input variables). The program will return the RKHS (Reproducing Kernel Hilbert Space) norms for each dimension and (optionally) posterior draws of the dimension reduction directions and GOP.

For example, suppose we have training data X and Y formatted as aforementioned and we can type:

[gnorm, dr, gop]=gradlearn(Y, X, 'r')

in the command line. For the third argument, 'r' tells the program to do regression and 'c' means binary classification. The output "gnorm" stores the posterior draws of the RKHS norms for each dimension with rows corresponding to dimensions and columns corresponding to draws. "dr" is a cell structure with dr{i} the i-th dimension reduction direction draw, again a matrix form with each column a draw for this direction vector. "gop" is the draw of the GOP matrix, with each column a draw for the GOP but in a vector form, so that to recover the matrix form of say the t-th GOP draw one needs to type:

GOP_t=reshape(gop(:,t), p, p)

where p is the number of dimensions.

Note that the output gop might take up a large amount of memory, so in practice if this quantity is not needed one can ignore it by specifying only the first two outputs:

[gnorm, dr]=gradlearn(Y, X, 'r')

Additional inputs are available. For a further detailed explanation of the inputs and outputs type "help gradlearn" in the command line.

### Examples

#### Dimension Reduction

We illustrate how dimension reduction can be performed in this section by considering a classification problem for handwritten digits. An illustration of the digit data is shown in Figure 1. Each digit is represented by a 28*28=784 vector that contains the pixel values.

![digifig](https://raw.githubusercontent.com/sayan-duke/machine-learning/master/readme_images/digifig.jpg)


Figure1: Illustration of handwritten digit data 1-9.
Suppose we want to classify digit "5" and digit "8". We can collect 100 samples for each digit and label digit "5" as response=1 and digit "8" as response=0, so that the covariates X is a 200*784 matrix for pixel values and the response Y is a 200 vector for class labels. We can type in the command line:

[gnorm, dr]=gradlearn(Y, X, 'c', [1000,1000])

where the fourth argument specifies [burn-in steps, number of posterior samples kept]

Now dr{1}, a 784*1000 matrix, contains the posterior samples for the first (top) dimension reduction direction. "mean(dr{1},2)" and "std(dr{1},0,2)" return the posterior mean and standard deviation for this direction, both 784 vectors. We can plot them in a visually friendly way by "imagesc(reshape(mean(dr{1},2),28,28)')" and "imagesc(reshape(std(dr{1},0,2),28,28)')" with the results shown in Figure 2. The red part in the left panel is exactly the region that differentiates digit "5" and "8", hence if we project the original data onto this direction we can immediately perform classification. The right panel indicates small uncertainty.

![digi58topf](https://raw.githubusercontent.com/sayan-duke/machine-learning/master/readme_images/digi58topf.jpg)

Figure2: The left panel is the posterior mean and the right panel is the posterior standard deviation for the top dimension reduction direction.

#### Graphical Models

The GOP matrix can be used to infer a graphical model that provides information for the partial correlation between any two dimensions w.r.t the response given all the other dimensions. Consider a toy example: Let $\theta_j$ be a n-vector with each element a standard normal random variable, j=1,...,5, and $x_1=\theta_1, x2=\theta_1+\theta_2$ $\theta_j$, where  $\X_i$ is the i-th column of the covariate X, a n*5 matrix with n the sample size. The response Y is generated in such a way that $Y_i=X_{i1}+\frac{X_{i3}+X_{i5}}{2}+\varepsilon_i$, for i=1,...,n, with $\varepsilon_i \sim N(0,0.25)$.

We can type: [gnorm, dr, gop]=gradlearn(Y, X, 'r', [1000,1000])

Now "gop" is the draw of the GOP matrix, with each column a draw for the GOP but in a vector form, so to obtain the posterior mean GOP matrix one needs a command like "GOPmean=reshape(mean(gop,2),5,5)", where the mean is in element-wise sense. To infer a graphical model one can compute the partial correlation matrix from the GOP matrix. The relationship between these two matrices can be found in K. Mao etal. (2008) Page 5.

Figure 3 shows the posterior mean and standard deviation for the GOP matrix and the partial correlation matrix. The partial correlation matrix clearly captures the negative covariation between Dimension 1,3,5 w.r.t the response.

![toyg](https://raw.githubusercontent.com/sayan-duke/machine-learning/master/readme_images/toyg.jpg)

Figure3: The posterior mean and standard deviation for the GOP matrix and the partial correlation matrix.

Full description [here](https://stat.duke.edu/~km68/gradlearn.htm).
























## Kernel Sliced Inverse Regression (kSIR)

### Introduction

This is a MATLAB function that implements the method proposed in the paper ["Regularized Sliced Inverse Regression for Kernel Models"](https://stat.duke.edu/~km68/files/Wu-KSIR.pdf) by Wu et.al (2008) from [Department of Statistical Science](http://stat.duke.edu/) of [http://www.duke.edu/](Duke University). It extends the sliced inverse regression framework for nonlinear dimension reduction using kernel models and regularization and can be applied to high-dimensional data.

### Inputs and Outputs

Suppose the sample size is n and the number of explanatory variables is p, and x_i is a p-vector denoting the covariates for the i-th observation, i=1,...n. The inputs of this function should consist of a n*n kernel matrix KER with the (i,j)-th element K(x_i,x_j) where K(.,.) is a suitable kernel function, a response vector Y with length n, the number of effective dimension reduction (EDR) dimensions d, a regularization parameter s, and (optionally) a structure type variable that specifies doing regression or classification and the number of slices. The function will return a structure type variable SIR that contains the estimated quantities for computing kSIR variates, specifically, the following command

SIR.C*(KER(:,i)-mean(KER(:,i)))+SIR.b

returns the kSIR variate(s) for the i-th observation. For a new observation (possibly from a test dataset) with a covariates vector x*, suppose KERx=(K(x*,x_1),...,K(x*,x_n))', then "SIR.C*(KERx-mean(KERx))+SIR.b" returns the kSIR variate(s) for this new observation.

Try "help KSIR" in the command line for more details.

### A toy Example

In this section we illustrate how nonlinear dimension reduction is achieved by the kSIR through a toy regression problem.

Suppose there are 400 observations, each x_i, (i=1,...400) is a 5-vector drawn from a 5-dimensional multivariate normal distribution with mean 0 and covariance the identity matrix. The i-th response y_i equals the sum of squares of the first and second dimensions of x_i plus a random error (from a univariate normal distribution with mean 0 and standard deviation 0.2).

Evidently any linear combination of the explanatory variables cannot explain the variance in this example. For kSIR one can first specify a suitable kernel function K(.,.), say, a Gaussian kernel with $K(x,y)= with s being some bandwidth parameter. Now suppose KER is a 400*400 matrix with the (i,j)-th element K(x_i,x_j), and type

[SIR] = KSIR(KER, Y, 1, 0.1, opts);

in the command line with opts.pType = 'r' and opts.H = 20. The third argument tells the function to choose 1 EDR hence 1 kSIR variate, and the fourth argument sets the regularization parameter to be 0.1. opts is a structure: opts.pType='r' means doing regression and opts.H=20 means the number of slices is taken to be 20. Now the command

var1=zeros(400,1); for i=1:400 var1(i)=SIR.C*(KER(:,i)-mean(KER(:,i)))+SIR.b; end

calculates one kSIR variate (since only one EDR is chosen) for each of the observations, and "plot(var1,Y, '.')" produces the Figure below, from which it is seen that this variate is highly predictive.

![ksirtoy](https://raw.githubusercontent.com/sayan-duke/machine-learning/master/readme_images/ksirtoy.jpg)

Full description [here](https://stat.duke.edu/~km68/ksir.htm).