TensorFlow V1 implementation (See [DMSP page](https://github.com/siavashBigdeli/DMSP) for Caffe and MatCovNet implementations, and [DMSP-TF2 page](https://github.com/siavashBigdeli/DMSP-TF2) for Tensorflow V2 implementation with super-resolution support))
## Deep Mean-Shift Priors for Image Restoration ([project page](https://www.cs.umd.edu/~zwicker/projectpages/DeepMeanShiftPriors-NIPS17.html))

Siavash Arjomand Bigdeli, Meiguang Jin, Paolo Favaro, Matthias Zwicker

Advances in Neural Information Processing Systems (NIPS), 2017

### Abstract:
In this paper we introduce a natural image prior that directly represents a Gaussian-smoothed version of the natural image distribution. We include our prior in a formulation of image restoration as a Bayes estimator that also allows us to solve noise-blind image restoration problems. We show that the gradient of our prior corresponds to the mean-shift vector on the natural image distribution. In addition, we learn the mean-shift vector field using denoising autoencoders, and use it in a gradient descent approach to perform Bayes risk minimization. We demonstrate competitive results for noise-blind deblurring, super-resolution, and demosaicing.


<img src="https://www.cs.umd.edu/~zwicker/projectpages/DeepMeanShiftPriors-NIPS17-teaser.jpg" alt="Drawing" style="height: 500px;" align="center"/>

See [manuscript](https://papers.nips.cc/paper/6678-deep-mean-shift-priors-for-image-restoration.pdf) for details of the method.

This code runs in Python and you need to install [TensorFlow](http://www.tensorflow.org).
### Contents:

[demo_DMSP.py](https://github.com/siavashBigdeli/DMSP-tensorflow/blob/master/demo_DMSP.py): Includes an example for non-blind and noise-blind image deblurring.

[DMSPDeblur.py](https://github.com/siavashBigdeli/DMSP-tensorflow/blob/master/DMSPDeblur.py): Implements MAP function for non-blind image deblurring. Use Python's help function to learn about the input and output arguments.

[DAE_model](https://github.com/siavashBigdeli/DMSP-tensorflow/tree/master/DAE_model.py): Includes DAE model and implementation.

[data](https://github.com/siavashBigdeli/DMSP-tensorflow/tree/master/data): Includes sample image(s).

### License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
