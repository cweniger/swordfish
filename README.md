`swordfish` is a Python tool to study the information yield of counting experiments.

Motivation
----------

With `swordfish` you can quickly and accurately forecast experimental
sensitivities without all the fuss with time-intensive Monte Carlos, mock data
generation and likelihood maximization.

With `swordfish` you can

- Calculate the expected upper limit or discovery reach of an instrument.
- Derive expected confidence contours for parameter reconstruction.
- Visualize confidence contours as well as the underlying information metric field.
- Calculate the *information flux*, an effective signal-to-noise ratio that
  accounts for background systematics and component degeneracies.
- Calculate the Euclideanized signal which approximately maps the signal to
 a new vector which can be used to calculate the Euclidean distance between points

A large range of experiments in particle physics and astronomy are
statistically described by a Poisson point process.  The `swordfish` module
implements at its core a rather general version of a Poisson point process with
background uncertainties described by a Gaussian random field, and provides
easy access to its information geometrical properties.  Based on this
information, a number of common and less common tasks can be performed.


Get started
-----------

Most of the functionality of `swordfish` is demonstrated in two jupyter
notebooks.

- [Equivalent counts method and Fisher Information Flux](https://github.com/cweniger/swordfish/tree/master/docs/jupyter/Examples_I.ipynb)
- [Confidence contours, streamline visualisation, and Euclideanized signal](https://github.com/cweniger/swordfish/tree/master/docs/jupyter/Examples_II.ipynb)

In addition we provide two physics examples from direct and indirect detection

- [CTA](https://github.com/cweniger/swordfish/blob/master/Examples/swordfish_ID.ipynb)
- [Xenon-1T](https://github.com/cweniger/swordfish/blob/master/Examples/swordfish_DD.ipynb)


Documentation
-------------

A full documentation of `swordfish` can be found on
[github.io](https://cweniger.github.io/swordfish).  For extensive details about
Fisher forecasting with Poisson likelihoods, the effective counts method, the
definition of information flux and the treatment of background systematics see
[http://arxiv.org/abs/1704.05458](http://arxiv.org/abs/1704.05458) and
[http://arxiv.org/abs/1712.05401](http://arxiv.org/abs/1712.05401).


Installation
------------

`swordfish` has been tested with Python 2.7.13 and the packages

- `numpy 1.13.1`
- `scipy 0.19.0`
- `matplotlib 2.0.0`

Let us know if you run into problems.

`swordfish` can be installed by invoking

    git clone https://github.com/cweniger/swordfish
    cd swordfish
    python setup.py install

or
    
    pip install git+https://github.com/cweniger/swordfish

Citation
--------

If you use the package, please cite one or both of the papers
[http://arxiv.org/abs/1712.05401](http://arxiv.org/abs/1712.05401)
and
[http://arxiv.org/abs/1704.05458](http://arxiv.org/abs/1704.05458).
