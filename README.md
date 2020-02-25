[![Build Status](https://travis-ci.com/dsweber2/ZygoteFFTs.jl.svg?branch=master)](https://travis-ci.com/dsweber2/ZygoteFFTs.jl)
[![Codecov](https://codecov.io/gh/dsweber2/ZygoteFFTs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/dsweber2/ZygoteFFTs.jl)
[![Coveralls](https://coveralls.io/repos/github/dsweber2/ZygoteFFTs.jl/badge.svg?branch=master)](https://coveralls.io/github/dsweber2/ZygoteFFTs.jl?branch=master)

# ZygoteFFTs


Simple collection of Zygote adjoints for the various FFTs in
[AbstractFFTs.jl](https://juliamath.github.io/AbstractFFTs.jl/latest/api/#Public-Interface-1);
these allow for 

* precomputing plans
* ffts of strictly real data, which uses 1/2 as many coefficients thanks to
  conjugate symmetry
* probably most relevant: allows for ffts to be executed on GPUs via the
  CuArray implementations of the AbstractFFTs methods
