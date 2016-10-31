# Detecting differential growth of microbial populations with Gaussian process regression.

### Peter Tonner, Cynthia Darnell, Barbara E Engelhardt, Amy Schmid

Code repository corresponding to paper "Detecting differential growth of microbial populations with Gaussian process regression."

## Setup

### Requirements

* git (or download repository as tarball)
* python (virtualenv recommended)

#### Download this repository
`git clone https://github.com/ptonner/gp_growth_phenotype.git`

#### Setup a local python environment (optional)
`virtualenv .`

`source bin/activate`

#### Install requirements
`pip install -r requirements.txt`

## Examples

Input to the B-GREAT method should come from two dataframes, *data* (n x p) and *meta* (p x k) where n, p, and k are:
* n: number of time-points
* p: number of time-course measurements (replicates)
* k: number of covariates to be tested, e.g. strain, condition.

The index of *data* is assumed to be the time column. Each row of *meta* should correspond to a single column of *data*, e.g. the number of columns in *data* should equal the number of rows in *meta*. 

```python
import pandas as pd
data = pd.read_csv("path/to/data.csv",index_col=0)
meta = pd.read_csv("path/to/meta.csv")

assert data.shape[1] == meta.shape[0]
```

These two variables should then be given to bgreat for use in analysis.
```python
import bgreat
bgreat.setGlobals(_data=data,_meta=meta)
```

### Testing strain effects

For a single effect on growth, a single column in _meta_ will be used for testing differential growth. _meta_ should include a column called _strain_ that defines the strain being growth in each column of _data_. A specific value in the _strain_ column should correspond to the 'parent' or 'control' strain, and is given to bgreat through the `setGlobals` function. Additionally, a column should be added to meta called `"strain-regression"` which represents the column used in building the GP regression model. It should contain 1's and 0's corresponding to a non-parent or parent strain sample, respectively.

```python
parent = 'my-parent'
meta['strain-regression'] = (meta.strain!=parent).astype(int)
bgreat.setGlobals(_parent=parent,_control=control)
```

A list of mutants can then be given to calculate the Bayes factor score, with permutation testing and FDR calculation.

```python
mutants = ['m1','m2',...]
results = testMutants(mutants)
```

### Testing interactions

Interaction testing requires an additional variable, the control condition, as well as two additional columns in meta - `"condition"` and `"interaction"`

```python
control = 'my-control'
meta['interaction'] = ((meta.Condition!=control) & (meta.strain!=parent)).astype(int)
meta['condition'] = (meta.Condition!=control).astype(int)
bgreat.setGlobals(_control=control,_meta=meta)

results = bgreat.testMutantCondition(mutants)
```

