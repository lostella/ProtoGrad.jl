# ProtoGrad.jl

ProtoGrad is an **experimental** Julia package to work with gradient-based model optimization, including deep learning.
It aims at being simple, composable, and flexible.

The package builds on top of much more mature and popular libraries, above all [NNLib](https://github.com/FluxML/NNlib.jl) (providing common operators in deep learning) and [Zygote](https://github.com/FluxML/Zygote.jl) (for automatic differentiation).

Check out the [examples folder](./examples/) on how to use ProtoGrad to construct and train models, or keep following the present README to get a feeling for the package philosophy.

## Models

Models are just callable objects, whose type extends the `ProtoGrad.Model` abstract type.
The following (overly) simple example defines some type of linear model (a better version of this is `ProtoGrad.Linear`):

```julia
struct LinearModel <: ProtoGrad.Model
    A
    b
end

(m::LinearModel)(x) = m.A * x .+ m.b
```

The only assumption on models is that their set of attributes is any combination of:
1. Numerical arrays, i.e. of type `<:AbstractArray{<:AbstractFloat}`;
2. Other `Model` objects;
3. Functions;
4. `Tuple`s of objects of the above types.

Models defined this way get the structure of a vector space, for free:

```julia
m = LinearModel(randn(3, 5), randn(3))
m_scaled = 2 * m # this is also of type LinearModel
m_sum = m + m_scaled # this too
```

The dot-syntax for in-place assignment and loop fusion can also be used:

```julia
m_scaled .= 2 .* m
m_sum .= m .+ m_scaled
```

And you can take dot products too!

```julia
using LinearAlgebra
dot(m, m_sum)
```

## Objective functions

Training a model usually amounts to optimizing some objective function.
In principle, any custom function of the model will do.
For example, we can use the mean squared error

```julia
mean_squared_error(yhat, y) = sum((yhat .- y).^2) / size(y)[end]
```

together with some artificial data (generated according to a random, noisy linear model)

```julia
A_original = randn(3, 5)
b_original = randn(3)
x = randn(5, 300)
y = A_original * x .+ b_original .+ randn(3, 300)
```

to define the objective:

```julia
objective = model -> mean_squared_error(model(x), y)

objective(m) # returns some large number
```

Stochastic approximations to the full-data objective above can be implemented by iterating the data in batches, and coupling it with the loss, as follows:

```julia
using StatsBase

batch_size = 64
batches = (
    begin
        idx = sample(1:size(x)[end], batch_size, replace = false)
        (x[:, idx], y[:, idx])
    end
    for _ in ProtoGrad.forever
)

stochastic_objective = ProtoGrad.SupervisedObjective(mean_squared_error, batches)
```

Evaluating this new objective on `m` will yield

```julia
stochastic_objective(m) # a different
stochastic_objective(m) # value
stochastic_objective(m) # every
stochastic_objective(m) # time
```

## Gradient computation

Computing the gradient of our objective with respect to the model is easy:

```julia
grad, val = ProtoGrad.gradient(objective, m)
```

Here `val` is the value of the objective evaluated at `m`, while `grad` contains its gradient with respect to **all** attributes of `m`. Most importantly **`grad` is itself a `LinearModel` object**. Therefore, `grad` can be added or subtracted from `m`, used in dot products and so on.

> **Note:** All attributes of a `ProtoGrad.Model` object are assumed to be parameters to be optimized, hence gradients will be taken with respect to them.
> This means, for example, that hyper-paramenters cannot be stored as attributes.
> Some hyperparameters are implicit in the model structure (e.g. number of layers or units);
> in any case, they can usually be stored as type parameters (as ["value types"](https://docs.julialang.org/en/v1/manual/types/#%22Value-types%22)).

## Fitting models to the objective

Fitting models using gradient-based algorithms is now relatively simple.
The following loop is plain gradient descent with constant stepsize:

```julia
m_fit = copy(m)
for it in 1:100
    grad, _ = ProtoGrad.gradient(objective, m_fit)
    m_fit .= m_fit .- 0.1 .* grad
end
```

To verify that this worked, we can check that the objective value is much smaller for `m_fit` than it was for `m`:

```julia
objective(m_fit) # returns a small number compared to `m`
```

ProtoGrad implements gradient descent and other optimization algorithms in the form of iterators. The following will yield the same iterations as we just did:

```julia
optimizer = ProtoGrad.GradientDescent(stepsize=1e-1)
iterations = optimizer(m, objective)
```

The `iterations` object is an iterator that can be looped over, and its elements be inspected (for example to decide when to stop training). For the sake of compactness, here we will just take a predefined iteration as solution: 

```julia
# NOTE: this is just compact, but not memory efficient
m_fit = collect(Iterators.take(iterations, 100))[end]
```
