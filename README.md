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
1. Numerical arrays, i.e. of type `<:AbstractArray{<:AbstractFloat}` ;
2. Other `Model` objects;
3. `Tuple` containing objects as in the previous points.

Models defined this way get the structure of a vector space, for free:

```julia
m = LinearModel(randn(3, 5), randn(3))
m_scaled = 2*m # this is also of type LinearModel
m_sum = m + m_scaled # this too
```

**Note:** all attributes of a `ProtoGrad.Model` object are assumed to be parameters to be optimized. This means, for example, that hyper-paramenters cannot be stored as attributes: they can however be stored as type parameters.

## Objective functions

Training a model usually amounts to optimizing some objective function.
While ProtoGrad exposes tools to define common objectives, any custom function of the model will do. For example, the following implements the average squared Euclidean distance between the model output and ground-truth labels:

```julia
objective(x, y, model) = sum((model(x) - y).^2) / size(x, 2)
```

Now we can generate some data `(x, y)` and evaluate the model `m` we previously initialized:

```julia
A_original = randn(3, 5)
b_original = randn(3)
x = randn(5, 300)
y = A_original * x .+ b_original

objective(x, y, m) # returns some large number
```

## Gradient computation

Computing the gradient of our objective with respect to the model is easy:

```julia
grad, val = ProtoGrad.gradient(model -> objective(x, y, model), m)
```

We can check that `val` is the value of the objective evaluated at `m`, while `grad` contains the gradient of the objective with respect to **all** attributes of `m`: most importantly **`grad` is itself a `LinearModel` object**. Therefore, `grad` can be added or subtracted from `m`.

## Optimizing models

One is free to implement any custom training loop to optimize the model.
ProtoGrad implements gradient descent (and some of its variants) in the form of iterators:

```julia
optimizer = ProtoGrad.GradientDescent(stepsize=1e-1)
iterations = optimizer(m, model -> objective(x, y, model))
```

The `iterations` object is an iterator that can be looped over, and its elements be inspected e.g. to decide when to stop training. For the sake of compactness, here we will just take a predefined iteration as solution: 

```julia
# NOTE: this is just compact, but not memory efficient
solution = collect(Iterators.take(iterations, 100))[end]
```

To verify that this worked, we can check that the objective value is much smaller for `solution` than it was for `m`:

```julia
objective(x, y, solution) # returns a small number compared to `m`
```
