# ProtoGrad.jl

[![Build status](https://github.com/lostella/ProtoGrad.jl/workflows/Test/badge.svg)](https://github.com/lostella/ProtoGrad.jl/actions?query=workflow%3ATest)

ProtoGrad is an experimental Julia package to work with gradient-based model optimization, including (of course!) deep learning.
It aims at being simple, composable, and flexible.
This said, it's very much of a work-in-progress playground for ideas, so don't expect feature completeness or stability just yet.

The package builds on top of much more mature and popular libraries, above all [Zygote](https://github.com/FluxML/Zygote.jl) (for automatic differentiation) and [NNLib](https://github.com/FluxML/NNlib.jl) (providing common operators in deep learning).

Check out the [examples folder](./examples/) on how to use ProtoGrad to construct and train models, or keep following the present README to get a feeling for the package philosophy.

It all begins, naturally, with

````julia
using ProtoGrad
````

## Models

Models are just callable objects, whose type extends the `ProtoGrad.Model` abstract type.
The following (overly) simple example defines some type of linear model (a better version of this is `ProtoGrad.Linear`):

````julia
struct LinearModel <: ProtoGrad.Model
    A
    b
end

(m::LinearModel)(x) = m.A * x .+ m.b
````

All attributes of a model are interpreted as parameters to be optimized, and so gradients will be taken with respect to them. It is therefore assumed that all attributes are
1. Numerical arrays, i.e. of type `<:AbstractArray{<:AbstractFloat}`;
2. Functions;
3. Other `Model` objects;
4. `Tuple`s of objects of the above types.

> **Note:** This means, for example, that hyper-paramenters cannot be stored as attributes.
> Some hyperparameters are implicit in the model structure (e.g. number of layers or units);
> otherwise, they can be stored as type parameters (as ["value types"](https://docs.julialang.org/en/v1/manual/types/#%22Value-types%22)).

Models defined this way get the structure of a vector space, for free:

````julia
m = LinearModel(randn(3, 5), randn(3))
m_scaled = 2 * m # this is also of type LinearModel
m_sum = m + m_scaled # this too
````

````
Main.##310.LinearModel
````

The dot-syntax for in-place assignment and loop fusion can also be used:

````julia
m_scaled .= 2 .* m
m_sum .= m .+ m_scaled
````

````
Main.##310.LinearModel
````

And you can take dot products too!

````julia
using LinearAlgebra
dot(m, m_sum)
````

````
60.98490613000809
````

## Objective functions

Training a model usually amounts to optimizing some objective function.
In principle, any custom function of the model will do.
For example, we can use the mean squared error

````julia
mean_squared_error(yhat, y) = sum((yhat .- y).^2) / size(y)[end]
````

````
mean_squared_error (generic function with 1 method)
````

together with some data (here artificially generated, according to a random, noisy linear model)

````julia
A_original = randn(3, 5)
b_original = randn(3)
x = randn(5, 300)
y = A_original * x .+ b_original .+ randn(3, 300)
````

to define the objective:

````julia
objective = model -> mean_squared_error(model(x), y)

objective(m) # returns some "large" loss value
````

````
16.988103019562196
````

Stochastic approximations to the full-data objective above can be implemented by iterating the data in batches, and coupling it with the loss, as follows:

````julia
using StatsBase

batch_size = 64
batches = ProtoGrad.forever() do
    idx = sample(1:size(x)[end], batch_size, replace = false)
    return (x[:, idx], y[:, idx])
end

stochastic_objective = ProtoGrad.SupervisedObjective(mean_squared_error, batches)
````

````
ProtoGrad.SupervisedObjective{typeof(Main.##310.mean_squared_error), Base.Generator{Base.Iterators.Repeated{Nothing}, ProtoGrad.var"#1#2"{Main.##310.var"#3#4"}}, Tuple{Matrix{Float64}, Matrix{Float64}}, Nothing}(mean_squared_error, Base.Generator{Base.Iterators.Repeated{Nothing}, ProtoGrad.var"#1#2"{Main.##310.var"#3#4"}}(ProtoGrad.var"#1#2"{Main.##310.var"#3#4"}(Main.##310.var"#3#4"()), Base.Iterators.Repeated{Nothing}(nothing)))
````

Evaluating this new objective on `m` will yield

````julia
stochastic_objective(m) |> println # a different
stochastic_objective(m) |> println # value
stochastic_objective(m) |> println # every
stochastic_objective(m) |> println # time
````

````
13.799104481641637
16.473127356129876
18.438358936788205
16.652131475339687

````

## Gradient computation

Computing the gradient of our objective with respect to the model is easy:

````julia
grad, val = ProtoGrad.gradient(objective, m)
````

````
(Main.##310.LinearModel, 16.988103019562196)
````

Here `val` is the value of the objective evaluated at `m`, while `grad` contains its gradient with respect to **all** attributes of `m`. Most importantly **`grad` is itself a `LinearModel` object**. Therefore, `grad` can be added or subtracted from `m`, used in dot products and so on.

## Fitting models to the objective

Fitting models using gradient-based algorithms is now relatively simple.
The following loop is plain gradient descent with constant stepsize:

````julia
m_fit = copy(m)
for it in 1:100
    grad, _ = ProtoGrad.gradient(objective, m_fit)
    m_fit .= m_fit .- 0.1 .* grad
end
````

````
??? Warning: Assignment to `grad` in soft scope is ambiguous because a global variable by the same name exists: `grad` will be treated as a new local. Disambiguate by using `local grad` to suppress this warning or `global grad` to assign to the existing global variable.
??? @ string:3

````

To verify that this worked, we can check that the objective value is much smaller for `m_fit` than it was for `m`:

````julia
objective(m_fit) # returns a small loss value compared to `m`
````

````
2.949699249534306
````

ProtoGrad implements gradient descent and other optimization algorithms in the form of iterators. The following will yield the same iterations as we just did:

````julia
optimizer = ProtoGrad.GradientDescent(stepsize=1e-1)
iterations = Iterators.take(optimizer(m, objective), 100)
````

````
Base.Iterators.Take{ProtoGrad.GradientDescentIterable{Main.##310.LinearModel, Main.##310.var"#1#2", Float64}}(ProtoGrad.GradientDescentIterable{Main.##310.LinearModel, Main.##310.var"#1#2", Float64}(Main.##310.LinearModel, Main.##310.var"#1#2"(), 0.1), 100)
````

The `iterations` object is an iterator that can be looped over, and its elements be inspected (for example to decide when to stop training). For the sake of compactness, here we will just take the output of the last iteration as solution:

````julia
m_fit = ProtoGrad.last(iterations).solution
````

````
Main.##310.LinearModel
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

