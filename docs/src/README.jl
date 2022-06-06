# # ProtoGrad.jl

# [![Build status](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/workflows/Test/badge.svg)](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/actions?query=workflow%3ATest)

# ProtoGrad is an experimental Julia package to work with gradient-based model optimization, including (of course!) deep learning.
# It aims at being simple, composable, and flexible.
# This said, it's very much of a work-in-progress playground for ideas, so don't expect feature completeness or stability just yet.

# The package builds on top of much more mature and popular libraries, above all [Zygote](https://github.com/FluxML/Zygote.jl) (for automatic differentiation) and [NNLib](https://github.com/FluxML/NNlib.jl) (providing common operators in deep learning).

# Check out the [examples folder](./examples/) on how to use ProtoGrad to construct and train models, or keep following the present README to get a feeling for the package philosophy.

# It all begins, naturally, with

using ProtoGrad

# ## Models

# Models are just callable objects, whose type extends the `ProtoGrad.Model` abstract type.
# The following (overly) simple example defines some type of linear model (a better version of this is `ProtoGrad.Linear`):

struct LinearModel <: ProtoGrad.Model
    A
    b
end

(m::LinearModel)(x) = m.A * x .+ m.b

# All attributes of a model are interpreted as parameters to be optimized, and so gradients will be taken with respect to them. It is therefore assumed that all attributes are
# 1. Numerical arrays, i.e. of type `<:AbstractArray{<:AbstractFloat}`;
# 2. Functions;
# 3. Other `Model` objects;
# 4. `Tuple`s of objects of the above types.

# > **Note:** This means, for example, that hyper-paramenters cannot be stored as attributes.
# > Some hyperparameters are implicit in the model structure (e.g. number of layers or units);
# > otherwise, they can be stored as type parameters (as ["value types"](https://docs.julialang.org/en/v1/manual/types/#%22Value-types%22)).

# Models defined this way get the structure of a vector space, for free:

m = LinearModel(randn(3, 5), randn(3))
m_scaled = 2 * m # this is also of type LinearModel
m_sum = m + m_scaled # this too

# The dot-syntax for in-place assignment and loop fusion can also be used:

m_scaled .= 2 .* m
m_sum .= m .+ m_scaled

# And you can take dot products too!

using LinearAlgebra
dot(m, m_sum)

# ## Objective functions

# Training a model usually amounts to optimizing some objective function.
# In principle, any custom function of the model will do.
# For example, we can use the mean squared error

mean_squared_error(yhat, y) = sum((yhat .- y).^2) / size(y)[end]

# together with some data (here artificially generated, according to a random, noisy linear model)

A_original = randn(3, 5)
b_original = randn(3)
x = randn(5, 300)
y = A_original * x .+ b_original .+ randn(3, 300)

# to define the objective:

objective = model -> mean_squared_error(model(x), y)

objective(m) # returns some "large" loss value

# Stochastic approximations to the full-data objective above can be implemented by iterating the data in batches, and coupling it with the loss, as follows:

using StatsBase

batch_size = 64
batches = ProtoGrad.forever() do
    idx = sample(1:size(x)[end], batch_size, replace = false)
    return (x[:, idx], y[:, idx])
end

stochastic_objective = ProtoGrad.SupervisedObjective(mean_squared_error, batches)

# Evaluating this new objective on `m` will yield

stochastic_objective(m) |> println # a different
stochastic_objective(m) |> println # value
stochastic_objective(m) |> println # every
stochastic_objective(m) |> println # time

# ## Gradient computation

# Computing the gradient of our objective with respect to the model is easy:

grad, val = ProtoGrad.gradient(objective, m)

# Here `val` is the value of the objective evaluated at `m`, while `grad` contains its gradient with respect to **all** attributes of `m`. Most importantly **`grad` is itself a `LinearModel` object**. Therefore, `grad` can be added or subtracted from `m`, used in dot products and so on.

# ## Fitting models to the objective

# Fitting models using gradient-based algorithms is now relatively simple.
# The following loop is plain gradient descent with constant stepsize:

m_fit = copy(m)
for it in 1:100
    grad, _ = ProtoGrad.gradient(objective, m_fit)
    m_fit .= m_fit .- 0.1 .* grad
end

# To verify that this worked, we can check that the objective value is much smaller for `m_fit` than it was for `m`:

objective(m_fit) # returns a small loss value compared to `m`

# ProtoGrad implements gradient descent and other optimization algorithms in the form of iterators. The following will yield the same iterations as we just did:

optimizer = ProtoGrad.GradientDescent(stepsize=1e-1)
iterations = Iterators.take(optimizer(m, objective), 100)

# The `iterations` object is an iterator that can be looped over, and its elements be inspected (for example to decide when to stop training). For the sake of compactness, here we will just take the output of the last iteration as solution: 

m_fit = ProtoGrad.last(iterations).solution
