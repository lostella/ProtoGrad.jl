using Base.Iterators: take, enumerate, cycle, repeated
using IterTools: takenth
using ProtoGrad
using LinearAlgebra
using DataStructures
using Test

@testset "Quadratic" begin

Q = Diagonal([1e2, 1e-1, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0])
q = randn(10)

objective = x -> 0.5*dot(x, Q*x) + dot(x, q)

x_star = -Q\q
f_star = objective(x_star)

w0 = zeros(10)

optimizers = OrderedDict(
    "GradientDescent" => ProtoGrad.GradientDescent(stepsize=1/opnorm(Q)),
    "Nesterov" => ProtoGrad.Nesterov(stepsize=1/opnorm(Q)),
    "Barzilai-Borwein" => ProtoGrad.BarzilaiBorwein(alpha=1/opnorm(Q)),
    "Adam" => ProtoGrad.Adam(stepsize=1e-1, beta1=0.9, beta2=0.999, epsilon=1e-8),
)

logging_interval = 1
max_iter = 100

results = OrderedDict(
    optimizer_name => Dict(
        iter_nr => Dict(
            "grad" => copy(output.gradient),
            "objective" => output.value,
        )
        for (iter_nr, output) in takenth(
            enumerate(take(optimizer(w0, objective), max_iter)),
            logging_interval
        )
    )
    for (optimizer_name, optimizer) in optimizers
)

end