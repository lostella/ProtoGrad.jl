using Base.Iterators: take, enumerate, cycle, repeated
using IterTools: takenth
using ProtoGrad
using LinearAlgebra
using Plots
using DataStructures
using Test

@testset "Quadratic" begin

struct QuadraticFunction{TQ, Tq}
    Q::TQ
    q::Tq
end

function (f::QuadraticFunction)(x)
    return 0.5*dot(x, f.Q*x) + dot(x, f.q)
end

Q = Diagonal([1e2, 1e-1, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0])
q = randn(10)

objective = QuadraticFunction(Q, q)

x_star = -Q\q
f_star = objective(x_star)

w0 = zeros(10)

optimizers = OrderedDict(
    # Isn't 1/opnorm(Q) the "best" stepsize for GD?
    # TODO: Check stepsize selection on Nesterov's book
    "GD" => ProtoGrad.GradientDescent(stepsize=1/opnorm(Q)),
    "Nesterov" => ProtoGrad.Nesterov(stepsize=1/opnorm(Q)),
    # For BB, all choices of initial stepsize seem OK
    # However, 1/opnorm(Q) seems to "jump" less at the beginning
    "Barzilai-Borwein" => ProtoGrad.BarzilaiBorwein(alpha=1/opnorm(Q)),
    # Adam's performance appears to be sensitive to stepsize
    # It's not clear if there's a good choice based on objective properties
    "Adam" => ProtoGrad.Adam(stepsize=1e-1, beta1=0.9, beta2=0.999, epsilon=1e-8),
)

logging_interval = 1
max_iter = 100

results = OrderedDict(
    optimizer_name => Dict(
        batch_no => let grad = Q*w + q
            Dict(
                "grad" => norm(grad),
                "objective" => 0.5*(dot(w, grad) + dot(w, q)),
            )
        end
        for (batch_no, w) in takenth(
            enumerate(take(optimizer(w0, objective), max_iter)),
            logging_interval
        )
    )
    for (optimizer_name, optimizer) in optimizers
)

# p_grad = plot(
#     [
#         sort(collect(keys(d))) for (optimizer_name, d) in results
#     ],
#     [
#         [d[batch_no]["grad"] for batch_no in sort(collect(keys(d)))]
#         for (optimizer_name, d) in results
#     ],
#     yaxis=:log,
#     ylims=(1e-8, 10),
#     labels=[optimizer_name for (optimizer_name, d) in results],
#     linewidth=2,
#     title="Gradient norm",
# )
# 
# p_objective = plot(
#     [
#         sort(collect(keys(d))) for (optimizer_name, d) in results
#     ],
#     [
#         [
#             max.(1e-16, d[batch_no]["objective"]-f_star)
#             for batch_no in sort(collect(keys(d)))
#         ]
#         for (optimizer_name, d) in results
#     ],
#     yaxis=:log,
#     ylims=(1e-8, 10),
#     labels=[optimizer_name for (optimizer_name, d) in results],
#     linewidth=2,
#     title="Objective",
# )
# 
# plot(p_grad, p_objective, layout=(1, 2))

end