using ProtoGrad
using Test
using Aqua
using JET
using Zygote

@testset "Aqua" begin
    Aqua.test_all(ProtoGrad; ambiguities = false)
end

@testset "JET" begin
    JET.test_package(ProtoGrad)
end

include("test_utils.jl")
include("test_models.jl")
include("test_optimizers.jl")
include("test_fine_tuning.jl")
