using ProtoGrad
using Test
using Aqua
using JET

@testset "Aqua" Aqua.test_all(ProtoGrad; ambiguities=false)

if VERSION >= v"1.9.0"
    @testset "JET" JET.test_package(ProtoGrad)
end

include("test_utils.jl")
include("test_models.jl")
include("test_optimizers.jl")
include("test_fine_tuning.jl")
