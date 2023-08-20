using ProtoGrad
using Test
using Aqua
using JET

@testset "Aqua" begin
    Aqua.test_all(ProtoGrad; ambiguities=false)
end

if VERSION >= v"1.9.0"
    @testset "JET" begin
        JET.test_package(ProtoGrad)
    end
end

include("test_utils.jl")
include("test_models.jl")
include("test_optimizers.jl")
include("test_fine_tuning.jl")
