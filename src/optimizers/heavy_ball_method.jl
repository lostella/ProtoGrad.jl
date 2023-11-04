@kwdef struct HeavyBallMethod{S}
    stepsize::S
    momentum::S
end

struct HeavyBallMethodState{S,V}
    optimizer::HeavyBallMethod{S}
    stepsize::S
    variable::V
    step::V
end

init(optimizer::HeavyBallMethod, variable) =
    HeavyBallMethodState(optimizer, optimizer.stepsize, variable, zero(variable))

function step!(state::HeavyBallMethodState, gradient)
    state.step .= state.optimizer.momentum .* state.step .- state.stepsize .* gradient
    state.variable .+= state.step
end
