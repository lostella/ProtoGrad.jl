@kwdef struct GradientDescent{S}
    stepsize::S
end

struct GradientDescentState{S,V}
    optimizer::GradientDescent{S}
    stepsize::S
    variable::V
end

function init(optimizer::GradientDescent, variable)
    return GradientDescentState(optimizer, optimizer.stepsize, variable)
end

function step!(state::GradientDescentState, gradient)
    return state.variable .-= state.stepsize .* gradient
end
