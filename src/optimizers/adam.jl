@kwdef struct Adam{R}
    stepsize::R
    beta1::R = 0.9
    beta2::R = 0.999
    epsilon::R = 1e-8
end

mutable struct AdamState{R,V}
    optimizer::Adam{R}
    stepsize::R
    variable::V
    m::V
    v::V
    m_hat::V
    v_hat::V
    it::Int
end

function init(optimizer::Adam, variable)
    return AdamState(
        optimizer,
        optimizer.stepsize,
        variable,
        zero(variable),
        zero(variable),
        zero(variable),
        zero(variable),
        1,
    )
end

function step!(state::AdamState, gradient)
    state.m .= state.optimizer.beta1 .* state.m .+ (1 - state.optimizer.beta1) .* gradient
    state.v .=
        state.optimizer.beta2 .* state.v .+ (1 - state.optimizer.beta2) .* (gradient .^ 2)
    state.m_hat .= state.m ./ (1 - state.optimizer.beta1^state.it)
    state.v_hat .= state.v ./ (1 - state.optimizer.beta2^state.it)
    state.variable .-=
        state.stepsize .* (state.m_hat ./ (sqrt.(state.v_hat) .+ state.optimizer.epsilon))
    return state.it += 1
end
