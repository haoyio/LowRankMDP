module InvertedPendulum

using MDPs, PGFPlots

export state_space, action_space, transition, reward, simulation,
       viz_policy, viz_trajectory
export PMIN, PMAX, VMIN, VMAX


# mdp
const MP = 50
const MV = 50
const N = 1000

# state
const PMIN = float(-pi)
const PMAX = float(pi)
const VMIN = -10.0
const VMAX = 10.0

# action
const AMIN = -1.0
const AMAX = 1.0

# transition
const A = 1.0
const B = 1.0
const DT = MV / MP * pi / VMAX  # for resolution
const NOISE = deg2rad(0.5)

# reward
const K = 1.0
const R = 0.1

# test
const T = 100

# visualization
const EPS = 1e-4


function state_space(mp::Int64=MP, mv::Int64=MV)
    p = linspace(PMIN, PMAX, mp)
    v = linspace(VMIN, VMAX, mv)
    return RectangleGrid(p, v)
end # function state_space


function action_space(n::Int64=N)
    return linspace(AMIN, AMAX, n)
end # function action_space


function wrap_around(angle::Float64)
    while angle < -pi
        angle += 2 * pi
    end # while
    while angle > pi
        angle -= 2 * pi
    end # while
    return angle
end # function wrap_around


function transition(state::Vector{Float64}, action::Float64, noise::Float64=0.0)
    snext = zeros(length(state))
    snext[1] = wrap_around(state[1] + DT * state[2])
    snext[2] = state[2] + DT * (A * sin(state[1]) - B * state[2] + action) + noise
    return snext
end # function transition


function reward(state::Vector{Float64}, action::Float64)
    return exp(K * (cos(state[1]) - 1)) - R * action^2 - 1
end # function reward


function simulation(mdp::MDP, policy::Policy, state::Vector{Float64})
    trajectory = zeros(T, dimensions(mdp.S))
    actions = zeros(T)
    for t = 1:T
        trajectory[t, :] = state
        if state[1] == 0.0
            trajectory = trajectory[1:t, :]
            actions = actions[1:t - 1]
            break
        end # if
        action = policy!(policy, get_belief(mdp, state))
        state = mdp.transition(state, action, NOISE * randn())
        actions[t] = action
    end # for t
    return trajectory, actions
end # function simulation


function viz_policy(mdp::MDP, policy::Policy)
    function getmap(p::Float64, v::Float64)
        return policy!(policy, get_belief(mdp, [p, v]))
    end # function getmap

    Axis([Plots.Image(getmap, (PMIN + EPS, PMAX - EPS), 
                      (VMIN + EPS, VMAX - EPS),
                      xbins = 250, ybins = 250,
                      colormap = ColorMaps.Named("RdBu"))],
          width="12cm", height="12cm",
          xlabel="angle", ylabel="angular speed",
          title="Policy heatmap")
end # function viz_policy


function viz_trajectory(trajectory::Matrix{Float64}, actions::Vector{Float64})
    g = GroupPlot(2, 1, groupStyle="horizontal sep=1.5cm")
    push!(g, Axis([Plots.Linear(rad2deg(trajectory[:, 1]))],
                  width="10cm", height="10cm",
                  xlabel="time", ylabel="angle (deg)",
                  title="angular position over time"))
    push!(g, Axis([Plots.Linear(actions)],
                  width="10cm", height="10cm",
                  xlabel="time", ylabel="input",
                  title="control input over time"))
    g
end # function viz_trajectory

end # module