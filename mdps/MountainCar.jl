module MountainCar

using MDPs, PGFPlots

export state_space, action_space, transition, reward, viz_policy, 
       simulation, viz_trajectory


# mdp
const MX = 100
const MV = 100
const N = 2500

# state
const DSTATE = 2
const XMIN = -1.2
const XMAX = 0.5
const VMIN = -0.07
const VMAX = 0.07

# action
const AMIN = -1.0
const AMAX = 1.0

# reward
const TERM_REWARD = 10
const NONTERM_REWARD = -1

# test
const T = 1000

# data file
const Q_CSV = "../data/q_mountain_car.csv"


function state_space(mx::Int64=MX, mv::Int64=MV)
    xs = linspace(XMIN, XMAX, mx)
    vs = linspace(VMIN, VMAX, mv)
    return RectangleGrid(xs, vs)
end # function state_space


function action_space(n::Int64=N)
    return linspace(AMIN, AMAX, n)
end # function action_space


function transition(state::Vector{Float64}, action::Float64)
    snext = zeros(DSTATE)
    snext[2] = state[2] + 0.001 * action - 0.0025 * cos(3 * state[1])
    snext[2] = clip(snext[2], VMIN, VMAX)
    snext[1] = clip(state[1] + snext[2], XMIN, XMAX)
    return snext
end # function transition


function clip(val::Float64, minval::Float64, maxval::Float64)
    return min(maxval, max(minval, val))
end # function clip


function reward(state::Vector{Float64}, action::Float64)
    if state[1] == XMAX
        return TERM_REWARD
    else
        return NONTERM_REWARD
    end # if
end # function reward


function simulation(mdp::MDP, policy::Policy, state::Vector{Float64})
    trajectory = zeros(T, DSTATE)
    actions = zeros(T - 1)
    for t = 1:T
        trajectory[t, :] = state
        if state[1] == XMAX
            trajectory = trajectory[1:t, :]
            actions = actions[1:t - 1]
            break
        end # if
        action = policy!(policy, get_belief(mdp, state))
        state = transition(state, action)
        actions[t] = action
    end # for t
    return trajectory, actions
end # function simulation


function viz_policy(mdp::MDP, policy::Policy)
    function getmap(x::Float64, v::Float64)
        return policy!(policy, get_belief(mdp, [x, v]))
    end # function getmap

    Axis([Plots.Image(getmap, (XMIN, XMAX), (VMIN, VMAX),
                      xbins = 250, ybins = 250,
                      colormap = ColorMaps.Named("RdBu"))],
          width="9cm", height="8cm",
          xlabel="position", ylabel="speed",
          title="Policy heatmap")
end # function viz_policy


function viz_trajectory(trajectory::Matrix{Float64}, actions::Vector{Float64})
    g = GroupPlot(2, 1, groupStyle="horizontal sep=1.5cm")
    push!(g, Axis(Plots.Linear(trajectory[:, 1]),
                  width="9cm", height="8cm",
                  xlabel="time", ylabel="position",
                  title="position plot"))
    push!(g, Axis(Plots.Linear(actions),
                  width="9cm", height="8cm",
                  xlabel="time", ylabel="acceleration",
                  title="acceleration plot"))
    g
end # function viz_trajectory

end # module