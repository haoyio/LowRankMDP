module MountainCar

using GridInterpolations, SPDot, PGFPlots

export MDP, Policy, policy!, state_space, action_space, transition, 
       reward, value_iteration, viz_policy, simulation, viz_trajectory


# mdp
const MX = 100
const MV = 100
const N = 1000

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

# value iteration
const VTOL = 1e-6
const MAXITER = 500
const GAMMA = 0.95

# test
const T = 1000

# data file
const Q_CSV = "qvalue.csv"


type MDP
    S   :: RectangleGrid
    A   :: Vector{Float64}
    nstate  :: Int64
    naction :: Int64

    function MDP(S, A)
        return new(S, A, length(S), length(A))
    end # function MDP
end # type MDP


type Policy
    Q       :: Matrix{Float64}
    A       :: Vector{Float64}
    naction :: Int64
    qvals   :: Vector{Float64}
    
    function Policy(Q, A)
        return new(Q, A, length(A), zeros(length(A)))
    end # function Policy
end # type Policy


function policy!(policy::Policy, belief::SparseMatrixCSC{Float64, Int64})
    fill!(policy.qvals, 0.0)
    for iaction in 1:policy.naction
        for ib in 1:length(belief.rowval)
            policy.qvals[iaction] += 
                belief.nzval[ib] * policy.Q[belief.rowval[ib], iaction]
        end # for b
    end # for iaction
    ibest = indmax(policy.qvals)
    return policy.A[ibest]
end # function policy!


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


function reward(state::Vector{Float64})
    if state[1] == XMAX
        return TERM_REWARD
    else
        return NONTERM_REWARD
    end # if
end # function reward


function saveq(Q::Matrix{Float64})
    writecsv(Q_CSV, Q)
end # function saveq


function value_iteration(mdp::MDP, save::Bool=false, verbose::Bool=true)
    nstate = length(mdp.S)
    naction = length(mdp.A)
    V = zeros(nstate)
    Q = zeros(nstate, naction)
    state = zeros(DSTATE)
    cputime = 0.0
    
    println("Starting value iteration...")
    for iter = 1:MAXITER
        tic()
        residual = 0.0
        for istate = 1:nstate
            for iaction = 1:naction
                ind2x!(mdp.S, istate, state)
                action = mdp.A[iaction]
                snext = transition(state, action)
                vnext = interpolate(mdp.S, V, snext)
                Qprev = Q[istate, iaction]
                Q[istate, iaction] = reward(state) + GAMMA * vnext
                residual += (Q[istate, iaction] - Qprev)^2
            end # for iaction
        end # for istate

        V = [maximum(Q[istate, :]) for istate = 1:nstate]
        
        iter_time = toq()
        cputime += iter_time
        
        if verbose
            @printf("Iteration %d: residual = %.2e, cputime = %.2e sec\n", 
                    iter, residual, iter_time)
        end # if
        
        if residual < VTOL
            break
        end # if

        if iter == MAXITER
            println("Warning: maximum number of iterations reached; ",
                    "solution may be inaccurate")
        end # if
    end # for iter
    @printf("Value iteration completed; cputime = %.2e sec\n", cputime)

    if save
        saveq(Q)
    end # if

    return Policy(Q, mdp.A)
end # function value_iteration


function get_belief(mdp::MDP, state::Vector{Float64})
    belief = spzeros(mdp.nstate, 1)
    indices, weights = interpolants(mdp.S, state)
    belief[indices] = weights
    return belief
end # function get_belief


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


function viz_trajectory(trajectory::Matrix{Float64}, actions::Vector{Float64})
    g = GroupPlot(2, 1)
    push!(g, Axis(Plots.Linear(trajectory[:, 1])))
    push!(g, Axis(Plots.Linear(actions)))
    g
end # function viz_trajectory

end # module