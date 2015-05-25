module MDPs

using GridInterpolations

export MDP, Policy, policy!, value_iteration, get_belief, RectangleGrid


# value iteration
const VTOL = 1e-6
const MAXITER = 500
const GAMMA = 0.95


type MDP
    S           :: RectangleGrid
    nstate      :: Int64
    A           :: Vector{Float64}
    naction     :: Int64
    transition  :: Function
    reward      :: Function
    
    function MDP(S, A, transition, reward)
        return new(S, length(S), A, length(A), transition, reward)
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


function saveq(Q::Matrix{Float64})
    writecsv(Q_CSV, Q)
end # function saveq


function value_iteration(mdp::MDP, save::Bool=false, verbose::Bool=true)
    nstate = length(mdp.S)
    naction = length(mdp.A)
    V = zeros(nstate)
    Q = zeros(nstate, naction)
    state = zeros(dimensions(mdp.S))
    cputime = 0.0
    
    println("Starting value iteration...")
    iter = 0
    for iter = 1:MAXITER
        tic()
        residual = 0.0
        for istate = 1:nstate
            for iaction = 1:naction
                ind2x!(mdp.S, istate, state)
                action = mdp.A[iaction]
                snext = mdp.transition(state, action)
                vnext = interpolate(mdp.S, V, snext)
                Qprev = Q[istate, iaction]
                Q[istate, iaction] = mdp.reward(state) + GAMMA * vnext
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
    @printf("Value iteration took %d iterations and %.2e sec\n", iter, cputime)

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

end # module