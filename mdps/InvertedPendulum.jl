module InvertedPendulum

using MDPs, PGFPlots

export state_space, action_space, transition, reward, simulation,
       viz_policy, viz_trajectory


# data file
const Q_CSV = "../data/q_inverted_pendulum.csv"


function state_space()

end # function state_space


function action_space()

end # function action_space


function transition(state::Vector{Float64}, action::Float64)

end # function transition


function reward(state::Vector{Float64}, action::Float64)

end # function reward


function simulation()

end # function simulation


function viz_policy()

end # function viz_policy


function viz_trajectory()

end # function viz_trajectory

end # module