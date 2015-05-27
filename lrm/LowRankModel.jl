module LowRankModel

push!(LOAD_PATH, "../mdps")

using MDPs, PGFPlots

export sparsify, rankify, viz_policies, viz_trajectories


const SPTOL = 1e-6
const EPS = 1e-4


function sparsify(M::Matrix{Float64}, tol::Float64=SPTOL)
    m, n = size(M)
    Msp = spzeros(m, n)
    for i = 1:m
        for j = 1:n
            if abs(M[i, j]) > tol
                Msp[i, j] = M[i, j]
            end # if
        end # for j
    end # for i
    return Msp
end # function sparsify


function sparsify(M::Matrix{Float64})
    return sparse(M)
end # function sparsify


function rankify(M::Matrix{Float64}, rank::Int64)
    U, S, V = svd(M)
    return U[:, 1:rank], S[1:rank], V[:, 1:rank]
end # function rankify


function viz_policies(mdp::MDP, p1::Policy, p2::Policy, minx::Float64, 
                      maxx::Float64, miny::Float64, maxy::Float64,
                      xlabel::ASCIIString="", ylabel::ASCIIString="")
    function getmap1(x::Float64, y::Float64)
        return policy!(p1, get_belief(mdp, [x, y]))
    end # function getmap1

    function getmap2(x::Float64, y::Float64)
        return policy!(p2, get_belief(mdp, [x, y]))
    end # function getmap2

    g = GroupPlot(2, 1, groupStyle="horizontal sep=2cm")
    push!(g, Axis([
        Plots.Image(getmap1, (minx + EPS, maxx - EPS), 
                    (miny + EPS, maxy - EPS),
                    xbins=250, ybins=250,
                    colormap=ColorMaps.Named("RdBu"),
                    colorbar=false)
        ], width="10cm", height="10cm", title="Original policy",
           xlabel=xlabel, ylabel=ylabel))
    push!(g, Axis([
        Plots.Image(getmap2, (minx + EPS, maxx - EPS), 
                    (miny + EPS, maxy - EPS),
                    xbins=250, ybins=250,
                    colormap=ColorMaps.Named("RdBu"))
        ], width="10cm", height="10cm", title="Low-rank + sparse policy",
           xlabel=xlabel, ylabel=ylabel))
    g
end # function viz_policies


function viz_trajectories(traj1::Matrix{Float64}, act1::Vector{Float64}, 
                          traj2::Matrix{Float64}, act2::Vector{Float64})
    g = GroupPlot(2, 2, groupStyle="horizontal sep=2cm, vertical sep=2cm")
    push!(g, Axis([Plots.Linear(traj1[:, 1])],
                  width="10cm", height="10cm",
                  xlabel="time", ylabel="angle (deg)",
                  title="Original policy trajectory"))
    push!(g, Axis([Plots.Linear(act1)],
                  width="10cm", height="10cm",
                  xlabel="time", ylabel="input",
                  title="Original policy input"))
    push!(g, Axis([Plots.Linear(traj2[:, 1])],
                  width="10cm", height="10cm",
                  xlabel="time", ylabel="angle (deg)",
                  title="Low-rank + sparse policy trajectory"))
    push!(g, Axis([Plots.Linear(act2)],
                  width="10cm", height="10cm",
                  xlabel="time", ylabel="input",
                  title="Low-rank + sparse policy input"))
    g
end # function viz_trajectories

end # module