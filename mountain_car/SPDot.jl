module SPDot

export spdot


function spdot(v1::SparseMatrixCSC{Float64, Int64},
               v2::SparseMatrixCSC{Float64, Int64})
    sum = 0.0
    if length(v1.rowval) < length(v2.rowval)
        for iv1 in 1:length(v1.rowval)
            sum += v1.nzval[iv1] * v2[v1.rowval[iv1]]
        end # for iv1
    else
        for iv2 in 1:length(v2.rowval)
            sum += v2.nzval[iv2] * v1[v2.rowval[iv2]]
        end # for iv2
    end # if
    return sum
end # function spdot


function spdot(v1::Vector{Float64},
               v2::SparseMatrixCSC{Float64, Int64})
    sum = 0.0
    for iv2 in 1:length(v2.rowval)
        sum += v2.nzval[iv2] * v1[v2.rowval[iv2]]
    end # for iv2
    return sum
end # function spdot


function spdot(m1::Matrix{Float64}, icol::Int64,
               v2::SparseMatrixCSC{Float64, Int64})
    sum = 0.0
    for iv2 in 1:length(v2.rowval)
        sum += v2.nzval[iv2] * m1[v2.rowval[iv2], icol]
    end # for iv2
    return sum
end # function spdot


function spdot(v1::SparseMatrixCSC{Float64, Int64},
               v2::Vector{Float64})
    sum = 0.0
    for iv1 in 1:length(v1.rowval)
        sum += v1.nzval[iv1] * v2[v1.rowval[iv1]]
    end # for iv1
    return sum
end # function spdot


function spdot(v1::SparseMatrixCSC{Float64, Int64},
               m2::Matrix{Float64}, icol::Int64)
    sum = 0.0
    for iv1 in 1:length(v1.rowval)
        sum += v1.nzval[iv1] * m2[v1.rowval[iv1], icol]
    end # for iv1
    return sum
end # function spdot

end # module