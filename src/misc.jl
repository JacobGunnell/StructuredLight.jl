"""
    overlap(ψ₁,ψ₂,xs,ys)

Calculate ∫ `conj(ψ₁)` `ψ₂` dx dy.

`xs` and `ys` are the grids over which `ψ₁` and `ψ₂` are calculated.
"""
function overlap(ψ₁, ψ₂, xs, ys)
    # the parentheses are load bearing: `*` and `⋅` have equal precedence and associate to the left,
    # so without them the scalar is multiplied into the whole of `ψ₁` before the dot product, and
    # every call allocates a copy of the field
    step(xs) * step(ys) * (reshape(ψ₁, :) ⋅ reshape(ψ₂, :))
end

"""
    overlap(ψ₁::AbstractArray{T1,3},ψ₂::AbstractArray{T2,3},xs,ys) where {T1,T2}

Calculate ∫ `conj(ψ₁[:,:,j])` `ψ₂[:,:,j]` dx dy, where j runs over all indices of the third dimension. 

The output is a vector.

`xs` and `ys` are the grids over which `ψ₁` and `ψ₂` are calculated.
"""
function overlap(ψ₁::AbstractArray{T1,3}, ψ₂::AbstractArray{T2,3}, xs, ys) where {T1,T2}
    map((a, b) -> overlap(a, b, xs, ys), eachslice(ψ₁, dims=3), eachslice(ψ₂, dims=3))
end