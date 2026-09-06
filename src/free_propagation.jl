#Without scaling
@kernel function fresnel_kernel!(ψ, x, y, z, k)
    i, j, l = @index(Global, NTuple)
    ψ[i, j, l] *= cis(-z[l] * (x[i]^2 + y[j]^2) / 2k)
end

#With scaling
@kernel function pre_kernel!(ψ, x, y, z, k, scaling)
    i, j, l = @index(Global, NTuple)
    ψ[i, j, l] *= cis(k * (x[i]^2 + y[j]^2) * (1 - scaling[l]) / 2z[l]) / scaling[l]
end

@kernel function fresnel_kernel!(ψ, x, y, z, k, scaling)
    i, j, l = @index(Global, NTuple)
    ψ[i, j, l] *= cis(-z[l] / scaling[l] * (x[i]^2 + y[j]^2) / 2k)
end

@kernel function post_kernel!(ψ, x, y, z, k, scaling)
    i, j, l = @index(Global, NTuple)
    ψ[i, j, l] *= cis(k * (x[i]^2 + y[j]^2) * (scaling[l] - 1) * scaling[l] / 2z[l])
end

three_d_size(x::AbstractMatrix) = (size(x)..., 1)
three_d_size(x::AbstractArray{T,3}) where {T} = size(x)


"""
    free_propagation(ψ, x, y, z [, scaling]; k=1)
Propagate an inital profile `ψ`.

The propagation is the solution of `∇² ψ + 2ik ∂_z ψ = 0` at distance `z` under the initial condition `ψ`.

`x` and `y` are the grids over which `ψ` is calculated.

If `z` is an `AbstractArray`, the output is a 3D array representing the solution at every element of `z`.

The output at a distance `z[n]` is calculated on a scalled grid defined by `scaling[n] * x` and `scaling[n] * y`.

`k` is the wavenumber.

# Example

```jldoctest
x = LinRange(-10, 10, 256)
y = LinRange(-10, 10, 512)
z = LinRange(0.1, 1, 10)

ψ = hg(x, y; m=3, n=2)
ψ′ = hg(2x, 2y; m=3, n=2)

(
    free_propagation(ψ, x, y, z) ≈ stack(free_propagation(ψ, x, y, z) for z ∈ z)
    &&
    free_propagation(ψ, x, y, z, fill(2, length(z))) ≈ stack(free_propagation(ψ, x, y, z, 2) for z ∈ z)
    &&
    free_propagation(ψ, x, y, 0.5, 2) ≈ free_propagation(ψ′, 2x, 2y, 0.5)
)

# output

true
```
"""
function free_propagation(ψ, x, y, z; k=1)
    free_propagation!(stack(ψ for _ in z), x, y, z; k)
end

"""
    free_propagation!(ψ, x, y, z; k=1, plan=plan_fft!(ψ, (1, 2)), iplan=plan_ifft!(ψ, (1, 2)))

Propagate `ψ` over a distance `z`, overwriting it with the result.

This is the in-place counterpart of [`free_propagation`](@ref), for the case where the output shares
the shape of the input: `ψ` may be a 2D profile propagated to a single `z`, or a 3D stack of
profiles propagated to one `z` each.

`plan` and `iplan` are the in-place forward and inverse transforms over the first two dimensions,
as built by `FFTW.plan_fft!` and `FFTW.plan_ifft!`. Passing them in lets a sequence of propagations
over the same grid reuse a single plan instead of building a new one on every call, so that a
propagation costs a small constant allocation rather than a copy of the field. A plan may only be
applied to an array of the size,
strides and memory alignment it was created from, so build it from the very buffer that is
propagated.

# Example

```jldoctest
x = LinRange(-10, 10, 256)
y = LinRange(-10, 10, 512)

ψ = hg(x, y; m=3, n=2)
ψ′ = copy(ψ)

free_propagation!(ψ′, x, y, 0.5) ≈ free_propagation(ψ, x, y, 0.5)

# output

true
```

See also [`free_propagation`](@ref).
"""
function free_propagation!(ψ, x, y, z; k=1,
    plan=plan_fft!(ψ, (1, 2)), iplan=plan_ifft!(ψ, (1, 2)))
    qx = fftfreq(length(x), 2π / step(x))
    qy = fftfreq(length(y), 2π / step(y))

    backend = get_backend(ψ)
    _fresnel_kernel! = fresnel_kernel!(backend)
    ndrange = three_d_size(ψ)

    plan * ψ
    _fresnel_kernel!(ψ, qx, qy, z, k; ndrange)
    iplan * ψ

    ψ
end

function free_propagation(ψ, x, y, z, scaling; k=1)
    @assert length(z) == length(scaling) "`z` and `scaling` should have the same length"
    @assert 0 ∉ z "This method does not support `z` containing `0`"

    Δx = step(x)
    Δy = step(y)
    Nx = length(x)
    Ny = length(y)
    _x = StepRangeLen(first(x), Δx, Nx)
    _y = StepRangeLen(first(y), Δy, Ny)
    qx = fftfreq(Nx, 2π / Δx)
    qy = fftfreq(Ny, 2π / Δy)

    result = stack(ψ for _ in z)
    backend = get_backend(result)
    _fresnel_kernel! = fresnel_kernel!(backend)
    _pre_kernel! = pre_kernel!(backend)
    _post_kernel! = post_kernel!(backend)
    ndrange = three_d_size(result)

    _pre_kernel!(result, _x, _y, z, k, scaling; ndrange)
    fft!(result, (1, 2))
    _fresnel_kernel!(result, qx, qy, z ./ scaling, k; ndrange)
    ifft!(result, (1, 2))
    _post_kernel!(result, _x, _y, z, k, scaling; ndrange)

    result
end