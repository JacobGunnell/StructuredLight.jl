function propagation_test(mode, xs, ys, zs; atol::Real=0, rtol::Real=atol > 0 ? 0 : √eps(eltype(xs)), kwargs...)
    ψ₀ = mode(xs, ys; kwargs...)
    for z ∈ zs
        ψ₁ = mode(xs, ys, z; kwargs...)
        ψ₂ = free_propagation(ψ₀, xs, ys, z)
        @test isapprox(overlap(ψ₁, ψ₂, xs, ys), 1; rtol, atol)
    end
end

function propagation_test(mode, xs, ys, zs, scaling; atol::Real=0, rtol::Real=atol > 0 ? 0 : √eps(eltype(xs)), kwargs...)
    ψ₀ = mode(xs, ys; kwargs...)
    for z ∈ 2 .* zs
        ψ₁ = mode(2xs, 2ys, z; kwargs...)
        ψ₂ = free_propagation(ψ₀, xs, ys, z, scaling)
        @test isapprox(overlap(ψ₁, ψ₂, xs, ys), 1 / scaling^2; rtol, atol)
    end
end

function run_propagation_tests(xs, ys, identifier=""; atol::Real=0, rtol::Real=atol > 0 ? 0 : √eps(eltype(xs)), kwargs...)
    zs = (0.1, 0.5, 1)
    scaling = 2

    @testset "Hermite Gauss$identifier" begin
        for m ∈ 0:3, n ∈ 0:3
            propagation_test(diagonal_hg, xs, ys, zs; m, n, rtol, kwargs...)
            propagation_test(diagonal_hg, xs, ys, zs, scaling; m, n, rtol, kwargs...)
            for θ ∈ LinRange(0, π, 5)
                propagation_test(hg, xs, ys, zs; m, n, θ, rtol, kwargs...)
                propagation_test(hg, xs, ys, zs, scaling; m, n, rtol, kwargs...)
            end
        end
    end

    @testset "Laguerre Gauss$identifier" begin
        for p ∈ 0:3, l ∈ 0:3
            propagation_test(lg, xs, ys, zs; p, l, rtol, kwargs...)
            propagation_test(lg, xs, ys, zs, scaling; p, l, rtol, kwargs...)
        end
    end
end

xs = LinRange(-20, 20, 1024)
ys = LinRange(-10, 10, 512)
rtol = 0.03

run_propagation_tests(xs, ys; rtol)

if CUDA.functional()
    CUDA.allowscalar(false)

    run_propagation_tests(xs, ys, " (CUDA)"; rtol=rtol, backend=CUDABackend())

    @testset "Linear Combination (CUDA)" begin
        rs = LinRange(-3, 3, 100)
        grid = (rs, rs)
        f1(args) = hg(args..., m=1)
        f2(args) = hg(args..., n=1)

        funcs = (f1, f2)
        coeffs = (1 / √2, -im / √2)

        @test Array(grid_linear_combination(funcs, coeffs, grid, backend=CUDABackend())) ≈ lg(rs, rs, l=-1)
    end
end
@testset "In-place propagation" begin
    xs = LinRange(-20, 20, 256)
    ys = LinRange(-10, 10, 128)
    ψ₀ = hg(xs, ys; m=2, n=1)

    for z ∈ (0.1, 0.5, 1)
        @test free_propagation!(copy(ψ₀), xs, ys, z) ≈ free_propagation(ψ₀, xs, ys, z)
    end

    # a 3D stack is propagated slice by slice, one distance each
    zs = [0.1, 0.5, 1.0]
    @test free_propagation!(stack(ψ₀ for _ ∈ zs), xs, ys, zs) ≈ free_propagation(ψ₀, xs, ys, zs)

    # reusing a plan gives the same answer without copying the field: what is left is the constant
    # overhead of launching the kernel, which does not grow with the size of ψ
    ψ = copy(ψ₀)
    plan = FFTW.plan_fft!(ψ, (1, 2))
    iplan = FFTW.plan_ifft!(ψ, (1, 2))
    @test free_propagation!(ψ, xs, ys, 0.5; plan, iplan) ≈ free_propagation(ψ₀, xs, ys, 0.5)

    copyto!(ψ, ψ₀)
    free_propagation!(ψ, xs, ys, 0.5; plan, iplan)
    copyto!(ψ, ψ₀)
    @test @allocated(free_propagation!(ψ, xs, ys, 0.5; plan, iplan)) < sizeof(ψ) ÷ 100
end

@testset "Overlap" begin
    xs = LinRange(-20, 20, 256)
    ys = LinRange(-10, 10, 128)
    # same indices but a different waist, so the two are not orthogonal and the overlap is O(1)
    ψ = hg(xs, ys; m=2, n=1)
    φ = hg(xs, ys; m=2, n=1, w=1.5)

    # `overlap` conjugates its first argument, and must not copy the field to scale it
    @test overlap(ψ, φ, xs, ys) ≈ step(xs) * step(ys) * sum(conj(ψ) .* φ)
    @test overlap(ψ, φ, xs, ys) ≈ conj(overlap(φ, ψ, xs, ys))
    overlap(ψ, φ, xs, ys)
    @test @allocated(overlap(ψ, φ, xs, ys)) < sizeof(ψ) ÷ 8
end
