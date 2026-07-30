# solves f(x) - λ ∫([K(x,y) + K(x,-y)] f(y), 0, Q) dy = g(x)
"""
    solve_quasimomentum_distribution(c, Q; N=100, quadrature_rule=gausslobatto)

Solve the Lieb-Liniger integral equation [ρ(x) - 1/2π ∫([K(x,y) + K(x,-y)] ρ(y), 0, Q) dy = 1/2π] for the root density distribution ρ(k) on the interval [0, Q].

Uses a symmetrized kernel to improve quadrature accuracy (assuming ρ(k) = ρ(-k)!).

# Arguments
- `c`: Interaction strength.
- `Q`: Fermi rapidity.
- `N`: Number of quadrature points for the integral solver.
- `quadrature_rule`: Function providing the quadrature nodes and weights.

# Returns
- `rho`: The solved density distribution function ρ(k).
- `particle_density`: Total physical density n.
- `energy_density`: Total energy density E/L.
"""
function solve_quasimomentum_distribution(c, Q; N=default_quadrature_points(), quadrature_rule=default_quadrature_rule())
    # original kernel: 2c / (c^2 + (k-q)^2)
    # symmetrized:     K(k, q) + K(k, -q)
    kernel(k, q) = c / π * (1 / (c^2 + (k - q)^2) + 1 / (c^2 + (k + q)^2))
    kernel_traced(k) = 1 / π * (atan((k + Q) / c) - atan((k - Q) / c))

    rho, xs, ws = solve(
        ModifiedQuadratureSolver(quadrature_rule(N)),
        kernel,
        (x) -> 1. / (2π),
        kernel_traced,
        0.,
        Q
    )

    # density n = 2 * ∫(ρ(k), 0, Q) dk
    particle_density = 2 * dot(ws, rho.(xs))

    # energy E/L = 2 * ∫(k^2 ρ(k), 0, Q) dk
    energy_density = 2 * dot(ws, (xs .^ 2) .* rho.(xs))

    return rho, particle_density, energy_density
end

"""
    get_ground_state(; γ=nothing, μ=nothing, c=1.0, Ql=1e-8, Qh=1., maxiter=100, kwargs...)

Find the ground state properties by solving for the Fermi rapidity Q that satisfies the 
target interaction γ (canonical) or chemical potential μ (grand canonical).

# Arguments
- `γ`: Target dimensionless interaction parameter (c/n).
- `μ`: Target chemical potential.
- `c`: Coupling constant.
- `Ql`: Lower bound for the bisection search of Q.
- `Qh`: Initial upper bound for the bisection search of Q.
- `maxiter`: Maximum iterations to bracket the root for Q.
- `kwargs`: Keyword arguments passed to the distribution solver.

# Returns
- `rho`: The ground state density distribution.
- `e`: Energy density.
- `n`: Particle density.
- `Q`: The determined Fermi rapidity.
"""
function get_ground_state(; γ=nothing, μ=nothing, c=1.0, Ql=1e-8, Qh=1., maxiter=100, kwargs...)
    if !isnothing(γ) && isnothing(μ)
        # particle density
        target = c / γ
        metric_func = Q -> solve_quasimomentum_distribution(c, Q; kwargs...)[2]
    elseif !isnothing(μ) && isnothing(γ)
        # chemical potential
        target = μ
        metric_func = Q -> compute_chemical_potential(c, Q; kwargs...)
    else
        error("Cannot specify both γ (canonical) and μ (grand canonical) at once!")
    end

    function residual(Q)
        (Q <= 1e-9) && return -target
        return metric_func(Q) - target
    end

    # root finding through bisection algorithm
    iter = 0
    while residual(Qh) < 0
        Qh *= 2.0
        iter += 1
        (iter > maxiter) && error("Could not bracket root (Target too high?)")
    end

    Q = find_zero(residual, (Ql, Qh), Bisection())

    rho, n, e = solve_quasimomentum_distribution(c, Q; kwargs...)
    return rho, e, n, Q
end

"""
    compute_dressed_energy(c, Q; N=100, quadrature_rule=gausslobatto)

Solve the dressed energy equation  ε(k) - ∫ K(k - q) ε(q) dq = k^2 - μ , where the chemical potential μ is determined from the boundary condition ε(Q) = ε₀(Q) - μ ε₁(Q) = 0.

# Arguments
- `c`: Interaction strength.
- `Q`: Fermi rapidity.
- `N`: Number of quadrature points.
- `quadrature_rule`: Function providing the quadrature nodes and weights.

# Returns
- `ε`: The dressed energy function.
- `μ`: The determined chemical potential.
"""
function compute_dressed_energy(c, Q; N=default_quadrature_points(), quadrature_rule=default_quadrature_rule())
    kernel(k, q) = c / π * (1 / (c^2 + (k - q)^2) + 1 / (c^2 + (k + q)^2))
    kernel_traced(k) = 1 / π * (atan((k + Q) / c) - atan((k - Q) / c))

    solver = ModifiedQuadratureSolver(quadrature_rule(N))

    # solve auxiliary equations:
    #   ε₀(k) - ∫ K(k, q) ε₀(q) dq = k²
    #   ε₁(k) - ∫ K(k, q) ε₁(q) dq = 1
    ε₀, _, _ = solve(solver, kernel, k -> k^2, kernel_traced, 0., Q)
    ε₁, _, _ = solve(solver, kernel, k -> 1.0, kernel_traced, 0., Q)

    # ε(k) - ∫ K(k - q) ε(q) dq = k^2 - μ
    # enforce ε(Q) = 0 to find μ
    μ = ε₀(Q) / ε₁(Q)

    return (k) -> ε₀(k) - μ * ε₁(k), μ
end

"""
    compute_chemical_potential(c, Q; N=100, quadrature_rule=gausslobatto, kwargs...)

Helper function to calculate the chemical potential μ for a given interaction strength c and rapidity cutoff Q.
"""
function compute_chemical_potential(c, Q; N=default_quadrature_points(), quadrature_rule=default_quadrature_rule(), kwargs...)
    _, μ = compute_dressed_energy(c, Q; kwargs...)
    return μ
end

"""
    get_particle_hole_spectrum(γ, c=1.; quadrature_rule=gausslobatto, N=100, num_points=100, kwargs...)

Calculate the particle and hole excitation spectrum for the Lieb-Liniger model.

# Arguments
- `γ`: Dimensionless interaction parameter.
- `c`: Coupling constant.
- `quadrature_rule`: Function providing the quadrature nodes and weights.
- `N`: Number of quadrature points for the integral solvers.
- `num_points`: Number of points to sample in the momentum/energy curves.
- `kwargs`: Additional keyword arguments passed to internal solvers.

# Returns
- `p_h`: Momenta of the hole excitations (from 0 to 2kF).
- `e_h`: Energies of the hole excitations.
- `p_p`: Momenta of the particle excitations.
- `e_p`: Energies of the particle excitations.
"""
function get_particle_hole_spectrum(γ, c=1.; rho_gs=nothing, Q=nothing, ε=nothing, n=nothing, N=default_quadrature_points(), quadrature_rule=default_quadrature_rule(), num_points=100, kwargs...)
    if isnothing(rho_gs) || isnothing(Q) || isnothing(n)
        rho_gs, _, _, Q = get_ground_state(γ=γ, c=c, kwargs...)
    end

    if isnothing(ε)
        ε, _ = compute_dressed_energy(c, Q, kwargs...)
    end

    xs, ws = rescale(quadrature_rule(N)..., 0., Q)
    kf = π * n

    # dressed momentum P(k) = k + ∫ θ(k-q)ρ(q)dq
    θ(x) = 2 * atan(x / c)
    P(k) = k + dot(ws, (θ.(k .- xs) .+ θ.(k .+ xs)) .* rho_gs.(xs))

    # Type I (particle) branch (k > Q)
    k_p = range(Q, 3 * Q, length=num_points)
    p_p = P.(k_p) .- kf # add a hole w/ energy 0
    e_p = ε.(k_p)

    # Type II (hole) branch (k ∈ [Q, -Q])
    # maps to [0, 2kf] momentum since P(±Q) = ±kf = ±πn
    k_h = range(Q, -Q, length=2 * num_points)
    p_h = kf .- P.(k_h) # add a particle w/ energy 0
    e_h = -ε.(k_h)

    return p_h, e_h, p_p, e_p
end

"""
    get_magnon_spectrum(γ, c=1.0; quadrature_rule=gausslobatto, N=100, num_points=100, kwargs...)

Calculate the magnon excitation spectrum for the Yang-Gaudin model.

# Arguments
- `γ`: Dimensionless interaction strength (c/n).
- `c`: Coupling constant.
- `quadrature_rule`: Integration rule for the solvers.
- `N`: Number of quadrature points.
- `num_points`: Number of points used to sample the spectrum.
- `kwargs`: Keyword arguments passed to ground state and dressed energy solvers.

# Returns
- `p_m`: Physical momentum of the magnon branch.
- `e_m`: Excitation energy of the magnon branch.
"""
function get_magnon_spectrum(γ, c=1.; rho_gs=nothing, Q=nothing, ε=nothing, n=nothing, N=default_quadrature_points(), quadrature_rule=default_quadrature_rule, num_points=100, kwargs...)
    if isnothing(rho_gs) || isnothing(Q) || isnothing(n)
        rho_gs, _, n, Q = get_ground_state(γ=γ, c=c, kwargs...)
    end

    if isnothing(ε)
        ε, _ = compute_dressed_energy(c, Q; kwargs...)
    end

    xs, ws = rescale(quadrature_rule(N)..., 0., Q)
    kf = π * n

    # dressed momentum P(Λ) = kf + ∫ θ(q - Λ)ρ(q)dq
    θ(x) = 2 * atan(2 * x / c)
    P(Λ) = kf + dot(ws, (θ.(xs .- Λ) .- θ.(xs .+ Λ)) .* rho_gs.(xs)) # defined s.t. P(∞)=0

    # dressed energy E(Λ) = -∫K(q - Λ)ε(q) dq    
    K(x) = (2 * c) / (π * (c^2 + 4 * x^2)) #dθ/dx
    E(Λ) = -dot(ws, (K.(xs .- Λ) .+ K.(xs .+ Λ)) .* ε.(xs))

    # magnon branch (0 < k < ∞)
    Λs = c .* tan.(range(0, π / 2, length=num_points))
    p_m = P.(Λs)
    e_m = E.(Λs)

    return p_m, e_m
end

## solver interface

"""
    InfiniteLLProblem(c; γ=nothing, μ=nothing)

Problem definition for the Lieb-Liniger model in the thermodynamic limit.
- `c`: Interaction strength.
- `γ`: Dimensionless interaction (c/n).
- `μ`: Chemical potential.
Exactly one of `γ` or `μ` must be provided.
"""
struct InfiniteLLProblem <: LLProblem
    c::Float64
    γ::Union{Nothing,Float64}
    μ::Union{Nothing,Float64}
    function InfiniteLLProblem(; c=1., γ=nothing, μ=nothing)
        if (isnothing(γ) == isnothing(μ))
            throw(ArgumentError("Must specify exactly one of γ (canonical) or μ (grandcanonical)."))
        end
        new(c, γ, μ)
    end
end

"""
    InfiniteLLState(prob, Q, rho_k, eps_k, n, e, μ)

Result of a thermodynamic limit calculation.
- `prob`: The originating `InfiniteLLProblem`.
- `Q`: Fermi rapidity cutoff.
- `rho_k`: Root distribution ρ(k).
- `eps_k`: Dressed energy ε(k).
- `n`: Particle density.
- `e`: Energy density.
- `μ`: Chemical potential.
"""
struct InfiniteLLState <: LLState
    prob::InfiniteLLProblem
    Q::Float64
    rho_k::Function
    eps_k::Function
    n::Float64
    e::Float64
    μ::Float64
end

function solve(p::InfiniteLLProblem; kwargs...)
    rho, e, n, Q = get_ground_state(γ=p.γ, μ=p.μ, c=p.c; kwargs...)
    ε, μ_calc = compute_dressed_energy(p.c, Q; kwargs...)
    return InfiniteLLState(p, Q, rho, ε, n, e, μ_calc)
end

energy(s::InfiniteLLState) = Inf
energy_density(s::InfiniteLLState) = s.e
average_particle_density(s::InfiniteLLState) = s.n
particle_density(s::InfiniteLLState) = x -> s.n
quasimomentum_distribution(s::InfiniteLLState) = s.rho_k
fermi_quasimomentum(s::InfiniteLLState) = s.Q
get_particle_hole_spectrum(s::InfiniteLLState; kwargs...) = get_particle_hole_spectrum(s.prob.c / s.n, s.prob.c; n=s.n, rho_gs=s.rho_k, Q=s.Q, ε=s.eps_k, kwargs...)
get_magnon_spectrum(s::InfiniteLLState; kwargs...) = get_magnon_spectrum(s.prob.c / s.n, c=s.prob.c; rho_gs=s.rho_k, Q=s.Q, ε=s.eps_k, n=s.n, kwargs...)








