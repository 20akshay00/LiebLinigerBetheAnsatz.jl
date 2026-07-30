"""
    get_magnon_spectrum(γ, c=1.0; quadrature_rule=gausslobatto, N=100, num_points=100, kwargs...)

Calculate the magnon excitation spectrum (dark solitons) for the Lieb-Liniger model using 
the dressed energy and phase shift formalism.

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
function get_magnon_spectrum(γ, c=1.; rho_gs=nothing, Q=nothing, ε=nothing, N=default_quadrature_points(), quadrature_rule=default_quadrature_rule, num_points=100, kwargs...)
    if isnothing(rho_gs) || isnothing(Q)
        rho_gs, _, _, Q = get_ground_state(γ=γ, c=c, kwargs...)
    end

    if isnothing(ε)
        ε, _ = compute_dressed_energy(c, Q; kwargs...)
    end

    xs, ws = rescale(quadrature_rule(N)..., 0., Q)
    n = average_particle_density(rho_gs)
    kf = π * n

    # dressed momentum P(Λ) = kf + ∫ θ(q - Λ)ρ(q)dq
    θ(x) = 2 * atan(2 * x / c)
    P(Λ) = kf + dot(ws, (θ.(xs .- Λ) .- θ.(xs .+ Λ)) .* rho_gs.(xs)) # defined as P(Λ) - P(∞)

    # dressed energy E(Λ) = -∫K(q - Λ)ε(q) dq    
    K(x) = (2 * c) / (π * (c^2 + 4 * x^2)) #dθ/dx
    E(Λ) = -dot(ws, (K.(xs .- Λ) .+ K.(xs .+ Λ)) .* ε.(xs))

    # magnon branch (0 < k < ∞)
    Λs = c .* tan.(range(0, π / 2, length=num_points))
    p_m = P.(Λs)
    e_m = E.(Λs)

    return p_m, e_m
end

magnon_spectrum(s::InfiniteLLState; kwargs...) = get_magnon_spectrum(s.prob.c / s.n, c=s.prob.c; rho_gs=s.rho_k, Q=s.Q, ε=s.eps_k, n=s.n, kwargs...)