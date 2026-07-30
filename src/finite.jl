"""
    get_finite_ground_state(; L, c, N=nothing, μ=nothing, bc=:hardwall)

Orchestrator that finds the ground state by handling either a fixed particle number `N` 
or a fixed chemical potential `μ`.
"""
function get_finite_ground_state(; L, c, N=nothing, μ=nothing, bc=:hardwall)
    if !isnothing(N)
        ns = bc == :periodic ? collect(1:N) .- (N + 1) / 2 : collect(1:N)
        k, G, E = solve_quasimomentum_distribution(c, L, ns; bc=bc)
        return k, G, E, N
    else
        # find N such that E(N+1) - E(N) > μ
        curr_N = 1
        E_prev = 0.0
        while true
            ns = bc == :periodic ? collect(1:curr_N) .- (curr_N + 1) / 2 : collect(1:curr_N)
            _, _, E_curr = solve_quasimomentum_distribution(c, L, ns; bc=bc)
            (E_curr - E_prev > μ) && break
            E_prev = E_curr
            curr_N += 1
        end
        return get_finite_ground_state(L=L, c=c, N=curr_N - 1, bc=bc)
    end
end

"""
    solve_quasimomentum_distribution(c, L, ns; bc=:hardwall, maxiter=50, atol=1e-10, relaxation=0.8)

Solves the Bethe equations for a finite Lieb-Liniger system using a damped Newton method.

# Arguments
- `c`: Coupling constant.
- `L`: System length.
- `ns`: Vector of quantum numbers.
- `bc`: Boundary conditions (:periodic or :hardwall).
- `maxiter`: Maximum Newton iterations.
- `atol`: Absolute tolerance for convergence.
- `relaxation`: Damping factor (α) for stability at high particle counts.

# Returns
- `k`: List of quasi-momenta in the ground state.
- `G`: Gaudin matrix (Jacobian of the Bethe equations).
- `E`: Total energy.
"""
function solve_quasimomentum_distribution(c, L, ns; bc=:hardwall, maxiter=50, atol=1e-10, relaxation=0.8)
    N = length(ns)

    if (N == 0)
        @warn "Attempting to solve a system with zero particles!"
        return 0., 0., 0.
    end

    # non-interacting k values
    k = bc === :periodic ? (2π / L) .* ns : (π / L) .* ns

    # avoid division by zero for c=0 (also force G to be positive-definite)
    c += 1e-12
    θ(x) = atan(x / c)
    K(x) = c / (c^2 + x^2)

    local G = zeros(N, N)
    local F = zeros(N)

    for iter in 1:maxiter
        if bc === :periodic
            for j in 1:N
                # Bethe equation residual: L*kj - 2π*nj + 2sum(θ) = 0
                interact = 2 * sum(θ(k[j] - k[l]) for l in 1:N if l != j; init=0.)
                F[j] = k[j] * L - 2π * ns[j] + interact

                # Jacobian
                for m in 1:N
                    if j == m
                        G[j, j] = L + 2 * sum(K(k[j] - k[l]) for l in 1:N if l != j; init=0.)
                    else
                        G[j, m] = -2 * K(k[j] - k[m])
                    end
                end
            end
        else
            for j in 1:N
                # Bethe equation residual: L*kj - π*nj + sum(θ_minus + θ_plus) = 0
                inter_minus = sum(θ(k[j] - k[l]) for l in 1:N if l != j; init=0.)
                inter_plus = sum(θ(k[j] + k[l]) for l in 1:N if l != j; init=0.)

                F[j] = k[j] * L - π * ns[j] + inter_minus + inter_plus

                # Jacobian
                for m in 1:N
                    if j == m
                        sum_K = sum(K(k[j] - k[l]) + K(k[j] + k[l]) for l in 1:N if l != j; init=0.)
                        G[j, j] = L + sum_K
                    else
                        # dFj / dkm = -K(kj - km) + K(kj + km)
                        G[j, m] = -K(k[j] - k[m]) + K(k[j] + k[m])
                    end
                end
            end
        end

        # Newton iteration
        dk = cholesky(Symmetric(G)) \ F
        k .-= dk * relaxation
        step_norm = sqrt(sum(x -> x^2, dk))
        step_norm < atol && (@info "Converged in $iter iterations."; break)
        (iter == maxiter) && @warn "Solution did not converge! ($step_norm > $atol)"
    end

    return k, G, sum(k .^ 2)
end

## solver interface

"""
    FiniteLLProblem(L, c; N=nothing, μ=nothing, bc=:hardwall, n_q=nothing)

Problem definition for the Lieb-Liniger model with finite system size.
- `L`: The spatial extent of the system.
- `c`: Interaction strength.
- `N`: Number of particles. Specify either `N` or `μ`.
- `μ`: Chemical potential. Specify either `N` or `μ`.
- `bc`: Boundary condition, either `:hardwall` (box) or `:periodic`.
- `n_q`: Custom quantum numbers. If `nothing`, defaults to the ground state:
    - `:hardwall`: `1, 2, ..., N`
    - `:periodic`: centered integers or half-integers `-(N-1)/2, ..., (N-1)/2`.
"""
struct FiniteLLProblem <: LLProblem
    L::Float64
    c::Float64
    N::Union{Nothing,Int}
    μ::Union{Nothing,Float64}
    bc::Symbol
    n_q::Vector{Float64}

    function FiniteLLProblem(L, c; N=nothing, μ=nothing, bc=:hardwall, n_q=nothing)
        if (isnothing(N) && isnothing(μ)) || (!isnothing(N) && !isnothing(μ))
            throw(ArgumentError("Must specify exactly one of N or μ."))
        end

        if !(bc in [:hardwall, :periodic])
            throw(ArgumentError("Boundary condition must either be :hardwall or :periodic."))
        end

        # default quantum numbers for the ground state if not provided
        ns = if isnothing(n_q)
            if isnothing(N)
                Float64[] # to be determined by μ search
            else
                bc == :periodic ? collect(1:N) .- (N + 1) / 2 : collect(1:N)
            end
        else
            n_q
        end

        new(L, c, N, μ, bc, ns)
    end
end

"""
    get_particle_hole_spectrum(s::FiniteLLState; num_points=20)

Calculate the particle and hole excitation spectrum for the Lieb-Liniger model.

# Arguments
- `s`: The ground state `FiniteLLState`.
- `num_points`: Number of discrete shifts to compute for the branches.

# Returns
- `p_h`: Momenta of the hole excitations (from 0 to 2kF).
- `e_h`: Energies of the hole excitations.
- `p_p`: Momenta of the particle excitations.
- `e_p`: Energies of the particle excitations.
"""
function get_particle_hole_spectrum(s::FiniteLLState; num_points=20)
    p = s.prob
    E_gs = s.E
    L, c, bc, N = p.L, p.c, p.bc, s.N

    # ground state total momentum (typically 0)
    ns_gs = bc == :periodic ? collect(1:N) .- (N + 1) / 2 : collect(1:N)
    P_gs = (2π / L) * sum(ns_gs)

    # function to calculate E and P for a shift of the outermost particle
    function get_shift_data(dn)
        ns_exc = copy(ns_gs)
        ns_exc[end] += dn
        k_exc, _, E_exc = solve_quasimomentum_distribution(c, L, ns_exc; bc=bc)
        ΔP = (2π / L) * sum(ns_exc) - P_gs
        ΔE = E_exc - E_gs
        return ΔP, ΔE
    end

    # construct branches
    particle_res = [get_shift_data(dn) for dn in 1:num_points]
    p_p = [r[1] for r in particle_res]
    e_p = [r[2] for r in particle_res]

    # hole branch is limited by the distance to the next quantum number
    max_h = bc == :periodic ? num_points : min(num_points, N - 1)
    hole_res = [get_shift_data(-dn) for dn in 1:max_h]
    p_h = [r[1] for r in hole_res]
    e_h = [r[2] for r in hole_res]

    return p_h, e_h, p_p, e_p
end


"""
    FiniteLLState(prob, k, G, N, E)

Result of a finite system calculation.
- `prob`: The originating `FiniteLLProblem`.
- `k`: Vector of solved quasi-momenta (rapidities).
- `G`: Gaudin matrix (Jacobian of the Bethe equations).
- `N`: Particle number
- `E`: Total energy.
"""
struct FiniteLLState <: LLState
    prob::FiniteLLProblem
    k::Vector{Float64}
    G::Matrix{Float64}
    N::Int
    E::Float64
end

function solve(p::FiniteLLProblem; kwargs...)
    k, G, E, N = get_finite_ground_state(L=p.L, c=p.c, N=p.N, μ=p.μ, bc=p.bc)
    return FiniteLLState(p, k, G, N, E)
end

energy(s::FiniteLLState) = s.E
energy_density(s::FiniteLLState) = s.E / s.prob.L
average_particle_density(s::FiniteLLState) = s.N / s.prob.L
quasimomentum_distribution(s::FiniteLLState) = s.k
fermi_quasimomentum(s::FiniteLLState) = maximum(s.k)