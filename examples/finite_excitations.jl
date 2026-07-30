using LiebLinigerBetheAnsatz
using LinearAlgebra, Plots, LaTeXStrings

begin # single parameter
    L, c, N = 1., 0.1, 100
    @time state = solve(FiniteLLProblem(L, c, N=N, bc=:periodic))
    rho_k = quasimomentum_distribution(state)
    e = energy_density(state)
    n = average_particle_density(state)
    Q = fermi_quasimomentum(state)
    println(energy(state))
end

begin
    p_h, E_h, p_p, E_p = get_particle_hole_spectrum(state; num_points=100)
end;

begin
    γ = c/state.N
    Eph(p) = sqrt(4 * γ * n^2 * p^2 + p^4)

    plot(xtick=pitick(0, 2π, 4, mode=:latex))

    scatter!(p_p ./ n, E_p ./ n^2, label="Type II (Particles)", lw=2, c=1, ms=3)
    plot!(p_p ./ n, Eph.(p_p) ./ n^2, ls=:dash, c=:black, lw=1.5, lab="BdG")

    c_sol = sqrt(4 * γ * n^2)
    Esol(v) = (4 / 3) * n * c_sol * (1 - (v / c_sol)^2)^(3 / 2)
    psol(v) = 2 * n * (acos(v / c_sol) - (v / c_sol) * sqrt(1 - (v / c_sol)^2))
    vs = range(-c_sol, c_sol, 100)

    plot!(psol.(vs) ./ n, Esol.(vs) ./ n^2, c=:black, ls=:dash, lw=1.5, lab="Soliton")
    scatter!(p_h ./ n, E_h ./ n^2, label="Type I (Holes)", lw=2, title="Lieb-Liniger Spectrum (γ=$γ)", c=0, ms=3)

    xlabel!("Momentum " * L"p/ρ")
    ylabel!("Energy " * L"\epsilon/ρ^2")
    plot!(framestyle=:box)
end