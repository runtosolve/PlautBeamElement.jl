using PlautBeamElement: Section, Material, Restraint, BeamModel, BeamSolution,
                              elastic_stiffness, geometric_stiffness, load_vector,
                              constrained_dofs, solve_beam, solve_beam_linear, sparse_stiffness,
                              deflections, twist,
                              lateral_deflection, vertical_deflection, element_end_forces,
                              critical_load, prebuckling_moments
using Test
using LinearAlgebra

# ==================================================================
# Reference data from Plaut & Moen (2021), JCSR 179, 106534
# Units: N, mm (1 kN/m = 1 N/mm; E = 200 kN/mm² = 200e3 N/mm²)
# ==================================================================

const L = 7620.0
const mat = Material(E = 200e3)          # G = E/2.6, Section 3

# Example 1: 9CS2.5×059 C-section (Section 3)
const sec1 = Section(Ix = 4.287e6, Iy = 290500.0, Ixy = 0.0,
                     J = 424.56, Cw = 3.1956e9, ax = 57.67, ay = 114.3)

# Example 2: 8ZS2.25×059 Z-section (Section 4; Ixy negative with y downward)
const sec2 = Section(Ix = 3.230e6, Iy = 449530.0, Ixy = -865760.0,
                     J = 397.09, Cw = 3.4104e9, ax = 27.826, ay = 101.6)

"""
Exact Fourier sine-series solution of Eqs. (2)-(4) for a pinned-pinned beam
(u = u'' = v = v'' = ϕ = ϕ'' = 0 at both ends). With constant coefficients the
equations decouple mode by mode: substituting (u, v, ϕ) = Σ (uₙ, vₙ, ϕₙ) sin(λₙ z),
λₙ = nπ/L, and expanding the uniform load, 1 = Σ (4/nπ) sin(λₙ z) over odd n,
gives a 3×3 linear system per mode.
"""
function series_solution(z; L, section, material, restraint, q, θ = 0.0, nmodes = 399)
    (; E, G) = material
    (; Ix, Iy, Ixy, J, Cw, ax, ay) = section
    (; kx, kϕ) = restraint
    qx, qy = -q * sin(θ), q * cos(θ)                       # Eq. (1)
    u = v = ϕ = 0.0
    for n in 1:2:nmodes
        λ, c = n * π / L, 4 / (n * π)
        A = [E*Iy*λ^4 + kx    E*Ixy*λ^4      kx*ay
             E*Ixy*λ^4        E*Ix*λ^4       0.0
             kx*ay            0.0            E*Cw*λ^4 + G*J*λ^2 + kx*ay^2 + kϕ + qx*ax - qy*ay]
        x = A \ (c .* [qx, qy, qx*ay + qy*ax])
        s = sin(λ * z)
        u += x[1] * s; v += x[2] * s; ϕ += x[3] * s
    end
    return (u = u, v = v, ϕ = ϕ)
end

# The paper-reference testsets below check Eqs. (2)-(4) exactly, i.e. the geometric
# stiffness with the load-height term only, so these helpers turn the moment
# coupling off; the "moment coupling" testsets at the end turn it on.
model1(; moment_coupling = false, kwargs...) =
    BeamModel(; L, nel = 40, section = sec1, material = mat, moment_coupling, kwargs...)
model2(; moment_coupling = false, kwargs...) =
    BeamModel(; L, nel = 40, section = sec2, material = mat, moment_coupling, kwargs...)

peak_abs_twist(sol; npts = 401) = maximum(abs, twist.(Ref(sol), range(0, sol.model.L, npts)))

# ==================================================================

@testset "PlautBeamElement" begin

@testset "stiffness matrix properties" begin
    m = model1(q = 1.0, restraint = Restraint(kx = 0.1, kϕ = 1000.0))
    Ke = elastic_stiffness(m)
    Kg = geometric_stiffness(m)
    @test Ke ≈ Ke'                       # symmetric
    @test Kg ≈ Kg'
    free = setdiff(1:size(Ke, 1), constrained_dofs(m))
    @test isposdef(Symmetric(Ke[free, free]))
    # gravity load above the shear center softens the ϕϕ block (qx ax - qy ay < 0);
    # Kg is negative semi-definite (zero rows/columns for the u and v DOFs)
    λmin = eigmin(Symmetric(Kg[free, free]))
    @test λmin < 0
    @test eigmax(Symmetric(Kg[free, free])) <= 1e-9 * abs(λmin)
    # uplift stiffens it (positive semi-definite up to round-off)
    Kg_up = geometric_stiffness(model1(q = -1.0))
    @test eigmin(Symmetric(Kg_up[free, free])) >= -1e-9 * eigmax(Symmetric(Kg_up[free, free]))
    # geometric stiffness vanishes without load
    @test iszero(geometric_stiffness(model1(q = 0.0)))
end

@testset "vertical bending benchmarks (Eq. 3, C-section: v uncoupled)" begin
    q = 1.0
    EIx = mat.E * sec1.Ix
    # pinned-pinned: v_max = 5qL⁴/(384 EIx), exact at nodes for Hermite elements
    sol = solve_beam(model1(q = q))
    @test vertical_deflection(sol, L / 2) ≈ 5q * L^4 / (384EIx) rtol = 1e-9
    # fixed-fixed: v_max = qL⁴/(384 EIx)
    sol = solve_beam(model1(q = q, bc_left = :fixed, bc_right = :fixed))
    @test vertical_deflection(sol, L / 2) ≈ q * L^4 / (384EIx) rtol = 1e-9
    # pinned-fixed: v(L/2) = qL⁴/(192 EIx)
    sol = solve_beam(model1(q = q, bc_right = :fixed))
    @test vertical_deflection(sol, L / 2) ≈ q * L^4 / (192EIx) rtol = 1e-9
    # v is independent of the twist/lateral problem for the C-section (Ixy = 0)
    solr = solve_beam(model1(q = q, restraint = Restraint(kx = 0.1, kϕ = 1000.0)))
    @test vertical_deflection(solr, L / 2) ≈ 5q * L^4 / (384EIx) rtol = 1e-9
end

@testset "Example 1 (C-section) vs exact series, pinned-pinned" begin
    cases = [
        (restraint = Restraint(),                          q =  0.05, θ = 0.0),  # Figs. 3/8 lowest curves
        (restraint = Restraint(kϕ = 500.0),                q =  1.0,  θ = 0.0),  # Fig. 3
        (restraint = Restraint(kϕ = 500.0),                q = -1.0,  θ = 0.0),  # Fig. 4
        (restraint = Restraint(kx = 0.1, kϕ = 1000.0),     q =  1.0,  θ = 0.0),  # Fig. 7
        (restraint = Restraint(kx = 0.1, kϕ = 1000.0),     q = -1.0,  θ = 0.0),  # Fig. 7
        (restraint = Restraint(kx = 0.01, kϕ = 0.0),       q =  0.3,  θ = 0.0),  # Fig. 8
        (restraint = Restraint(kx = 0.1, kϕ = 1000.0),     q =  1.0,  θ = 0.083),   # Fig. 18, 1:12 upslope
        (restraint = Restraint(kx = 0.1, kϕ = 1000.0),     q =  1.0,  θ = -0.083),  # downslope
        (restraint = Restraint(kx = 0.1, kϕ = 1000.0),     q =  2.0,  θ = 0.46),    # Fig. 17, 6:12 pitch
    ]
    for c in cases
        sol = solve_beam(model1(q = c.q, θ = c.θ, restraint = c.restraint))
        for z in (0.25L, 0.42L, 0.5L)
            ref = series_solution(z; L, section = sec1, material = mat,
                                  restraint = c.restraint, q = c.q, θ = c.θ)
            @test twist(sol, z) ≈ ref.ϕ rtol = 1e-4
            @test lateral_deflection(sol, z) ≈ ref.u rtol = 1e-4 atol = 1e-10
            @test vertical_deflection(sol, z) ≈ ref.v rtol = 1e-4
        end
    end
end

@testset "Example 1 magnitudes match Fig. 7 (kx=0.1, kϕ=1000, q=±1)" begin
    r = Restraint(kx = 0.1, kϕ = 1000.0)
    solg = solve_beam(model1(q = 1.0, restraint = r))
    solu = solve_beam(model1(q = -1.0, restraint = r))
    # Fig. 7: gravity twist peaks ≈ +0.075 rad at midspan, uplift ≈ -0.055 rad;
    # gravity twist magnitude exceeds uplift
    @test 0.06 < twist(solg, L / 2) < 0.09
    @test -0.075 < twist(solu, L / 2) < -0.05
    @test twist(solg, L / 2) > abs(twist(solu, L / 2))
    # Section 6: C-section lateral deflection is negative for gravity, positive for uplift
    @test lateral_deflection(solg, L / 2) < 0
    @test lateral_deflection(solu, L / 2) > 0
end

@testset "Example 2 (Z-section) vs exact series, pinned-pinned" begin
    cases = [
        (restraint = Restraint(kϕ = 500.0),                q =  1.0,  θ = 0.0),   # Fig. 10
        (restraint = Restraint(kϕ = 500.0),                q = -1.0,  θ = 0.0),   # Fig. 11
        (restraint = Restraint(kx = 0.1, kϕ = 1000.0),     q =  3.0,  θ = 0.0),   # Fig. 12
        (restraint = Restraint(kx = 0.1, kϕ = 1000.0),     q = -3.0,  θ = 0.0),   # Fig. 13
        (restraint = Restraint(kx = 0.1, kϕ = 1500.0),     q =  1.0,  θ = 0.0),   # Fig. 14
        (restraint = Restraint(kx = 0.1, kϕ = 1000.0),     q =  5.0,  θ = 0.3),   # Fig. 19
    ]
    for c in cases
        sol = solve_beam(model2(q = c.q, θ = c.θ, restraint = c.restraint))
        for z in (0.2L, 0.42L, 0.5L)
            ref = series_solution(z; L, section = sec2, material = mat,
                                  restraint = c.restraint, q = c.q, θ = c.θ)
            @test twist(sol, z) ≈ ref.ϕ rtol = 1e-4 atol = 1e-8
            @test lateral_deflection(sol, z) ≈ ref.u rtol = 1e-4
            @test vertical_deflection(sol, z) ≈ ref.v rtol = 1e-4
        end
    end
end

@testset "Example 2 wavy twist shape, Fig. 14 (kx=0.1, kϕ=1500, q=1)" begin
    sol = solve_beam(model2(q = 1.0, restraint = Restraint(kx = 0.1, kϕ = 1500.0)))
    # pinned-pinned curve in Fig. 14: outer portions twist clockwise (ϕ > 0),
    # central portion counter-clockwise (ϕ < 0)
    @test twist(sol, 0.2L) > 0
    @test twist(sol, L / 2) < 0
    # Section 6: Z-section lateral deflection sign is opposite the C-section
    # (positive for gravity)
    @test lateral_deflection(sol, L / 2) > 0
end

@testset "boundary condition effects, Fig. 3 (kϕ=500, q=1)" begin
    r = Restraint(kϕ = 500.0)
    ϕpp = peak_abs_twist(solve_beam(model1(q = 1.0, restraint = r)))
    ϕpf = peak_abs_twist(solve_beam(model1(q = 1.0, restraint = r, bc_right = :fixed)))
    ϕff = peak_abs_twist(solve_beam(model1(q = 1.0, restraint = r,
                                           bc_left = :fixed, bc_right = :fixed)))
    # "the twist is smallest for a fixed-fixed beam and largest for a pinned-pinned beam"
    @test ϕpp > ϕpf > ϕff > 0
    # pinned-fixed peak "at or near z = 0.42 L" (Section 3)
    solpf = solve_beam(model1(q = 1.0, restraint = r, bc_right = :fixed))
    zs = range(0, L, 763)
    zpeak = zs[argmax(twist.(Ref(solpf), zs))]
    @test 0.35L < zpeak < 0.50L
    # gravity twists positive, uplift negative (Figs. 3-4)
    @test twist(solve_beam(model1(q = 1.0, restraint = r)), L / 2) > 0
    @test twist(solve_beam(model1(q = -1.0, restraint = r)), L / 2) < 0
end

@testset "two-span beam ≡ pinned-fixed per span (Section 1)" begin
    # A symmetric two-span beam with a central support behaves like two
    # pinned-fixed beams with the fixed end at the central support.
    r = Restraint(kx = 0.1, kϕ = 1000.0)
    for coupling in (false, true)
    two = BeamModel(L = 2L, nel = 80, section = sec1, material = mat,
                    restraint = r, q = 1.0, interior_supports = [L], moment_coupling = coupling)
    sol2 = solve_beam(two)
    solpf = solve_beam(model1(q = 1.0, restraint = r, bc_right = :fixed, moment_coupling = coupling))
    for z in (0.2L, 0.42L, 0.6L, 0.8L)
        @test twist(sol2, z) ≈ twist(solpf, z) rtol = 1e-8 atol = 1e-12
        @test lateral_deflection(sol2, z) ≈ lateral_deflection(solpf, z) rtol = 1e-8
        @test vertical_deflection(sol2, z) ≈ vertical_deflection(solpf, z) rtol = 1e-8
        # symmetry about the central support
        @test twist(sol2, z) ≈ twist(sol2, 2L - z) rtol = 1e-8 atol = 1e-12
    end
    # warping/twist rotation continuity at the central support: ϕ'(L) = 0 by symmetry
    @test twist(sol2, L - 1.0) ≈ twist(sol2, L + 1.0) rtol = 1e-6 atol = 1e-12
    end
end

@testset "geometric stiffness: load-deformation nonlinearity (Figs. 3-4)" begin
    r = Restraint(kϕ = 500.0)
    ϕ1 = twist(solve_beam(model1(q = 0.5, restraint = r)), L / 2)
    ϕ2 = twist(solve_beam(model1(q = 1.0, restraint = r)), L / 2)
    # gravity: softening — twist grows faster than the load (Fig. 3 slopes decrease)
    @test ϕ2 > 2ϕ1
    # uplift: stiffening — twist grows slower than the load (Fig. 4 slopes steepen)
    ϕ1u = twist(solve_beam(model1(q = -0.5, restraint = r)), L / 2)
    ϕ2u = twist(solve_beam(model1(q = -1.0, restraint = r)), L / 2)
    @test abs(ϕ2u) < 2abs(ϕ1u)
    # without Kg the response is exactly linear in q
    ψ1 = twist(solve_beam(model1(q = 0.5, restraint = r); geometric = false), L / 2)
    ψ2 = twist(solve_beam(model1(q = 1.0, restraint = r); geometric = false), L / 2)
    @test ψ2 ≈ 2ψ1 rtol = 1e-9
end

@testset "critical load from geometric stiffness (unbraced C-section)" begin
    # For kx = kϕ = 0 and pinned ends, ϕ decouples and the tangent stiffness of
    # mode 1 vanishes at q_cr = (ECw λ₁⁴ + GJ λ₁²)/ay
    λ1 = π / L
    qcr = (mat.E * sec1.Cw * λ1^4 + mat.G * sec1.J * λ1^2) / sec1.ay
    for (q, sgn) in ((0.98qcr, +1), (1.02qcr, -1))
        m = model1(q = q)
        K = elastic_stiffness(m) + geometric_stiffness(m)
        free = setdiff(1:size(K, 1), constrained_dofs(m))
        @test sign(eigmin(Symmetric(K[free, free]))) == sgn
    end
    # direct linear solve satisfies equilibrium on either side of q_cr
    @test solve_beam(model1(q = 0.98qcr)).residual_norm < 1e-8
    @test solve_beam(model1(q = 1.02qcr)).residual_norm < 1e-8
    # gravity softening: twist magnifies as q -> q_cr
    @test twist(solve_beam(model1(q = 0.98qcr)), L / 2) / (0.98qcr) >
          10 * twist(solve_beam(model1(q = 0.5qcr)), L / 2) / (0.5qcr)
end

@testset "mesh convergence" begin
    r = Restraint(kx = 0.1, kϕ = 1000.0)
    ϕ = [twist(solve_beam(BeamModel(L = L, nel = n, section = sec2, material = mat,
                                    restraint = r, q = 3.0, bc_right = :fixed,
                                    moment_coupling = false)), 0.42L)
         for n in (8, 16, 32, 64)]
    err = abs.(ϕ[1:3] .- ϕ[4])
    @test err[2] < err[1] && err[3] < err[2]
    @test abs(ϕ[3] - ϕ[4]) / abs(ϕ[4]) < 1e-4
end

@testset "element end forces: bending statics (C-section, v uncoupled)" begin
    q = 1.0
    # pinned-pinned: Vy(0) = +qL/2, Vy(L) = -qL/2, Mx(0) = 0, Mx(L/2) = -qL²/8
    f = element_end_forces(solve_beam(model1(q = q)))
    @test f[1].end1.Vy ≈ q * L / 2 rtol = 1e-9
    @test f[end].end2.Vy ≈ -q * L / 2 rtol = 1e-9
    @test abs(f[1].end1.Mx) < 1e-6 * q * L^2 / 8
    @test f[20].end2.Mx ≈ -q * L^2 / 8 rtol = 1e-9          # nel = 40 → midspan boundary
    @test f[20].end2.z ≈ L / 2
    # fixed-fixed: Mx(0) = +qL²/12 (hogging in this convention), Mx(L/2) = -qL²/24
    f = element_end_forces(solve_beam(model1(q = q, bc_left = :fixed, bc_right = :fixed)))
    @test f[1].end1.Mx ≈ q * L^2 / 12 rtol = 1e-9
    @test f[20].end2.Mx ≈ -q * L^2 / 24 rtol = 1e-9
    @test f[1].end1.Vy ≈ q * L / 2 rtol = 1e-9
    # pinned-fixed: reactions 3qL/8 and 5qL/8, fixed-end moment qL²/8
    f = element_end_forces(solve_beam(model1(q = q, bc_right = :fixed)))
    @test f[1].end1.Vy ≈ 3q * L / 8 rtol = 1e-9
    @test f[end].end2.Vy ≈ -5q * L / 8 rtol = 1e-9
    @test f[end].end2.Mx ≈ q * L^2 / 8 rtol = 1e-9
end

@testset "element end forces: continuity across element boundaries" begin
    for m in (model1(q = 1.0, restraint = Restraint(kx = 0.1, kϕ = 1000.0), θ = 0.083),
              model2(q = -1.0, restraint = Restraint(kx = 0.1, kϕ = 500.0), bc_right = :fixed))
        f = element_end_forces(solve_beam(m))
        for e in 1:(m.nel - 1), a in (:Vx, :My, :Vy, :Mx, :T, :B)
            x2, x1 = getfield(f[e].end2, a), getfield(f[e + 1].end1, a)
            @test isapprox(x2, x1; rtol = 1e-6,
                           atol = 1e-8 * maximum(abs, [getfield(f[k].end2, a) for k in 1:m.nel]) + 1e-12)
        end
    end
end

@testset "element end forces: torsion and moments vs exact series" begin
    # Series section forces for pinned-pinned beams: with (u, v, ϕ) = Σ (uₙ, vₙ, ϕₙ) sin(λₙz),
    #   My = -E Σ (Iy uₙ + Ixy vₙ) λₙ² sin(λₙz)      Mx = -E Σ (Ix vₙ + Ixy uₙ) λₙ² sin(λₙz)
    #   T  =  Σ ϕₙ λₙ (GJ + ECw λₙ²) cos(λₙz)        B  = -ECw Σ ϕₙ λₙ² sin(λₙz)
    function series_forces(z; section, restraint, q, nmodes = 19999)
        (; E, G) = mat
        (; Ix, Iy, Ixy, J, Cw, ax, ay) = section
        (; kx, kϕ) = restraint
        My = Mx = T = B = 0.0
        for n in 1:2:nmodes
            λ, c = n * π / L, 4 / (n * π)
            A = [E*Iy*λ^4 + kx   E*Ixy*λ^4   kx*ay
                 E*Ixy*λ^4       E*Ix*λ^4    0.0
                 kx*ay           0.0         E*Cw*λ^4 + G*J*λ^2 + kx*ay^2 + kϕ - q*ay]
            un, vn, ϕn = A \ (c .* [0.0, q, q * ax])
            s, co = sin(λ * z), cos(λ * z)
            My -= E * (Iy * un + Ixy * vn) * λ^2 * s
            Mx -= E * (Ix * vn + Ixy * un) * λ^2 * s
            T += ϕn * λ * (G * J + E * Cw * λ^2) * co
            B -= E * Cw * ϕn * λ^2 * s
        end
        return (My = My, Mx = Mx, T = T, B = B)
    end
    cases = [(section = sec1, restraint = Restraint(kϕ = 500.0),            q = 1.0),
             (section = sec1, restraint = Restraint(kx = 0.1, kϕ = 1000.0), q = -1.0),
             (section = sec2, restraint = Restraint(kx = 0.1, kϕ = 1000.0), q = 3.0)]
    for c in cases
        m = BeamModel(L = L, nel = 40, section = c.section, material = mat,
                      restraint = c.restraint, q = c.q, moment_coupling = false)
        f = element_end_forces(solve_beam(m))
        for e in (11, 21)                                     # z = 0.25L, 0.5L
            z = f[e].end1.z
            ref = series_forces(z; section = c.section, restraint = c.restraint, q = c.q)
            @test f[e].end1.My ≈ ref.My rtol = 1e-3 atol = 1e-8
            @test f[e].end1.Mx ≈ ref.Mx rtol = 1e-3
            @test f[e].end1.T ≈ ref.T rtol = 1e-3 atol = 1e-6 * abs(c.q) * c.section.ax * L
            @test f[e].end1.B ≈ ref.B rtol = 1e-3 atol = 1e-8
        end
    end
end

@testset "element end forces: bimoment boundary conditions and torque balance" begin
    r = Restraint(kx = 0.1, kϕ = 1000.0)
    # pinned ends are free to warp → bimoment vanishes there; fixed ends restrain warping
    f = element_end_forces(solve_beam(model1(q = 1.0, restraint = r, bc_right = :fixed)))
    Bmax = maximum(abs(fe.end2.B) for fe in f)
    @test abs(f[1].end1.B) < 1e-8 * Bmax
    @test abs(f[end].end2.B) > 0.1Bmax
    # global torque balance, unbraced pinned-pinned C-section (kx = kϕ = 0):
    # T(0) - T(L) = ∫ q (ax + ay ϕ) dz, and T(L) = -T(0) by symmetry
    q = 0.05
    sol = solve_beam(model1(q = q))
    f = element_end_forces(sol)
    zs = range(0, L, 2001)
    tint = q * (sec1.ax * L + sec1.ay * sum(twist.(Ref(sol), zs)) * step(zs))
    @test f[1].end1.T - f[end].end2.T ≈ tint rtol = 1e-6
    @test f[end].end2.T ≈ -f[1].end1.T rtol = 1e-9
end

@testset "trapezoidal load: reduction to uniform and linearity" begin
    r = Restraint(kx = 0.1, kϕ = 1000.0)
    # equal end intensities assemble identically to a scalar uniform load
    @test solve_beam(model1(q = (1.0, 1.0), restraint = r)).d ≈
          solve_beam(model1(q = 1.0, restraint = r)).d rtol = 1e-12
    # without Kg the problem is linear: trapezoid = uniform + triangle
    dsum = solve_beam(model1(q = 1.0, restraint = r); geometric = false).d .+
           solve_beam(model1(q = (0.0, 1.0), restraint = r); geometric = false).d
    @test solve_beam(model1(q = (1.0, 2.0), restraint = r); geometric = false).d ≈
          dsum rtol = 1e-9
end

@testset "trapezoidal load: triangular bending statics (v uncoupled)" begin
    # pinned-pinned, load 0 → w: reactions wL/6 and wL/3, M(L/2) = wL²/16
    w = 1.0
    f = element_end_forces(solve_beam(model1(q = (0.0, w))))
    @test f[1].end1.Vy ≈ w * L / 6 rtol = 1e-9
    @test f[end].end2.Vy ≈ -w * L / 3 rtol = 1e-9
    @test f[20].end2.Mx ≈ -w * L^2 / 16 rtol = 1e-9
    # max moment of a triangular load at z = L/√3, M = wL²/(9√3)
    zmax = L / sqrt(3)
    Mmax = -w * L^2 / (9 * sqrt(3))
    m = BeamModel(L = L, nel = 100, section = sec1, material = mat, q = (0.0, w))
    zs = [fe.end2.z for fe in element_end_forces(solve_beam(m))]
    Ms = [fe.end2.Mx for fe in element_end_forces(solve_beam(m))]
    e = argmin(abs.(zs .- zmax))
    @test Ms[e] ≈ Mmax rtol = 1e-3
end

@testset "trapezoidal load vs spectral Galerkin reference" begin
    # Independent reference: sine-basis Galerkin solution of Eqs. (2)-(4) for a
    # pinned-pinned beam. A linearly varying load q(z) makes the geometric
    # coefficient w(z) = qx(z)ax - qy(z)ay = α + βz non-constant, which couples
    # the modes: with ϕ = Σ ϕₙ sin(λₙz),
    #   (2/L)∫ z sin(λₘz) sin(λₙz) dz = L/2 (m = n),
    #                                 = (2L/π²)[1/(m+n)² - 1/(m-n)²] (m±n odd), 0 else.
    function spectral_solution(z; section, restraint, q1, q2, θ = 0.0, N = 200)
        (; E, G) = mat
        (; Ix, Iy, Ixy, J, Cw, ax, ay) = section
        (; kx, kϕ) = restraint
        s, c = sin(θ), cos(θ)
        # modal coefficient of a field varying linearly from ra (z=0) to rb (z=L)
        lincoeff(ra, rb, m) = 2 * (1 - (-1)^m) * ra / (m * π) +
                              2 * (-1)^(m + 1) * (rb - ra) / (m * π)
        A = zeros(3N, 3N)
        b = zeros(3N)
        α, β = -(s * ax + c * ay) * q1, -(s * ax + c * ay) * (q2 - q1) / L
        for m in 1:N
            λ = m * π / L
            iu, iv, iϕ = 3m - 2, 3m - 1, 3m
            A[iu, iu] = E * Iy * λ^4 + kx
            A[iu, iv] = E * Ixy * λ^4
            A[iu, iϕ] = kx * ay
            A[iv, iu] = E * Ixy * λ^4
            A[iv, iv] = E * Ix * λ^4
            A[iϕ, iu] = kx * ay
            A[iϕ, iϕ] = E * Cw * λ^4 + G * J * λ^2 + kx * ay^2 + kϕ + α
            for n in 1:N
                Z = m == n ? L / 2 :
                    isodd(m - n) ? (2L / π^2) * (1 / (m + n)^2 - 1 / (m - n)^2) : 0.0
                A[iϕ, 3n] += β * Z
            end
            b[iu] = lincoeff(-q1 * s, -q2 * s, m)
            b[iv] = lincoeff(q1 * c, q2 * c, m)
            b[iϕ] = lincoeff(q1 * (c * ax - s * ay), q2 * (c * ax - s * ay), m)
        end
        x = A \ b
        u = v = ϕ = 0.0
        for m in 1:N
            sλ = sin(m * π * z / L)
            u += x[3m - 2] * sλ; v += x[3m - 1] * sλ; ϕ += x[3m] * sλ
        end
        return (u = u, v = v, ϕ = ϕ)
    end
    cases = [
        (model = model1, section = sec1, restraint = Restraint(kx = 0.1, kϕ = 1000.0),
         q = (0.5, 1.5), θ = 0.0),
        (model = model1, section = sec1, restraint = Restraint(kϕ = 500.0),
         q = (-0.5, -1.5), θ = 0.0),                                    # uplift trapezoid
        (model = model1, section = sec1, restraint = Restraint(kx = 0.1, kϕ = 1000.0),
         q = (1.0, 2.0), θ = 0.083),                                    # with roof slope
        (model = model2, section = sec2, restraint = Restraint(kx = 0.1, kϕ = 1000.0),
         q = (0.0, 3.0), θ = 0.0),                                      # triangular, Z-section
    ]
    for cs in cases
        sol = solve_beam(cs.model(q = cs.q, θ = cs.θ, restraint = cs.restraint))
        for z in (0.25L, 0.5L, 0.75L)
            ref = spectral_solution(z; section = cs.section, restraint = cs.restraint,
                                    q1 = cs.q[1], q2 = cs.q[2], θ = cs.θ)
            @test twist(sol, z) ≈ ref.ϕ rtol = 1e-4 atol = 1e-8
            @test lateral_deflection(sol, z) ≈ ref.u rtol = 1e-4 atol = 1e-10
            @test vertical_deflection(sol, z) ≈ ref.v rtol = 1e-4
        end
    end
    # asymmetry: heavier end draws the twist peak past midspan
    sol = solve_beam(model1(q = (0.0, 1.0), restraint = Restraint(kϕ = 500.0)))
    zs = range(0, L, 763)
    @test zs[argmax(twist.(Ref(sol), zs))] > 0.5L
end

@testset "trapezoidal load: torque balance (unbraced C-section)" begin
    # kx = kϕ = 0: T(0) - T(L) = ∫ q(z)(ax + ay ϕ(z)) dz, twist-dependent torque included
    q1, q2 = 0.02, 0.08
    sol = solve_beam(model1(q = (q1, q2)))
    f = element_end_forces(sol)
    zs = range(0, L, 2001)
    qz = q1 .+ (q2 - q1) .* zs ./ L
    tz = qz .* (sec1.ax .+ sec1.ay .* twist.(Ref(sol), zs))
    tint = (sum(tz) - (tz[1] + tz[end]) / 2) * step(zs)     # trapezoidal rule
    @test f[1].end1.T - f[end].end2.T ≈ tint rtol = 1e-5
end

@testset "per-span loads: reduce to whole-beam q when spans agree" begin
    # Two equal spans, same uniform load on each -- must exactly match the
    # existing whole-beam scalar-q API (Ixy=0 C-section decouples v from u/ϕ).
    two_scalar = BeamModel(L = 2L, nel = 80, section = sec1, material = mat,
                           q = 1.0, interior_supports = [L])
    two_perspan = BeamModel(L = 2L, nel = 80, section = sec1, material = mat,
                            q = [1.0, 1.0], interior_supports = [L])
    @test solve_beam(two_perspan).d ≈ solve_beam(two_scalar).d rtol = 1e-12
    # same check with a trapezoidal whole-beam load split into two per-span
    # trapezoids that agree at the shared support
    two_scalar_trap = BeamModel(L = 2L, nel = 80, section = sec1, material = mat,
                                q = (0.0, 2.0), interior_supports = [L])
    two_perspan_trap = BeamModel(L = 2L, nel = 80, section = sec1, material = mat,
                                 q = [(0.0, 1.0), (1.0, 2.0)], interior_supports = [L])
    @test solve_beam(two_perspan_trap).d ≈ solve_beam(two_scalar_trap).d rtol = 1e-9
end

@testset "per-span loads: two equal spans, different uniform loads per span" begin
    # Classical three-moment theorem, two equal spans L, pinned outer ends,
    # UDL w1 on span 1 and w2 on span 2: M_B = (w1+w2) L² / 16 at the shared
    # support (reduces to the standard wL²/8 continuous-beam result at w1=w2=w).
    w1, w2 = 1.0, 3.0
    m = BeamModel(L = 2L, nel = 80, section = sec1, material = mat,
                 q = [w1, w2], interior_supports = [L])
    sol = solve_beam(m)
    f = element_end_forces(sol)
    e_at_support = m.nel ÷ 2
    @test f[e_at_support].end2.z ≈ L
    MB_expected = (w1 + w2) * L^2 / 16
    @test f[e_at_support].end2.Mx ≈ MB_expected rtol = 1e-6
    @test f[e_at_support].end2.Mx ≈ f[e_at_support + 1].end1.Mx rtol = 1e-9   # moment continuity
    # global vertical equilibrium: end reactions + interior reaction (the shear
    # jump across the support) sum to the total applied load
    R_left, R_right = f[1].end1.Vy, -f[end].end2.Vy
    R_interior = f[e_at_support + 1].end1.Vy - f[e_at_support].end2.Vy
    @test R_left + R_interior + R_right ≈ (w1 + w2) * L rtol = 1e-6
end

@testset "per-span loads: trapezoidal exterior spans, uniform interior spans" begin
    # 5-span line mirroring the RoofHugger snow-drift use case: exterior spans
    # carry a trapezoidal drift load, interior spans a uniform load, with an
    # intentional jump discontinuity at each interior support.
    spans = fill(L, 5)
    zs = [0.0; cumsum(spans)]
    q_per_span = [(0.5, 1.0), 1.0, 1.5, 1.0, (1.0, 0.5)]   # jumps: 1.0->1.0 (none), 1.0->1.5, 1.5->1.0, 1.0->1.0
    m = BeamModel(L = zs[end], nel = 5 * 80, section = sec1, material = mat,
                 q = q_per_span, interior_supports = zs[2:end-1])
    sol = solve_beam(m)
    f = element_end_forces(sol)
    # global vertical equilibrium: sum of all reactions (both ends + the shear
    # jump at each interior support) equals the total applied load
    total_load = sum((q_per_span[i] isa Real ? q_per_span[i] : sum(q_per_span[i]) / 2) * spans[i]
                      for i in 1:5)
    per_span_elements = m.nel ÷ 5
    R_left, R_right = f[1].end1.Vy, -f[end].end2.Vy
    R_interior = [f[k * per_span_elements + 1].end1.Vy - f[k * per_span_elements].end2.Vy for k in 1:4]
    @test R_left + sum(R_interior) + R_right ≈ total_load rtol = 1e-6
    # moment continuity at every interior support despite the load jumps
    for k in 1:4
        e = k * per_span_elements
        @test f[e].end2.z ≈ zs[k + 1]
        @test f[e].end2.Mx ≈ f[e + 1].end1.Mx rtol = 1e-6 atol = 1e-8
    end
end

@testset "per-span loads: input validation" begin
    @test_throws ArgumentError model1(q = [1.0, 2.0])                 # wrong span count (1 span, 2 entries)
    @test_throws ArgumentError BeamModel(L = 2L, nel = 80, section = sec1, material = mat,
                                        q = [(1.0, 2.0, 3.0), 1.0], interior_supports = [L])
end

@testset "input validation" begin
    @test_throws ArgumentError model1(q = 1.0, bc_left = :clamped)
    @test_throws ArgumentError model1(q = (1.0, 2.0, 3.0))
    @test_throws ArgumentError BeamModel(L = 2L, nel = 80, section = sec1,
                                         material = mat, q = 1.0,
                                         interior_supports = [L + 13.0])
    sol = solve_beam(model1(q = 1.0))
    @test_throws DomainError twist(sol, -1.0)
    @test sol.residual_norm < 1e-8
end

@testset "sparse assembly ≡ dense assembly" begin
    r = Restraint(kx = 0.1, kϕ = 1000.0)
    models = [
        model1(q = 1.0, restraint = r),
        model1(q = -1.0, restraint = r, θ = 0.1),
        model2(q = 1.0, restraint = Restraint(kx = 0.1, kϕ = 1500.0), bc_right = :fixed),
        model2(q = (0.5, 1.5), restraint = r),
        BeamModel(L = 2L, nel = 80, section = sec2, material = mat, restraint = r,
                  q = [1.0, (1.0, 2.0)], interior_supports = [L]),
        model1(q = -1.0, restraint = r, θ = 0.1, moment_coupling = true),
        model2(q = 1.0, restraint = Restraint(kx = 0.1, kϕ = 1500.0), bc_right = :fixed,
               moment_coupling = true),
    ]
    for m in models
        @test sparse_stiffness(m) ≈ elastic_stiffness(m) + geometric_stiffness(m)
        @test sparse_stiffness(m; geometric = false) ≈ elastic_stiffness(m)
        F = load_vector(m)
        free = setdiff(1:size(F, 1), constrained_dofs(m))
        for geometric in (true, false)
            K = geometric ? elastic_stiffness(m) + geometric_stiffness(m) : elastic_stiffness(m)
            ddense = K[free, free] \ F[free]               # dense reference solve
            s = solve_beam(m; geometric)
            @test s isa BeamSolution
            @test s.geometric == geometric
            @test s.residual_norm < 1e-8
            @test s.d[free] ≈ ddense rtol = 1e-9
            @test solve_beam_linear(m; geometric).d == s.d   # alias
        end
    end
    # a mesh whose dense stiffness would need ~2.5 GB solves in sparse storage
    big = BeamModel(L = 10L, nel = 3000, section = sec2, material = mat, restraint = r, q = 1.0,
                    interior_supports = [5L])
    @test solve_beam(big).residual_norm < 1e-8
end

# ==================================================================
# Moment coupling: twist ↔ bending terms -(Mx ϕ)'', (My ϕ)'', -Mx u'', My v''
# ==================================================================

@testset "moment coupling: geometric stiffness properties" begin
    r = Restraint(kx = 0.1, kϕ = 1000.0)
    @test BeamModel(L = L, nel = 4, section = sec2, material = mat, q = 1.0).moment_coupling
    for m in (model2(q = 1.0, moment_coupling = true),
              model1(q = 1.0, restraint = r, θ = 0.2, moment_coupling = true),
              BeamModel(L = 2L, nel = 80, section = sec2, material = mat, restraint = r,
                        q = [1.0, (1.0, 2.0)], interior_supports = [L]))
        Kg = geometric_stiffness(m)
        @test Kg ≈ Kg' atol = 1e-12 * maximum(abs, Kg)
        # couples u and v to ϕ (rows/cols 1,3 vs 5 of each node), unlike load height alone
        @test maximum(abs, Kg[1:6:end, 5:6:end]) > 0
        @test iszero(geometric_stiffness(BeamModel(L = m.L, nel = m.nel, section = m.section,
                                                   material = m.material, restraint = m.restraint,
                                                   q = 1.0, moment_coupling = false))[1:6:end, 5:6:end])
    end
    # Kg is proportional to the load (moments are first-order), so q_cr is one eigenvalue
    m1 = model2(q = 1.0, restraint = r, moment_coupling = true)
    m2 = model2(q = 2.0, restraint = r, moment_coupling = true)
    @test geometric_stiffness(m2) ≈ 2 * geometric_stiffness(m1) rtol = 1e-10
    @test critical_load(m2) ≈ critical_load(m1) / 2 rtol = 1e-8
    @test iszero(geometric_stiffness(model2(q = 0.0, moment_coupling = true)))
end

@testset "moment coupling: pre-buckling moments" begin
    # single span: Mx = -q z (L - z)/2 (sagging negative), My = 0 without lateral load
    m = model2(q = 1.0, moment_coupling = true)
    M = prebuckling_moments(m)
    Le = L / m.nel
    for e in (1, 10, 20)
        z1, z2 = (e - 1) * Le, e * Le
        @test M[e].Mx1 ≈ -z1 * (L - z1) / 2 atol = 1e-6 * L^2
        @test M[e].Mx2 ≈ -z2 * (L - z2) / 2 atol = 1e-6 * L^2
        @test abs(M[e].My1) < 1e-8 * L^2
    end
    @test prebuckling_moments(model2(q = 1.0)) === nothing        # helper default: off
    # two equal spans: hogging moment qL²/8 over the central support (three-moment theorem)
    two = BeamModel(L = 2L, nel = 80, section = sec1, material = mat, q = 1.0,
                    interior_supports = [L])
    M2 = prebuckling_moments(two)
    @test M2[40].Mx2 ≈ L^2 / 8 rtol = 1e-8
    @test M2[41].Mx1 ≈ L^2 / 8 rtol = 1e-8
end

@testset "moment coupling: classical LTB of a doubly symmetric I-beam" begin
    # uniform load at the shear center, fork supports: q_cr L²/8 = C1 Mcr, C1 ≈ 1.13
    E, G = 200e3, 200e3 / 2.6
    Iy, J, Cw = 2.0e6, 1.0e5, 1.0e11
    Mcr = π / L * sqrt(E * Iy * G * J * (1 + π^2 * E * Cw / (G * J * L^2)))
    mk(ay) = BeamModel(L = L, nel = 60, section = Section(Ix = 50e6, Iy = Iy, J = J, Cw = Cw,
                       ax = 0.0, ay = ay), material = Material(E = E, G = G), q = 1.0)
    C1 = critical_load(mk(0.0)) * L^2 / 8 / Mcr
    @test 1.12 < C1 < 1.14
    # load above the shear center lowers q_cr, below raises it
    @test critical_load(mk(150.0)) < critical_load(mk(0.0)) < critical_load(mk(-150.0))
    # without the coupling there is no buckling for a load at the shear center
    @test critical_load(BeamModel(L = L, nel = 60, section = Section(Ix = 50e6, Iy = Iy, J = J,
                        Cw = Cw, ax = 0.0, ay = 0.0), material = Material(E = E, G = G), q = 1.0,
                        moment_coupling = false)) == Inf
end

@testset "moment coupling: critical load without coupling matches closed form" begin
    λ1 = π / L
    qcr = (mat.E * sec1.Cw * λ1^4 + mat.G * sec1.J * λ1^2) / sec1.ay
    @test critical_load(model1(q = 1.0)) ≈ qcr rtol = 1e-5
end

@testset "moment coupling vs spectral Galerkin reference" begin
    # Independent sine-series Galerkin solution of the coupled equations for a
    # pinned-pinned beam under uniform gravity load (qx = 0). Mx(z) = -q z (L - z)/2 from
    # statics; My(z) = E (Iy u'' + Ixy v'') from the reference's own first-order solution
    # (nonzero with bracing, whose reactions bend the beam laterally). With
    # u, v, ϕ = Σ (aₘ, cₘ, bₘ) sₘ the coupling energy ∫ (-Mx ϕ u'' + My ϕ v'') dz is
    # Σ aₘ bₙ λₘ² ∫ Mx sₘ sₙ dz - Σ cₘ bₙ λₘ² ∫ My sₘ sₙ dz.
    function galerkin(section, restraint, q; N = 60)
        (; E, G) = mat
        (; Ix, Iy, Ixy, J, Cw, ax, ay) = section
        (; kx, kϕ) = restraint
        nz = 6000                                            # composite Simpson
        zs = range(0, L, nz + 1)
        wz = [(i == 0 || i == nz) ? 1.0 : (isodd(i) ? 4.0 : 2.0) for i in 0:nz] .* (step(zs) / 3)
        Mx = -q .* zs .* (L .- zs) ./ 2
        S = [sin(m * π * z / L) for z in zs, m in 1:N]
        K = zeros(3N, 3N); F = zeros(3N)
        iu(m) = 3m - 2; iv(m) = 3m - 1; iϕ(m) = 3m
        for m in 1:N
            λ = m * π / L
            K[iu(m), iu(m)] = (E * Iy * λ^4 + kx) * L / 2
            K[iu(m), iv(m)] = K[iv(m), iu(m)] = E * Ixy * λ^4 * L / 2
            K[iv(m), iv(m)] = E * Ix * λ^4 * L / 2
            K[iu(m), iϕ(m)] = K[iϕ(m), iu(m)] = kx * ay * L / 2
            K[iϕ(m), iϕ(m)] = (E * Cw * λ^4 + G * J * λ^2 + kx * ay^2 + kϕ) * L / 2
            c = isodd(m) ? 2L / (m * π) : 0.0                # ∫ sₘ dz
            F[iv(m)] = q * c
            F[iϕ(m)] = q * ax * c
        end
        x0 = K \ F                                           # first order
        My = -E .* (S * [(Iy * x0[iu(m)] + Ixy * x0[iv(m)]) * (m * π / L)^2 for m in 1:N])
        K[diagind(K)[3:3:end]] .-= q * ay * L / 2             # load height
        for m in 1:N, n in 1:N
            Cx = (m * π / L)^2 * sum(wz .* Mx .* S[:, m] .* S[:, n])
            Cy = -(m * π / L)^2 * sum(wz .* My .* S[:, m] .* S[:, n])
            K[iu(m), iϕ(n)] += Cx; K[iϕ(n), iu(m)] += Cx
            K[iv(m), iϕ(n)] += Cy; K[iϕ(n), iv(m)] += Cy
        end
        x = K \ F
        field(z, off) = sum(x[3m - off] * sin(m * π * z / L) for m in 1:N)
        return (u = z -> field(z, 2), v = z -> field(z, 1), ϕ = z -> field(z, 0))
    end
    # loads ≈ 0.4 q_cr (coupled q_cr ≈ 0.117, 3.53, 0.776 N/mm respectively)
    cases = [(section = sec2, restraint = Restraint(), q = 0.05),                    # unbraced Z
             (section = sec2, restraint = Restraint(kx = 0.1, kϕ = 1000.0), q = 1.5), # braced Z
             (section = sec1, restraint = Restraint(kϕ = 500.0), q = 0.3)]          # C-section
    for c in cases
        ref = galerkin(c.section, c.restraint, c.q)
        sol = solve_beam(BeamModel(L = L, nel = 60, section = c.section, material = mat,
                                   restraint = c.restraint, q = c.q))
        for z in (0.25L, 0.5L)
            @test twist(sol, z) ≈ ref.ϕ(z) rtol = 1e-3
            @test lateral_deflection(sol, z) ≈ ref.u(z) rtol = 1e-3 atol = 1e-9
            @test vertical_deflection(sol, z) ≈ ref.v(z) rtol = 1e-3
        end
    end
end

@testset "moment coupling: Z8x2.5 vs Abaqus shell model (sign check)" begin
    # Three-way validation (rev. 29 Oct 2026): Z8x2.5 16 ga, 300 in, pinned, unbraced,
    # gravity on the top flange; kip, in. Abaqus S4R shell: q_cr = 11.2 lb/ft.
    secZ = Section(Ix = 8.434, Iy = 1.466, Ixy = -2.563, J = 0.001050, Cw = 16.76,
                   ax = 1.2793, ay = 3.97)
    plf = 1 / 12000
    mk(q, c) = BeamModel(L = 300.0, nel = 60, section = secZ, material = Material(E = 29500.0),
                         q = q * plf, moment_coupling = c)
    @test critical_load(mk(1.0, false)) ≈ 21.9 rtol = 2e-3          # load height only
    qcr = critical_load(mk(1.0, true))
    @test 10.0 < qcr < 11.2                                          # within 6 % of Abaqus
    # The coupling sign does not change q_cr (it enters squared) but it does change
    # the response: with the right sign every midspan quantity moves toward Abaqus.
    # Abaqus at 2.4 lb/ft in the Plaut frame: u = 0.358, v = 0.217, ϕ = 0.059.
    on, off = solve_beam(mk(2.4, true)), solve_beam(mk(2.4, false))
    for (f, ref) in ((lateral_deflection, 0.358), (vertical_deflection, 0.217), (twist, 0.059))
        @test abs(f(on, 150.0) - ref) < abs(f(off, 150.0) - ref)
    end
end

@testset "moment coupling: element end forces" begin
    m = model2(q = 1.0, moment_coupling = true)                    # unbraced Z, pinned
    sol = solve_beam(m)
    f = element_end_forces(sol)                                      # fixed axes (default)
    fs = element_end_forces(sol; axes = :section)
    # statics about the fixed axes are unchanged by the coupling
    @test f[1].end1.Vy ≈ L / 2 rtol = 1e-9
    @test f[20].end2.Mx ≈ -L^2 / 8 rtol = 1e-6
    # no lateral load: the fixed-axis My vanishes ...
    Mmax = L^2 / 8
    @test maximum(abs(fe.end2.My) for fe in f) < 1e-3 * Mmax
    # ... while about the twisted section axes the strong-axis moment has a
    # weak-axis component My ≈ Mx ϕ
    @test fs[20].end2.My ≈ f[20].end2.Mx * twist(sol, L / 2) rtol = 2e-2
    # fixed-axis actions are continuous between elements
    # (My is ≈ 0 here, so its tolerance is scaled by Mx)
    scale = Dict(a => maximum(abs, [getfield(fe.end2, a == :My ? :Mx : a) for fe in f])
                 for a in (:My, :Mx, :T, :B))
    for e in 1:(m.nel - 1), a in (:My, :Mx, :T, :B)
        x2, x1 = getfield(f[e].end2, a), getfield(f[e + 1].end1, a)
        @test isapprox(x2, x1; rtol = 1e-6, atol = 1e-8 * scale[a])
    end
    @test_throws ArgumentError element_end_forces(sol; axes = :principal)
end

@testset "moment coupling: mesh convergence of q_cr" begin
    qcr = [critical_load(BeamModel(L = L, nel = n, section = sec2, material = mat, q = 1.0,
                                   restraint = Restraint(kx = 0.01)))
           for n in (8, 16, 32, 64)]
    err = abs.(qcr[1:3] .- qcr[4])
    @test err[2] < err[1] && err[3] < err[2]
    @test err[3] / qcr[4] < 1e-4
end

end
