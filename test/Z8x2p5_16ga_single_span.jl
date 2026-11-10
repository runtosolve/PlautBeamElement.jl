# Z8x2.5 16 ga purlin (0.75 in lips), single 25 ft span, 50 plf gravity load on the top flange centerline.
#
#   - ends pinned: u = v = ϕ = "fixed", u' = v' = ϕ' (warping) = "free"
#   - axial translation fixed at the midspan cross-section. This removes the axial
#     rigid-body mode only. The Plaut-Moen formulation has no axial (w) DOF and no
#     axial load, so this condition has no effect on u, v, ϕ and is not modeled.
#   - no continuous bracing (kx = kϕ = 0)
#
# Run from the package root:   julia --project=. test/Z8x2p5_16ga_single_span.jl
# (Plots must be available, e.g. in the global environment.)
#
# Units: kip, in.

using PlautBeamElement: Section, Material, Restraint, BeamModel, solve_beam,
                              elastic_stiffness, geometric_stiffness, constrained_dofs,
                              lateral_deflection, vertical_deflection, twist
using LinearAlgebra
using Plots

# ------------------------------------------------------------------
# Cross section: Z8x2.5 060 0.75 lip (RoofHugger_PlautBeamFE database/Purlins.csv)
#   t = 0.060 in, H = 8 in, B = 2.5 in (out-to-out), lips D = 0.75 in at 55° to the
#   flanges, flaring away from the web. Centerline model with sharp corners.
# ------------------------------------------------------------------

t = 0.060
H, B, D = 8.0, 2.5, 0.75
θlip = deg2rad(55.0)

# Centerline dimensions
h = H - t
b = B - t

# Node coordinates in the paper's axes (Fig. 1): x horizontal, y DOWNWARD, origin
# at the web mid-height. Top flange extends toward +x, bottom flange toward -x.
nodes = [(-b - D * cos(θlip),  h / 2 - D * sin(θlip)),   # bottom lip tip
         (-b,  h / 2),                                  # bottom flange tip
         ( 0.0,  h / 2),                                # bottom web-flange corner
         ( 0.0, -h / 2),                                # top web-flange corner
         ( b, -h / 2),                                  # top flange tip
         ( b + D * cos(θlip), -h / 2 + D * sin(θlip))]  # top lip tip

"""
Thin-walled open-section properties from a single chain of straight segments of
thickness `t`: centroidal Ix, Iy, Ixy, St. Venant J, shear center (xs, ys), and
warping constant Cw about the shear center.
"""
function thin_walled_properties(nodes, t)
    segs = [(nodes[i], nodes[i + 1]) for i in 1:length(nodes) - 1]
    len(s) = hypot(s[2][1] - s[1][1], s[2][2] - s[1][2])
    # ∫ f g dA over a segment with f, g linear between nodal values
    seg_int(L, fi, fj, gi, gj) = t * L / 6 * (2fi * gi + fi * gj + fj * gi + 2fj * gj)

    A = sum(t * len(s) for s in segs)
    xc = sum(t * len(s) * (s[1][1] + s[2][1]) / 2 for s in segs) / A
    yc = sum(t * len(s) * (s[1][2] + s[2][2]) / 2 for s in segs) / A
    X = [n[1] - xc for n in nodes]
    Y = [n[2] - yc for n in nodes]
    Ls = len.(segs)

    integrate(f, g) = sum(seg_int(Ls[k], f[k], f[k + 1], g[k], g[k + 1]) for k in eachindex(segs))
    Ix, Iy, Ixy = integrate(Y, Y), integrate(X, X), integrate(X, Y)
    J = sum(Ls) * t^3 / 3

    # sectorial coordinate about the centroid, then shift the pole to the shear
    # center so that ∫ω x dA = ∫ω y dA = 0
    ωc = zeros(length(nodes))
    for k in eachindex(segs)
        ωc[k + 1] = ωc[k] + X[k] * Y[k + 1] - X[k + 1] * Y[k]
    end
    Iωx, Iωy = integrate(ωc, Y), integrate(ωc, X)
    xs, ys = [-Ixy Iy; -Ix Ixy] \ [-Iωy, -Iωx]          # relative to the centroid
    ω = ωc .- xs .* Y .+ ys .* X
    ω .-= integrate(ω, ones(length(nodes))) / A        # normalize: ∫ω dA = 0
    Cw = integrate(ω, ω)

    return (; A, xc, yc, Ix, Iy, Ixy, J, Cw, xs = xc + xs, ys = yc + ys)
end

props = thin_walled_properties(nodes, t)

# Load at the centerline of the top flange, located relative to the shear center
x_load, y_load = b / 2, -h / 2
ax = x_load - props.xs          # +x from the shear center
ay = props.ys - y_load          # above the shear center (y is downward)

section = Section(Ix = props.Ix, Iy = props.Iy, Ixy = props.Ixy,
                  J = props.J, Cw = props.Cw, ax = ax, ay = ay)
material = Material(E = 29500.0, G = 11300.0)          # ksi

println("Z8x2.5 16 ga (0.75 in lip) section properties (centerline, sharp corners):")
println("  A   = ", round(props.A, digits = 4), " in²")
println("  Ix  = ", round(props.Ix, digits = 4), " in⁴")
println("  Iy  = ", round(props.Iy, digits = 4), " in⁴")
println("  Ixy = ", round(props.Ixy, digits = 4), " in⁴")
println("  J   = ", round(props.J, sigdigits = 4), " in⁴")
println("  Cw  = ", round(props.Cw, digits = 4), " in⁶")
println("  ax  = ", round(ax, digits = 4), " in,  ay = ", round(ay, digits = 4), " in")

# ------------------------------------------------------------------
# Beam model
# ------------------------------------------------------------------

L = 25.0 * 12                   # in
q = 50.0 / 1000 / 12            # 50 plf → kip/in
pinned = ("fixed", "free", "fixed", "free", "fixed", "free")   # (u, u', v, v', ϕ, ϕ')

model = BeamModel(L = L, nel = 50, section = section, material = material,
                  restraint = Restraint(kx = 0.0, kϕ = 0.0), q = q,
                  bc_left = pinned, bc_right = pinned)

# Stability check: with Kg the tangent stiffness must stay positive definite,
# otherwise the load is beyond the critical (lateral-torsional) load and the
# second-order solution is not physically meaningful.
K = elastic_stiffness(model) + geometric_stiffness(model)
free = setdiff(1:size(K, 1), constrained_dofs(model))
stable = isposdef(Symmetric(K[free, free]))

# Unbraced, pinned-pinned: ϕ decouples from u, v (kx = 0), so the first-mode
# critical load is q_cr = (E Cw λ⁴ + G J λ²) / ay with λ = π / L.
λ = π / L
qcr = (material.E * section.Cw * λ^4 + material.G * section.J * λ^2) / ay
println("\nq     = ", round(q * 12000, digits = 2), " plf")
println("q_cr  = ", round(qcr * 12000, digits = 2), " plf  (load-height LTB, unbraced)")
stable || @warn "Tangent stiffness is not positive definite: q exceeds the critical load " *
                "for the unbraced purlin, so the second-order (geometric=true) twist is " *
                "not meaningful. First-order (geometric=false) results are plotted as well."

sol = solve_beam(model)                              # second-order (with Kg)
sol1 = solve_beam(model; geometric = false)          # first-order

z = range(0, L, 301)
results = [(name = "second order (with Kg)", s = sol), (name = "first order", s = sol1)]

for r in results
    println("\n", r.name, " results at midspan:")
    println("  v(L/2) = ", round(vertical_deflection(r.s, L / 2), sigdigits = 4), " in")
    println("  u(L/2) = ", round(lateral_deflection(r.s, L / 2), sigdigits = 4), " in")
    println("  ϕ(L/2) = ", round(twist(r.s, L / 2), sigdigits = 4), " rad")
end

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

zft = z ./ 12
pv = plot(xlabel = "z (ft)", ylabel = "v (in)", title = "Vertical deflection v (+ down)")
pu = plot(xlabel = "z (ft)", ylabel = "u (in)", title = "Lateral deflection u")
pϕ = plot(xlabel = "z (ft)", ylabel = "ϕ (rad)", title = "Twist ϕ")
for r in results
    plot!(pv, zft, vertical_deflection.(Ref(r.s), z), label = r.name)
    plot!(pu, zft, lateral_deflection.(Ref(r.s), z), label = r.name)
    plot!(pϕ, zft, twist.(Ref(r.s), z), label = r.name)
end
fig = plot(pv, pu, pϕ, layout = (3, 1), size = (800, 900), legend = :best)
display(fig)
savefig(fig, joinpath(@__DIR__, "Z8x2p5_16ga_single_span.png"))
