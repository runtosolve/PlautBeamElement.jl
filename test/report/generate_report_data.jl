# Generates the figures (figures/*.svg) and numerical data (data.typ) used by
# purlin_report.typ, for two unbraced single-span Z purlins:
#
#   Z8x2   16 ga, 0.5 in lips   (PurlinLine database: "Z8x2 060 0.5lip")
#   Z8x2.5 16 ga, 0.75 in lips  (RoofHugger_PlautBeamFE database: "Z8x2.5 060 0.75 lip")
#
# 25 ft span, 50 plf gravity load on the top flange centerline, pinned ends with
# free warping, no continuous bracing.
#
# Run from the package root:   julia --project=. test/report/generate_report_data.jl
# then compile the report:     typst compile test/report/purlin_report.typ
#
# Units: kip, in.

using PlautBeamElement: Section, Material, Restraint, BeamModel, solve_beam,
                              elastic_stiffness, geometric_stiffness, constrained_dofs,
                              elastic_element_stiffness, element_load_vector, element_dofs,
                              lateral_deflection, vertical_deflection, twist,
                              element_end_forces
using LinearAlgebra
using Plots

const OUT = @__DIR__
const FIG = joinpath(OUT, "figures")
mkpath(FIG)

# ------------------------------------------------------------------
# Thin-walled section properties (centerline model, sharp corners)
# ------------------------------------------------------------------

"Centerline node coordinates of a lipped Z in the paper's axes (x horizontal, y down)."
function z_nodes(; t, H, B, D, θlip = deg2rad(55.0))
    h, b = H - t, B - t
    [(-b - D * cos(θlip),  h / 2 - D * sin(θlip)),   # 1 bottom lip tip
     (-b,  h / 2),                                  # 2 bottom flange tip
     ( 0.0,  h / 2),                                # 3 bottom web-flange corner
     ( 0.0, -h / 2),                                # 4 top web-flange corner
     ( b, -h / 2),                                  # 5 top flange tip
     ( b + D * cos(θlip), -h / 2 + D * sin(θlip))]  # 6 top lip tip
end

"""
Thin-walled open-section properties of a single chain of straight segments of
thickness `t`, with per-segment contributions and nodal sectorial coordinates
for the report appendix.
"""
function thin_walled_properties(nodes, t)
    nseg = length(nodes) - 1
    Ls = [hypot(nodes[k + 1][1] - nodes[k][1], nodes[k + 1][2] - nodes[k][2]) for k in 1:nseg]
    # ∫ f g dA over segment k with f, g linear between nodal values
    seg_int(k, f, g) = t * Ls[k] / 6 * (2f[k] * g[k] + f[k] * g[k + 1] + f[k + 1] * g[k] + 2f[k + 1] * g[k + 1])
    integrate(f, g) = sum(seg_int(k, f, g) for k in 1:nseg)

    A = t * sum(Ls)
    xc = sum(t * Ls[k] * (nodes[k][1] + nodes[k + 1][1]) / 2 for k in 1:nseg) / A
    yc = sum(t * Ls[k] * (nodes[k][2] + nodes[k + 1][2]) / 2 for k in 1:nseg) / A
    X = [n[1] - xc for n in nodes]
    Y = [n[2] - yc for n in nodes]

    Ix, Iy, Ixy = integrate(Y, Y), integrate(X, X), integrate(X, Y)
    J = sum(Ls) * t^3 / 3
    segments = [(L = Ls[k], A = t * Ls[k], Ix = seg_int(k, Y, Y), Iy = seg_int(k, X, X),
                 Ixy = seg_int(k, X, Y), J = Ls[k] * t^3 / 3) for k in 1:nseg]

    # sectorial coordinate about the centroid, then shift the pole to the shear
    # center so that ∫ω x dA = ∫ω y dA = 0, and normalize so that ∫ω dA = 0
    ωc = zeros(length(nodes))
    for k in 1:nseg
        ωc[k + 1] = ωc[k] + X[k] * Y[k + 1] - X[k + 1] * Y[k]
    end
    Iωx, Iωy = integrate(ωc, Y), integrate(ωc, X)
    xs, ys = [-Ixy Iy; -Ix Ixy] \ [-Iωy, -Iωx]
    ω = ωc .- xs .* Y .+ ys .* X
    ω .-= integrate(ω, ones(length(nodes))) / A
    Cw = integrate(ω, ω)

    return (; A, xc, yc, Ix, Iy, Ixy, J, Cw, xs = xc + xs, ys = yc + ys,
              Iωx, Iωy, X, Y, ωc, ωn = ω, segments)
end

# ------------------------------------------------------------------
# Closed-form references (pinned-pinned, kx = kϕ = 0, θ = 0)
# ------------------------------------------------------------------

"Fourier sine-series twist modes ϕₙ; `g = 1` includes the load-height (geometric) term."
twist_mode(n; q, L, E, G, Cw, J, ax, ay, g) =
    let λ = n * π / L
        (4 / (n * π)) * q * ax / (E * Cw * λ^4 + G * J * λ^2 - g * q * ay)
    end

function twist_series(z; q, L, E, G, Cw, J, ax, ay, g, nmodes = 20001)
    ϕ = T = Bm = 0.0
    for n in 1:2:nmodes
        λ = n * π / L
        ϕn = twist_mode(n; q, L, E, G, Cw, J, ax, ay, g)
        ϕ += ϕn * sin(λ * z)
        T += ϕn * λ * (G * J + E * Cw * λ^2) * cos(λ * z)
        Bm -= E * Cw * ϕn * λ^2 * sin(λ * z)
    end
    return (ϕ = ϕ, T = T, B = Bm)
end

# ------------------------------------------------------------------
# Problem definition
# ------------------------------------------------------------------

const E, G = 29500.0, 11300.0           # ksi
const L = 25.0 * 12                     # in
const q = 50.0 / 1000 / 12              # 50 plf → kip/in
const NEL = 50
const PINNED = ("fixed", "free", "fixed", "free", "fixed", "free")   # (u, u', v, v', ϕ, ϕ')

cases = [
    (key = "z8x2",  label = "Z8x2 16 ga, 0.5 in lips",   db = "Z8x2 060 0.5lip",
     source = "PurlinLine.jl database/Purlins.csv",
     t = 0.060, H = 8.0, B = 2.0, D = 0.5),
    (key = "z8x25", label = "Z8x2.5 16 ga, 0.75 in lips", db = "Z8x2.5 060 0.75 lip",
     source = "RoofHugger_PlautBeamFE.jl database/Purlins.csv",
     t = 0.060, H = 8.0, B = 2.5, D = 0.75),
]

const VDOF = (3, 4, 9, 10)
const ΦDOF = (5, 6, 11, 12)

function run_case(c)
    nodes = z_nodes(; t = c.t, H = c.H, B = c.B, D = c.D)
    p = thin_walled_properties(nodes, c.t)
    b, h = c.B - c.t, c.H - c.t
    x_load, y_load = b / 2, -h / 2                  # top flange centerline
    ax, ay = x_load - p.xs, p.ys - y_load
    section = Section(Ix = p.Ix, Iy = p.Iy, Ixy = p.Ixy, J = p.J, Cw = p.Cw, ax = ax, ay = ay)
    material = Material(E = E, G = G)
    model = BeamModel(L = L, nel = NEL, section = section, material = material,
                      restraint = Restraint(kx = 0.0, kϕ = 0.0), q = q,
                      bc_left = PINNED, bc_right = PINNED)

    sol2 = solve_beam(model)                         # second order (with Kg)
    sol1 = solve_beam(model; geometric = false)      # first order

    # stability: tangent stiffness at the applied load, and the FE critical load
    # from the generalized eigenproblem (Ke + μ Kg(q=1)) φ = 0
    free = setdiff(1:6 * (NEL + 1), constrained_dofs(model))
    Ke = elastic_stiffness(model)[free, free]
    Kg = geometric_stiffness(model)[free, free]
    stable = isposdef(Symmetric(Ke + Kg))
    μ = eigvals(Ke \ (-Kg ./ q))
    qcr_fe = 1 / maximum(real, μ)
    λ1 = π / L
    qcr_cf = (E * p.Cw * λ1^4 + G * p.J * λ1^2) / ay

    # closed-form bending (qx = 0): v'''' = q Iy/(EΔ), u'''' = -q Ixy/(EΔ)
    Δ = p.Ix * p.Iy - p.Ixy^2
    v_cf = 5q * L^4 / 384 * p.Iy / (E * Δ)
    u_cf = -5q * L^4 / 384 * p.Ixy / (E * Δ)
    sargs = (; q, L, E, G, Cw = p.Cw, J = p.J, ax, ay)
    s1_mid, s2_mid = twist_series(L / 2; sargs..., g = 0), twist_series(L / 2; sargs..., g = 1)
    s1_end, s2_end = twist_series(0.0; sargs..., g = 0), twist_series(0.0; sargs..., g = 1)
    terms = [(n = n,
              t1 = twist_mode(n; sargs..., g = 0) * sin(n * π / 2),
              t2 = twist_mode(n; sargs..., g = 1) * sin(n * π / 2),
              k = E * p.Cw * (n * π / L)^4 + G * p.J * (n * π / L)^2)
             for n in (1, 3, 5, 7, 9)]

    # ---- field results along the span
    z = collect(range(0, L, 301))
    f1, f2 = element_end_forces(sol1), element_end_forces(sol2)
    znode = [[fe.end1.z for fe in f1]; f1[end].end2.z]
    nodal(f, a) = [[getfield(fe.end1, a) for fe in f]; getfield(f[end].end2, a)]
    imid = NEL ÷ 2                                   # element ending at z = L/2

    fe_mid(f) = (Vx = f[imid].end2.Vx, My = f[imid].end2.My, Vy = f[imid].end2.Vy,
                 Mx = f[imid].end2.Mx, T = f[imid].end2.T, B = f[imid].end2.B)
    fe_end(f) = (Vx = f[1].end1.Vx, My = f[1].end1.My, Vy = f[1].end1.Vy,
                 Mx = f[1].end1.Mx, T = f[1].end1.T, B = f[1].end1.B)
    maxabs(f, a) = maximum(abs, nodal(f, a))

    # ---- element 1 force recovery (first order): g = ke dₑ - fₑ
    ke = elastic_element_stiffness(model)
    de = sol1.d[element_dofs(1)]
    fe = element_load_vector(model, 1)
    kd = ke * de
    g = kd - fe
    Le = L / NEL

    # ---- plots
    plot_section(c, nodes, p, x_load, y_load)
    plot_deflections(c, z, sol1, sol2)
    plot_forces(c, znode, f1, f2, nodal)

    res(sol) = (v = vertical_deflection(sol, L / 2), u = lateral_deflection(sol, L / 2),
                phi = twist(sol, L / 2),
                phimax = maximum(abs, twist.(Ref(sol), z)))

    return (
        label = c.label, db = c.db, source = c.source,
        t = c.t, H = c.H, B = c.B, D = c.D, lipangle = 55.0, h = h, b = b,
        nodes = [(x = n[1], y = n[2]) for n in nodes],
        props = (A = p.A, xc = p.xc, yc = p.yc, Ix = p.Ix, Iy = p.Iy, Ixy = p.Ixy,
                 J = p.J, Cw = p.Cw, xs = p.xs, ys = p.ys, Iwx = p.Iωx, Iwy = p.Iωy,
                 Delta = Δ, ax = ax, ay = ay, xload = x_load, yload = y_load),
        segments = p.segments,
        omega = [(X = p.X[i], Y = p.Y[i], wc = p.ωc[i], wn = p.ωn[i]) for i in eachindex(nodes)],
        stable = stable, qcr_fe = qcr_fe, qcr_cf = qcr_cf,
        first = merge(res(sol1), (mid = fe_mid(f1), endf = fe_end(f1),
                                  My_max = maxabs(f1, :My), Vx_max = maxabs(f1, :Vx),
                                  residual = sol1.residual_norm)),
        second = merge(res(sol2), (mid = fe_mid(f2), endf = fe_end(f2),
                                   My_max = maxabs(f2, :My), Vx_max = maxabs(f2, :Vx),
                                   residual = sol2.residual_norm)),
        closed = (v = v_cf, u = u_cf,
                  Mx_mid = -q * L^2 / 8, Vy_end = q * L / 2, T_end1 = q * ax * L / 2,
                  phi1 = s1_mid.ϕ, phi2 = s2_mid.ϕ, B1 = s1_mid.B, B2 = s2_mid.B,
                  T1 = s1_end.T, T2 = s2_end.T, terms = terms),
        element1 = (Le = Le, d = de, kd = kd, f = fe, g = g,
                    EIx = E * p.Ix, kvv = E * p.Ix * 12 / Le^3,
                    fv = [fe[i] for i in VDOF], fphi = [fe[i] for i in ΦDOF]),
    )
end

# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

default(fontfamily = "sans-serif", titlefontsize = 10, guidefontsize = 9,
        tickfontsize = 8, legendfontsize = 8, linewidth = 1.8, framestyle = :box,
        grid = true, gridalpha = 0.15)
const C1, C2 = RGB(0.16, 0.40, 0.70), RGB(0.80, 0.33, 0.10)

function plot_section(c, nodes, p, xl, yl)
    xs, ys = first.(nodes), last.(nodes)
    plt = plot(xs, ys, yflip = true, aspect_ratio = :equal, color = :black, lw = 2.5,
               label = "centerline", legend = :outerbottom, xlabel = "x (in)",
               ylabel = "y (in), positive down", title = c.label,
               xlims = (-3.5, 3.5), ylims = (-6.0, 4.6), size = (380, 560))
    scatter!(plt, [p.xs], [p.ys], color = C1, ms = 5, label = "centroid = shear center")
    # arrowheads are drawn incorrectly on a flipped axis, so use a line plus a marker
    plot!(plt, [xl, xl], [yl - 1.6, yl - 0.35], color = C2, lw = 2, label = "")
    scatter!(plt, [xl], [yl - 0.45], color = C2, markershape = :dtriangle, ms = 6,
             markerstrokewidth = 0, label = "")
    annotate!(plt, xl + 0.15, yl - 1.4, text("q = 50 plf", 8, :left, C2))
    scatter!(plt, [xl], [yl], color = C2, ms = 5, markershape = :diamond,
             label = "load point (top flange CL)")
    savefig(plt, joinpath(FIG, "section_$(c.key).svg"))
end

function plot_deflections(c, z, sol1, sol2)
    zft = z ./ 12
    panel(f, ylab, ttl) = begin
        pl = plot(xlabel = "z (ft)", ylabel = ylab, title = ttl, legend = :outerright)
        plot!(pl, zft, f.(Ref(sol1), z), color = C1, label = "first order")
        plot!(pl, zft, f.(Ref(sol2), z), color = C2, ls = :dash, label = "second order")
        pl
    end
    fig = plot(panel(vertical_deflection, "v (in)", "Vertical deflection v (+ down)"),
               panel(lateral_deflection, "u (in)", "Lateral deflection u (+x)"),
               panel(twist, "ϕ (rad)", "Twist ϕ (+ clockwise, Fig. 1 of Plaut & Moen)"),
               layout = (3, 1), size = (700, 720), left_margin = 4Plots.mm)
    savefig(fig, joinpath(FIG, "deflections_$(c.key).svg"))
end

function plot_forces(c, znode, f1, f2, nodal)
    zft = znode ./ 12
    # My and Vx vanish identically for this load case (qx = 0); without a fixed
    # scale their round-off (~1e-10) would be auto-scaled into a misleading curve
    zero_scale = Dict(:My => maximum(abs, nodal(f1, :Mx)), :Vx => maximum(abs, nodal(f1, :Vy)))
    panel(a, ylab, ttl) = begin
        pl = plot(xlabel = "z (ft)", ylabel = ylab, title = ttl, legend = :best)
        plot!(pl, zft, nodal(f1, a), color = C1, label = "first order")
        plot!(pl, zft, nodal(f2, a), color = C2, ls = :dash, label = "second order")
        haskey(zero_scale, a) && plot!(pl, ylims = (-0.1, 0.1) .* zero_scale[a])
        pl
    end
    fig = plot(panel(:Mx, "Mx (kip·in)", "Bending moment Mx"),
               panel(:My, "My (kip·in)", "Bending moment My (≡ 0)"),
               panel(:Vy, "Vy (kip)", "Shear Vy"),
               panel(:Vx, "Vx (kip)", "Shear Vx (≡ 0)"),
               panel(:T, "T (kip·in)", "Torque T = GJϕ′ − ECwϕ‴"),
               panel(:B, "B (kip·in²)", "Bimoment B = ECwϕ″"),
               layout = (3, 2), size = (760, 760), left_margin = 4Plots.mm)
    savefig(fig, joinpath(FIG, "forces_$(c.key).svg"))
end

# ------------------------------------------------------------------
# Typst data export
# ------------------------------------------------------------------

typ(x::Bool) = string(x)
typ(x::Integer) = string(x)
typ(x::Real) = isfinite(x) ? (abs(x) < 1e-300 ? "0.0" : repr(Float64(x))) : "none"
typ(s::AbstractString) = "\"" * replace(s, "\\" => "\\\\", "\"" => "\\\"") * "\""
typ(v::Union{AbstractVector, Tuple}) =
    isempty(v) ? "()" : "(" * join(typ.(v), ", ") * (length(v) == 1 ? ",)" : ")")
typ(nt::NamedTuple) =
    isempty(nt) ? "(:)" : "(" * join(["$k: $(typ(v))" for (k, v) in pairs(nt)], ", ") * ")"

results = [c.key => run_case(c) for c in cases]

open(joinpath(OUT, "data.typ"), "w") do io
    println(io, "// Generated by generate_report_data.jl -- do not edit by hand.")
    println(io, "#let inputs = ", typ((E = E, G = G, L = L, q = q, q_plf = q * 12000,
                                       nel = NEL, Le = L / NEL)))
    println(io, "#let cases = (")
    for (k, r) in results
        println(io, "  ", k, ": ", typ(r), ",")
    end
    println(io, ")")
end

for (k, r) in results
    println(k, ": v = ", round(r.first.v, sigdigits = 4), " in, u = ", round(r.first.u, sigdigits = 4),
            " in, ϕ1 = ", round(r.first.phi, sigdigits = 4), ", ϕ2 = ", round(r.second.phi, sigdigits = 4),
            " rad, qcr = ", round(r.qcr_fe * 12000, sigdigits = 4), " plf (FE), ",
            round(r.qcr_cf * 12000, sigdigits = 4), " plf (closed form)")
end
