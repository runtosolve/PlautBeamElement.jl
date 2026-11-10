# Z8x2.5 16 ga, 25 ft single span, unbraced, gravity line load on the top-flange flat:
# Elikem's three-way validation (rev. 29 Oct 2026) redrawn with the v0.5 Plaut
# formulation (load height + moment coupling) added to his curves.
#
#   * Abaqus S4R shell (nlgeom = YES): digitized from the report's figures by
#     digitize_abaqus.py (abaqus_digitized.csv; checked against the report tables),
#     plus the report's tabulated linearised twist where its markers are hidden.
#   * Plaut theory, linear and v0.4 (load height only): recomputed here with
#     moment_coupling = false; they reproduce the report's curves and its 21.9 lb/ft.
#   * Plaut theory, v0.5: the default formulation, moment coupling included.
#
# Run from the package root:  julia --project=. test/abaqus_validation/plot_vs_abaqus.jl
# (Plots must be available, e.g. in the global environment.) Units: kip, in; loads in lb/ft.

using PlautBeamElement: Section, Material, BeamModel, solve_beam, critical_load,
                        lateral_deflection, vertical_deflection, twist
using Plots

const HERE = @__DIR__
const FIG = joinpath(HERE, "figures")
mkpath(FIG)

# ------------------------------------------------------------------
# Plaut models (inputs from Section 4 of the report)
# ------------------------------------------------------------------

const SEC = Section(Ix = 8.434, Iy = 1.466, Ixy = -2.563, J = 0.001050, Cw = 16.76,
                    ax = 1.2793, ay = 3.97)
const MAT = Material(E = 29500.0)              # G = E/2.6
const L = 300.0
const PLF = 1 / 12000                          # lb/ft → kip/in

model(q_plf; coupling) = BeamModel(L = L, nel = 60, section = SEC, material = MAT,
                                   q = q_plf * PLF, moment_coupling = coupling)

qcr_old = critical_load(model(1.0; coupling = false))   # lb/ft, since q = 1 lb/ft
qcr_new = critical_load(model(1.0; coupling = true))
const QCR_ABAQUS = 11.2

# midspan response in the Abaqus frame (v and ϕ change sign: Plaut's y is downward)
function midspan(q_plf; coupling, geometric = true)
    s = solve_beam(model(q_plf; coupling); geometric)
    (u = lateral_deflection(s, L / 2), v = -vertical_deflection(s, L / 2), ϕ = -twist(s, L / 2))
end

sweep(qmax, n = 400) = collect(range(0.0, qmax, n))
q_lin = sweep(25.0)
q_old = sweep(0.995 * qcr_old)                 # stop just short of each singularity
q_new = sweep(0.995 * qcr_new)
r_lin = midspan.(q_lin; coupling = false, geometric = false)
r_old = midspan.(q_old; coupling = false)
r_new = midspan.(q_new; coupling = true)

# ------------------------------------------------------------------
# Abaqus data
# ------------------------------------------------------------------

abq = Dict{String, Vector{Tuple{Float64, Float64}}}()
for line in Iterators.drop(eachline(joinpath(HERE, "abaqus_digitized.csv")), 1)
    qty, q, val, _ = split(line, ',')
    push!(get!(abq, qty, Tuple{Float64, Float64}[]), (parse(Float64, q), parse(Float64, val)))
end
foreach(sort!, values(abq))
# report table (Abaqus linearised twist); its markers are hidden under the hollow
# circles in the full-range figure
const TABLE_Q = [0.5, 1.0, 2.4, 4.5, 6.2, 7.7, 9.0, 10.6, 11.9]
const TABLE = (u = [0.0647, 0.132, 0.358, 0.775, 1.17, 1.57, 1.92, 2.32, 2.59],
               v = [-0.0375, -0.078, -0.217, -0.514, -0.859, -1.30, -1.80, -2.48, -3.09],
               ϕ = [-0.0095, -0.020, -0.059, -0.142, -0.223, -0.308, -0.390, -0.472, -0.535])
phi_lin = sort(vcat(abq["phi_lin"], [(q, p) for (q, p) in zip(TABLE_Q, TABLE.ϕ) if q > 6.0]))
xy(pts) = (first.(pts), last.(pts))

# ------------------------------------------------------------------
# Figures: Elikem's styling, the v0.5 curve added in purple
# ------------------------------------------------------------------

const C_LIN, C_OLD, C_ABQ, C_NEW = "#e8a317", "#e8643c", "#2a76d2", "#7b3fb0"
const LAB_LIN = "Plaut theory, linear (geometric terms off)"
const LAB_OLD = "Plaut theory, v0.4: load height only (report's curve)"
const LAB_NEW = "Plaut theory, v0.5: load height + moment coupling (new)"
buck = "buckling loads: Abaqus $(QCR_ABAQUS) · v0.4 $(round(qcr_old, digits = 1)) · " *
       "v0.5 $(round(qcr_new, digits = 1)) lb/ft"

default(fontfamily = "sans-serif", framestyle = :box, grid = true, gridalpha = 0.25,
        legendfontsize = 10, guidefontsize = 13, tickfontsize = 10, titlefontsize = 14,
        size = (1140, 840), left_margin = 6Plots.mm, bottom_margin = 4Plots.mm,
        legend = :outerbottom)

function base(title, ylabel, xmax, ylims; window = true)
    p = plot(; title, xlabel = "line load q (lb/ft)", ylabel, xlims = (0, xmax), ylims)
    window && vspan!(p, [0, 3.5], color = "#cfeee3", alpha = 0.5, linealpha = 0,
                     label = "small-deformation window (Abaqus twist ≤ 0.1 rad): q ≤ 3.5 lb/ft")
    return p
end

function add_theory!(p, f)
    plot!(p, q_lin, f.(r_lin), color = C_LIN, ls = :dash, lw = 2.5, label = LAB_LIN)
    plot!(p, q_old, f.(r_old), color = C_OLD, lw = 2.5, label = LAB_OLD)
    plot!(p, q_new, f.(r_new), color = C_NEW, lw = 3.5, label = LAB_NEW)
end

function add_buckling!(p)
    vline!(p, [QCR_ABAQUS], color = :gray40, ls = :dot, lw = 1.5, label = buck)
    vline!(p, [qcr_old], color = C_OLD, ls = :dot, lw = 1.5, label = "")
    vline!(p, [qcr_new], color = C_NEW, ls = :dot, lw = 1.5, label = "")
end

# u
pu = base("Midspan lateral deflection u vs line load (0–25 lb/ft)",
          "u (in), + toward the top-flange side", 25, (0, 4))
add_theory!(pu, r -> r.u)
plot!(pu, xy(abq["u"])..., color = C_ABQ, lw = 2.5, marker = :circle, ms = 5, msw = 0,
      label = "Abaqus S4R shell, nlgeom=YES, incremental")
add_buckling!(pu)
savefig(pu, joinpath(FIG, "u_vs_load.png"))

# v
pv = base("Midspan vertical deflection v vs line load (0–25 lb/ft)",
          "v (in), − is downward", 25, (-10, 0))
add_theory!(pv, r -> r.v)
plot!(pv, xy(abq["v"])..., color = C_ABQ, lw = 2.5, marker = :circle, ms = 5, msw = 0,
      label = "Abaqus S4R shell, nlgeom=YES, incremental")
add_buckling!(pv)
savefig(pv, joinpath(FIG, "v_vs_load.png"))

# ϕ, 0–25 lb/ft
pϕ = base("Midspan twist ϕ vs line load (0–25 lb/ft)", "ϕ (rad)", 25, (-3, 0.2))
add_theory!(pϕ, r -> r.ϕ)
plot!(pϕ, xy(phi_lin)..., color = C_ABQ, lw = 2.5, marker = :circle, ms = 5, msw = 0,
      label = "Abaqus S4R shell, nlgeom=YES, incremental — linearised twist −ΔU1/h")
scatter!(pϕ, xy(abq["phi_atan"])..., color = :white, msc = C_ABQ, msw = 2, ms = 7,
         label = "Abaqus, true inclination of the web line, atan(ΔU1/h′)")
add_buckling!(pϕ)
savefig(pϕ, joinpath(FIG, "phi_vs_load.png"))

# ϕ, 0–6 lb/ft
ps = base("Midspan twist, 0–6 lb/ft (the small-deformation range)", "ϕ (rad)", 6,
          (-0.25, 0.01))
add_theory!(ps, r -> r.ϕ)
plot!(ps, xy([(0.0, 0.0); abq["phi_lin"][first.(abq["phi_lin"]) .<= 6.0]])..., color = C_ABQ,
      lw = 2.5, marker = :circle, ms = 5, msw = 0, label = "Abaqus, incremental run (strict controls)")
scatter!(ps, xy(abq["phi_fixed"])..., color = C_ABQ, marker = :utriangle, ms = 7, msw = 0,
         label = "Abaqus, fixed 4.8 lb/ft run (8 increments)")
savefig(ps, joinpath(FIG, "phi_vs_load_0-6.png"))

# ------------------------------------------------------------------
# Comparison at the report's tabulated loads
# ------------------------------------------------------------------

pct(a, b) = round(Int, 100 * (a - b) / b)
open(joinpath(HERE, "comparison_table.csv"), "w") do io
    println(io, "q_plf,quantity,abaqus,plaut_v04,plaut_v05,v04_vs_abaqus_pct,v05_vs_abaqus_pct")
    for (i, q) in enumerate(TABLE_Q)
        o, n = midspan(q; coupling = false), midspan(q; coupling = true)
        for (name, a, vo, vn) in (("u", TABLE.u[i], o.u, n.u), ("v", TABLE.v[i], o.v, n.v),
                                  ("phi", TABLE.ϕ[i], o.ϕ, n.ϕ))
            println(io, join((q, name, a, round(vo, sigdigits = 4), round(vn, sigdigits = 4),
                              pct(vo, a), q < qcr_new ? pct(vn, a) : "past q_cr"), ','))
        end
    end
end

println("q_cr: v0.4 = ", round(qcr_old, digits = 2), " lb/ft, v0.5 = ", round(qcr_new, digits = 2),
        " lb/ft, Abaqus = ", QCR_ABAQUS, " lb/ft")
print(read(joinpath(HERE, "comparison_table.csv"), String))
