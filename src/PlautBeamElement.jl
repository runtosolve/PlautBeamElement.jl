"""
    PlautBeamElement

Finite element implementation of the lateral-torsional deformation equations for
single-span and two-span thin-walled beams with continuous bracing from:

    Plaut, R.H., Moen, C.D. (2021). "Lateral-torsional deformations of single-span
    and two-span thin-walled beams with continuous bracing." Journal of
    Constructional Steel Research, 179, 106534.
    https://doi.org/10.1016/j.jcsr.2021.106534

The governing equilibrium equations (primes denote d/dz), Eqs. (2)-(4):

    EIy u'''' + EIxy v'''' + kx (u + ay ϕ)                = qx
    EIx v'''' + EIxy u''''                                = qy
    ECw ϕ'''' - GJ ϕ'' + kx ay (u + ay ϕ) + kϕ ϕ          = qx (ay - ax ϕ) + qy (ax + ay ϕ)

are discretized with a two-node beam element using cubic Hermite interpolation for
each of the three fields u(z), v(z), ϕ(z). Each node carries six degrees of freedom:

    (u, u', v, v', ϕ, ϕ')

i.e. lateral and vertical translations, rotations about the y and x axes, twist,
and warping (rate of twist).

The distributed load may be uniform, trapezoidal (linearly varying between the
end intensities), or given independently per span (see [`BeamModel`](@ref)),
allowing a load to jump discontinuously at a shared support. The ϕ-dependent
load terms on the right-hand side of Eq. (4) are
moved to the left-hand side and assembled as a geometric (load) stiffness matrix

    Kg[ϕϕ] = ∫ (qx(z) ax - qy(z) ay) Nᵀ N dz

so the discrete equilibrium is  [Ke + Kg(q)] d = F(q). Kg depends on the load q
but not on the displacements d, so for a given load this is a linear system and
is solved directly (LU factorization of the reduced stiffness). The
load-deformation response is nonlinear in q only because Kg changes with q.

Consistent units are expected, e.g. N and mm: E, G in N/mm², I in mm⁴, J in mm⁴,
Cw in mm⁶, kx in N/mm², kϕ in N/rad (= N·mm/rad per mm), q in N/mm (= kN/m).
"""
module PlautBeamElement

using LinearAlgebra

# ------------------------------------------------------------------
# Input types
# ------------------------------------------------------------------

"""
    Section(; Ix, Iy, Ixy=0.0, J, Cw, ax, ay)

Cross-section properties in the paper's centroidal x-y axes (Fig. 1, y downward):
principal axes are NOT assumed; `Ixy` couples the two bending equations.
`ax` and `ay` locate the applied load (center of top flange) relative to the shear
center, parallel to the x and y axes.
"""
struct Section
    Ix::Float64
    Iy::Float64
    Ixy::Float64
    J::Float64
    Cw::Float64
    ax::Float64
    ay::Float64
end
Section(; Ix, Iy, Ixy = 0.0, J, Cw, ax, ay) = Section(Ix, Iy, Ixy, J, Cw, ax, ay)

"""
    Material(; E, G=E/2.6)

Linearly elastic material. The paper uses G = E/2.6 for steel.
"""
struct Material
    E::Float64
    G::Float64
end
Material(; E, G = E / 2.6) = Material(E, G)

"""
    Restraint(; kx=0.0, kϕ=0.0)

Continuous elastic bracing (Fig. 1): `kx` is the lateral stiffness per unit length
acting at the top of the top flange (offset `ay` from the shear center), `kϕ` is
the rotational stiffness per unit length.
"""
struct Restraint
    kx::Float64
    kϕ::Float64
end
Restraint(; kx = 0.0, kϕ = 0.0) = Restraint(kx, kϕ)

"""
    BeamModel(; L, nel, section, material, restraint=Restraint(), q, θ=0.0,
                bc_left=:pinned, bc_right=:pinned, interior_supports=Float64[],
                interior_dofs=nothing)

Uniform beam of length `L` meshed with `nel` equal elements, carrying a distributed
load applied at the center of the top flange (positive = gravity, negative =
uplift) with roof slope angle `θ` (rad, Eq. 1 and Fig. 2). The load may be

  * uniform — `q` a scalar, or
  * trapezoidal — `q = (q1, q2)`, the intensities at `z = 0` and `z = L`, varying
    linearly in between (a triangular load is the special case with one end zero), or
  * per-span — `q` a `Vector` with one entry per span (a "span" is the segment
    between two consecutive supports: the two ends plus every `interior_support`,
    in order), each entry itself a scalar (uniform over that span) or a 2-tuple
    (trapezoidal, linear from the first value at that span's own start to the
    second at that span's own end). Spans are independent: adjacent spans need
    not agree at a shared support, so a load can jump discontinuously there
    (e.g. a uniform interior span next to a trapezoidal drift-loaded end span).

With a non-uniform load, both the consistent load vector and the geometric (load)
stiffness vary from element to element; the element integrals are evaluated
exactly with 4-point Gauss quadrature.

End conditions `bc_left`/`bc_right`, and each entry of `interior_dofs`, accept
either a preset symbol or a full 6-element `(u, u', v, v', ϕ, ϕ')` tuple of
`"fixed"`/`"free"` strings, so any support -- end or interior -- can restrain an
arbitrary combination of DOFs:

  * `:pinned` — u = v = ϕ = "fixed" (free to warp and to rotate about x and y)
  * `:fixed`  — u = u' = v = v' = ϕ = ϕ' = "fixed"
  * `:free`   — nothing restrained
  * or a custom tuple, e.g. `("fixed", "free", "fixed", "free", "free", "free")`
    (u, v fixed; u', v', ϕ, ϕ' free)

`interior_supports` is a list of z locations (each must coincide with a mesh node),
e.g. the central support of a two-span beam. `interior_dofs` gives the condition
at each of those locations (one-to-one, same order); it defaults to `:pinned`
(u = v = ϕ = "fixed") at every interior support.
"""

struct BeamModel
    L::Float64
    nel::Int
    section::Section
    material::Material
    restraint::Restraint
    q_per_element::Vector{NTuple{2, Float64}}     # (q1, q2) load intensity at each element's two ends
    θ::Float64
    bc_left::NTuple{6, String}
    bc_right::NTuple{6, String}
    interior_supports::Vector{Float64}
    interior_dofs::Vector{NTuple{6, String}}
end

function BeamModel(; L, nel, section, material, restraint = Restraint(),
                   q, θ = 0.0, bc_left = :pinned, bc_right = :pinned,
                   interior_supports = Float64[], interior_dofs = nothing)

    bc_left = dof_condition(bc_left)
    bc_right = dof_condition(bc_right)
    for bc in (bc_left, bc_right), val in bc
        val in ("fixed", "free") ||
            throw(ArgumentError("bc_left/bc_right entries must be \"fixed\" or \"free\", got $(repr(val))"))
    end

    nel >= 2 || throw(ArgumentError("need at least 2 elements"))
    L = Float64(L)
    interior_supports = Float64.(collect(interior_supports))
    conditions = interior_dofs === nothing ?
        fill(("fixed","free","fixed","free","fixed","free"),length(interior_supports)) :
        [dof_condition(c) for c in interior_dofs]
    length(conditions) == length(interior_supports) || throw(ArgumentError("interior_dofs must have one entry per interior support" *
    "($(length(interior_supports)) supports, got $(length(conditions)) conditions)"))

    for cond in conditions, val in cond
        val in ("fixed", "free") ||
            throw(ArgumentError("interior_dofs entries must be \"fixed\" or \"free\", got $(repr(val))"))
    end
    Le = L / nel
    q_per_element = build_q_per_element(q,L,nel,Le,interior_supports)
    model = BeamModel(L,Int(nel),section,material,restraint,
                        q_per_element,Float64(θ), bc_left, bc_right, interior_supports, conditions)
    support_nodes(model)  # validate interior supports fall on mesh nodes

    return model
end

nnodes(m::BeamModel) = m.nel + 1
ndofs(m::BeamModel) = 6 * nnodes(m)
elength(m::BeamModel) = m.L / m.nel

"""
    compute_support_nodes(interior_supports, L, nel)

Node index (1-based) of each interior support, validating that it coincides
with a mesh node of a beam of length `L` meshed with `nel` equal elements.
"""
function compute_support_nodes(interior_supports, L, nel)
    Le = L / nel
    nn = nel + 1
    nodes = Int[]
    for zs in interior_supports
        node = round(Int, zs / Le) + 1
        isapprox((node - 1) * Le, zs; atol = 1e-8 * L) ||
            throw(ArgumentError("interior support at z = $zs does not coincide with a mesh node"))
        (1 < node < nn) ||
            throw(ArgumentError("interior support at z = $zs is not interior to the span"))
        push!(nodes, node)
    end
    return nodes
end

support_nodes(m::BeamModel) = compute_support_nodes(m.interior_supports, m.L, m.nel)

"""
    build_q_per_element(q, L, nel, Le, interior_supports)

Normalize the user-supplied `q` (see [`BeamModel`](@ref)) into one `(q1, q2)`
pair per element -- the load intensity at that element's two ends.
"""
function build_q_per_element(q, L, nel, Le, interior_supports)
    if q isa AbstractVector
        support_z = sort(interior_supports)
        span_bounds = [0.0; support_z; L]
        nspans = length(span_bounds) - 1
        length(q) == nspans || throw(ArgumentError(
            "q vector must have one entry per span ($nspans spans here), got $(length(q)) entries"))
        boundary_nodes = [1; compute_support_nodes(support_z, L, nel); nel + 1]
        q_per_element = Vector{NTuple{2, Float64}}(undef, nel)
        for i in 1:nspans
            qa, qb = q[i] isa Real ? (Float64(q[i]), Float64(q[i])) :
                     length(q[i]) == 2 ? (Float64(q[i][1]), Float64(q[i][2])) :
                     throw(ArgumentError("span $i entry must be a scalar or two values (q1, q2), got $(q[i])"))
            zA, zB = span_bounds[i], span_bounds[i + 1]
            span_len = zB - zA
            for e in boundary_nodes[i]:(boundary_nodes[i + 1] - 1)
                z1, z2 = (e - 1) * Le, e * Le
                q_per_element[e] = (qa + (qb - qa) * (z1 - zA) / span_len,
                                     qa + (qb - qa) * (z2 - zA) / span_len)
            end
        end
        return q_per_element
    end
    qt = if q isa Real
        (Float64(q), Float64(q))
    elseif length(q) == 2
        (Float64(q[1]), Float64(q[2]))
    else
        throw(ArgumentError("q must be a scalar (uniform), two values (q1, q2) at z = 0 " *
                            "and z = L (trapezoidal), or a per-span vector, got $q"))
    end
    return [(qt[1] + (qt[2] - qt[1]) * (e - 1) * Le / L,
             qt[1] + (qt[2] - qt[1]) * e * Le / L) for e in 1:nel]
end

"Load components (qx1,qy1,qx2,qy2) at the two ends of element `e`, Eq. (1)."
function element_load_endpoints(m::BeamModel, e::Int)
    q1, q2 = m.q_per_element[e]
    s, c = sin(m.θ), cos(m.θ)
    return (qx1 = -q1 * s, qy1 = q1 * c, qx2 = -q2 * s, qy2 = q2 * c)
end

# ------------------------------------------------------------------
# Hermite cubic element matrices on [0, Le], DOF order (w1, w1', w2, w2')
# ------------------------------------------------------------------

"∫ N''ᵀ N'' dz — bending/warping stiffness kernel."
k_bend(Le) = (1 / Le^3) * [ 12.0    6Le   -12.0    6Le
                             6Le   4Le^2   -6Le   2Le^2
                           -12.0   -6Le    12.0   -6Le
                             6Le   2Le^2   -6Le   4Le^2 ]

"∫ N'ᵀ N' dz — St. Venant torsion kernel."
k_grad(Le) = (1 / (30Le)) * [ 36.0    3Le   -36.0    3Le
                               3Le   4Le^2   -3Le   -Le^2
                             -36.0   -3Le    36.0   -3Le
                               3Le   -Le^2   -3Le   4Le^2 ]

"∫ Nᵀ N dz — elastic foundation / load-position kernel."
k_found(Le) = (Le / 420) * [ 156.0   22Le    54.0  -13Le
                              22Le   4Le^2   13Le  -3Le^2
                              54.0   13Le   156.0  -22Le
                             -13Le  -3Le^2  -22Le   4Le^2 ]

# 4-point Gauss-Legendre rule mapped to ξ ∈ [0, 1]: exact for polynomials of
# degree ≤ 7, which covers a linear load weight times Nᵢ Nⱼ (degree 7) and Nᵢ
# (degree 4).
const GAUSS4 = let a = sqrt(3 / 7 - 2 / 7 * sqrt(6 / 5)), b = sqrt(3 / 7 + 2 / 7 * sqrt(6 / 5)),
                   wa = (18 + sqrt(30)) / 72, wb = (18 - sqrt(30)) / 72
    ((ξ = (1 - b) / 2, w = wb), (ξ = (1 - a) / 2, w = wa),
     (ξ = (1 + a) / 2, w = wa), (ξ = (1 + b) / 2, w = wb))
end

"∫ w(ξ) Nᵀ N dz with w varying linearly from `w1` (ξ=0) to `w2` (ξ=1) — exact."
function k_found_linear(Le, w1, w2)
    K = zeros(4, 4)
    for gp in GAUSS4
        N = collect(hermite(gp.ξ, Le))
        c = (w1 * (1 - gp.ξ) + w2 * gp.ξ) * gp.w * Le
        K .+= c .* (N * N')
    end
    return K
end

"∫ w(ξ) Nᵀ dz — consistent nodal loads for a load varying linearly from `w1` to `w2`."
function f_linear(Le, w1, w2)
    f = zeros(4)
    for gp in GAUSS4
        N = collect(hermite(gp.ξ, Le))
        f .+= ((w1 * (1 - gp.ξ) + w2 * gp.ξ) * gp.w * Le) .* N
    end
    return f
end

# Local DOF indices of each field within the 12-DOF element (u, u', v, v', ϕ, ϕ' per node)
const UDOF = (1, 2, 7, 8)
const VDOF = (3, 4, 9, 10)
const ΦDOF = (5, 6, 11, 12)

function add_block!(ke::Matrix{Float64}, rows, cols, coeff::Float64, k4::Matrix{Float64})
    @inbounds for j in 1:4, i in 1:4
        ke[rows[i], cols[j]] += coeff * k4[i, j]
    end
    return ke
end

"""
    elastic_element_stiffness(model)

12×12 elastic stiffness matrix of one element: bending (EIy, EIx, EIxy), warping
(ECw) and St. Venant (GJ) torsion, and the continuous bracing terms kx and kϕ from
the left-hand sides of Eqs. (2)-(4).
"""
function elastic_element_stiffness(m::BeamModel)
    (; E, G) = m.material
    (; Ix, Iy, Ixy, J, Cw, ay) = m.section
    (; kx, kϕ) = m.restraint
    Le = elength(m)
    kb, kg, kf = k_bend(Le), k_grad(Le), k_found(Le)

    ke = zeros(12, 12)
    add_block!(ke, UDOF, UDOF, E * Iy, kb)              # EIy u''''
    add_block!(ke, UDOF, VDOF, E * Ixy, kb)             # EIxy v''''  (Eq. 2)
    add_block!(ke, VDOF, UDOF, E * Ixy, kb)             # EIxy u''''  (Eq. 3)
    add_block!(ke, VDOF, VDOF, E * Ix, kb)              # EIx v''''
    add_block!(ke, ΦDOF, ΦDOF, E * Cw, kb)              # ECw ϕ''''
    add_block!(ke, ΦDOF, ΦDOF, G * J, kg)               # -GJ ϕ''
    add_block!(ke, UDOF, UDOF, kx, kf)                  # kx u
    add_block!(ke, UDOF, ΦDOF, kx * ay, kf)             # kx ay ϕ    (Eq. 2)
    add_block!(ke, ΦDOF, UDOF, kx * ay, kf)             # kx ay u    (Eq. 4)
    add_block!(ke, ΦDOF, ΦDOF, kx * ay^2 + kϕ, kf)      # kx ay² ϕ + kϕ ϕ
    return ke
end

"""
    geometric_element_stiffness(model, e)

12×12 geometric (load) stiffness matrix of element `e`. The ϕ-dependent load terms
on the right-hand side of Eq. (4), qx(ay - ax ϕ) + qy(ax + ay ϕ), contribute
(qx ax - qy ay) ϕ when moved to the left-hand side. For gravity loading on the top
flange (qy > 0) this is destabilizing (softening); for uplift it is stabilizing.
With a trapezoidal load the coefficient varies linearly over the element and is
integrated exactly.
"""
function geometric_element_stiffness(m::BeamModel, e::Int)
    (; ax, ay) = m.section
    Le = elength(m)
    (; qx1, qy1, qx2, qy2) = element_load_endpoints(m, e)
    kf = k_found_linear(Le, qx1 * ax - qy1 * ay, qx2 * ax - qy2 * ay)
    ke = zeros(12, 12)
    add_block!(ke, ΦDOF, ΦDOF, 1.0, kf)
    return ke
end

"""
    element_load_vector(model, e)

12-component consistent load vector of element `e`: qx on u, qy on v, and the
ϕ-independent torque qx ay + qy ax on ϕ (right-hand sides of Eqs. 2-4), each
varying linearly over the element for a trapezoidal load.
"""
function element_load_vector(m::BeamModel, e::Int)
    (; ax, ay) = m.section
    Le = elength(m)
    (; qx1, qy1, qx2, qy2) = element_load_endpoints(m, e)
    fe = zeros(12)
    fe[collect(UDOF)] = f_linear(Le, qx1, qx2)
    fe[collect(VDOF)] = f_linear(Le, qy1, qy2)
    fe[collect(ΦDOF)] = f_linear(Le, qx1 * ay + qy1 * ax, qx2 * ay + qy2 * ax)
    return fe
end

# ------------------------------------------------------------------
# Global assembly
# ------------------------------------------------------------------

element_dofs(e::Int) = (6 * (e - 1)) .+ (1:12)

function assemble(m::BeamModel, ke::Matrix{Float64})
    K = zeros(ndofs(m), ndofs(m))
    for e in 1:m.nel
        dofs = element_dofs(e)
        @views K[dofs, dofs] .+= ke
    end
    return K
end

"Global elastic stiffness matrix (Eqs. 2-4 left-hand sides)."
elastic_stiffness(m::BeamModel) = assemble(m, elastic_element_stiffness(m))

"Global geometric (load) stiffness matrix (ϕ-dependent load terms of Eq. 4)."
function geometric_stiffness(m::BeamModel)
    K = zeros(ndofs(m), ndofs(m))
    for e in 1:m.nel
        dofs = element_dofs(e)
        @views K[dofs, dofs] .+= geometric_element_stiffness(m, e)
    end
    return K
end

"Global consistent load vector."
function load_vector(m::BeamModel)
    F = zeros(ndofs(m))
    for e in 1:m.nel
        @views F[element_dofs(e)] .+= element_load_vector(m, e)
    end
    return F
end

"1-based local DOF indices, in order (u,u',v,v',ϕ,ϕ'), that are \"fixed\" in 'condition'"
fixed_local_dofs(condition) = findall(==("fixed"),condition)

"Expand a bc/support preset symbol into a 6-element (u,u',v,v',ϕ,ϕ') condition; a non-symbol is passed through unchanged."
function dof_condition(bc)
    bc isa Symbol || return bc
    bc === :pinned && return ("fixed", "free", "fixed", "free", "fixed", "free")
    bc === :fixed && return ("fixed", "fixed", "fixed", "fixed", "fixed", "fixed")
    bc === :free && return ("free", "free", "free", "free", "free", "free")
    throw(ArgumentError("unknown support preset :$bc; use :pinned, :fixed, or :free, or a 6-element (u,u',v,v',ϕ,ϕ') tuple"))
end

"""
    constrained_dofs(model)

Global indices of the constrained (zero) degrees of freedom from the end boundary
conditions and any interior supports.
"""
function constrained_dofs(m::BeamModel)
    dofs = Int[]
    append!(dofs, fixed_local_dofs(m.bc_left))
    offset = 6 * (nnodes(m) - 1)
    append!(dofs, offset .+ fixed_local_dofs(m.bc_right))
    for (node,cond) in zip(support_nodes(m), m.interior_dofs)
        append!(dofs, 6 * (node - 1) .+ fixed_local_dofs(cond))                # u, v, ϕ
    end
    return sort(unique(dofs))
end

# ------------------------------------------------------------------
# Solution
# ------------------------------------------------------------------

"""
    BeamSolution

Result of [`solve_beam`](@ref). `d` is the full vector of nodal DOFs
(u, u', v, v', ϕ, ϕ') at each node. Evaluate fields anywhere along the beam with
[`deflections`](@ref), [`twist`](@ref), [`lateral_deflection`](@ref), and
[`vertical_deflection`](@ref).
"""
struct BeamSolution
    model::BeamModel
    d::Vector{Float64}
    residual_norm::Float64
    geometric::Bool
end

"""
    solve_beam(model; geometric=true)

Assemble K = Ke + Kg(q) and F(q) and solve the linear equilibrium system
K d = F for the free DOFs. Kg depends on the load but not on the displacements,
so no iteration is needed.

The reduced stiffness is factored by LU decomposition. An error is thrown if it
is singular -- the load is exactly at a critical level, or the supports do not
prevent rigid-body motion. Note that K is positive definite only below the first
critical load; above it K is indefinite but generally nonsingular, and a
solution is still returned (the uncoupled vertical deflection remains valid, but
the twist and lateral fields describe an unstable equilibrium).

`residual_norm` in the returned [`BeamSolution`](@ref) is the norm of the
residual K d - F equilibrated row-by-row by the stiffness diagonal (dimensionless,
insensitive to the very different scales of the translational, rotational, and
warping equations). Pass `geometric=false` to drop the geometric (load)
stiffness and recover the load-independent linear stiffness solution.
"""
function solve_beam(m::BeamModel; geometric::Bool = true)
    K = elastic_stiffness(m)
    geometric && (K .+= geometric_stiffness(m))
    F = load_vector(m)

    free = setdiff(1:ndofs(m), constrained_dofs(m))
    Kf = K[free, free]
    Ff = F[free]

    C = lu(Kf; check = false)
    issuccess(C) ||
        error("stiffness matrix K = Ke + Kg(q) is singular; the load may be at a critical " *
              "level, or the supports do not prevent rigid-body motion")
    df = C \ Ff

    scale = max.(abs.(diag(Kf)), eps())
    d = zeros(ndofs(m))
    d[free] = df
    return BeamSolution(m, d, norm((Kf * df - Ff) ./ scale), geometric)
end

# ------------------------------------------------------------------
# Field evaluation
# ------------------------------------------------------------------

function hermite(ξ::Float64, Le::Float64)
    (1 - 3ξ^2 + 2ξ^3,
     Le * (ξ - 2ξ^2 + ξ^3),
     3ξ^2 - 2ξ^3,
     Le * (ξ^3 - ξ^2))
end

function eval_field(sol::BeamSolution, z::Real, field::NTuple{4, Int})
    m = sol.model
    tol = 1e-8 * m.L
    (-tol <= z <= m.L + tol) || throw(DomainError(z, "z must be within [0, L]"))
    z = clamp(z, 0.0, m.L)  # absorb floating-point drift from a caller's cumsum-built z-mesh
    Le = elength(m)
    e = min(max(1, floor(Int, z / Le) + 1), m.nel)
    ξ = (z - (e - 1) * Le) / Le
    N = hermite(ξ, Le)
    dofs = element_dofs(e)
    return sum(N[i] * sol.d[dofs[field[i]]] for i in 1:4)
end

"Lateral deflection u(z) of the centroid (positive along +x, Fig. 1)."
lateral_deflection(sol::BeamSolution, z::Real) = eval_field(sol, z, UDOF)

"Vertical deflection v(z) of the centroid (positive along +y, i.e. downward, Fig. 1)."
vertical_deflection(sol::BeamSolution, z::Real) = eval_field(sol, z, VDOF)

"Twist ϕ(z), positive clockwise in Fig. 1."
twist(sol::BeamSolution, z::Real) = eval_field(sol, z, ΦDOF)

"""
    deflections(sol, z)

Named tuple `(u, v, ϕ)` of the centroid deflections and twist at location `z`.
"""
deflections(sol::BeamSolution, z::Real) =
    (u = lateral_deflection(sol, z), v = vertical_deflection(sol, z), ϕ = twist(sol, z))

# ------------------------------------------------------------------
# Internal forces at element ends
# ------------------------------------------------------------------

"""
    element_end_forces(sol::BeamSolution)

Internal forces, moments, torsion, and bimoment at both ends of every element,
recovered from the element deformation field by the matrix equilibrium method:
the nodal actions on element `e` are

    g = [ke + kg(q)] dₑ - fₑ

and the internal section forces follow as `-g[1:6]` at end 1 and `+g[7:12]` at
end 2, so that both ends of an element (and coincident ends of adjacent elements)
report the force system acting on the cross-section face whose outward normal is
+z. Because the nodal degrees of freedom are (u, u', v, v', ϕ, ϕ'), this is exact
at the element ends whenever the nodal solution is exact — no differentiation of
the cubic shape functions is involved.

Returns a vector with one entry per element; each entry is a named tuple

    (element, end1, end2)

where `end1`/`end2` are named tuples `(z, Vx, My, Vy, Mx, T, B)` conjugate to the
nodal DOFs:

  * `Vx`, `Vy` — shear forces along the +x and +y axes (Fig. 1: x horizontal,
    y downward), e.g. `Vy(0) = +q L/2` at the left support of a pinned-pinned
    span under gravity
  * `My = E (Iy u'' + Ixy v'')` — bending moment from lateral deflection,
    conjugate to u'
  * `Mx = E (Ix v'' + Ixy u'')` — bending moment from vertical deflection,
    conjugate to v' (negative at midspan under gravity in this convention:
    `Mx(L/2) = -q L²/8` for a pinned-pinned span)
  * `T = GJ ϕ' - ECw ϕ'''` — total torque (St. Venant + warping), conjugate to ϕ,
    positive clockwise in Fig. 1
  * `B = ECw ϕ''` — bimoment, conjugate to the warping DOF ϕ'; zero at ends that
    are free to warp (`:pinned`), nonzero at `:fixed` ends

Moments are about the paper's centroidal x-y axes (not principal axes for the
Z-section); as noted in the paper's Section 6, these section actions can be used
with capacity interaction equations for design.
"""
function element_end_forces(sol::BeamSolution)
    m = sol.model
    ke_elastic = elastic_element_stiffness(m)
    Le = elength(m)

    pack(f, z) = (z = z, Vx = f[1], My = f[2], Vy = f[3], Mx = f[4], T = f[5], B = f[6])
    return map(1:m.nel) do e
        ke = sol.geometric ? ke_elastic + geometric_element_stiffness(m, e) : ke_elastic
        g = ke * sol.d[element_dofs(e)] - element_load_vector(m, e)
        (element = e,
         end1 = pack(-g[1:6], (e - 1) * Le),
         end2 = pack(g[7:12], e * Le))
    end
end

end # module PlautBeamElement
