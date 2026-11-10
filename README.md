# PlautBeamElement.jl

A beam finite element for the lateral-torsional deformation of continuously braced
thin-walled beams, implementing the governing equations of:

> Plaut, R.H., Moen, C.D. (2021). "Lateral-torsional deformations of single-span and
> two-span thin-walled beams with continuous bracing." *Journal of Constructional
> Steel Research*, 179, 106534. https://doi.org/10.1016/j.jcsr.2021.106534

## Formulation

The coupled equilibrium equations (2)–(4) of the paper, extended with the classical
lateral-torsional buckling couplings between the first-order moments and the twist
(the `Mx`, `My` terms),

```
EIy u'''' + EIxy v'''' + kx (u + ay ϕ) - (Mx ϕ)''                 = qx
EIx v'''' + EIxy u''''                 + (My ϕ)''                 = qy
ECw ϕ'''' - GJ ϕ'' + kx ay (u + ay ϕ) + kϕ ϕ - Mx u'' + My v''    = qx (ay - ax ϕ) + qy (ax + ay ϕ)
```

`Mx = E(Ix v'' + Ixy u'')` and `My = E(Iy u'' + Ixy v'')` follow this package's sign
convention (y downward, so sagging `Mx` is negative). Twist turns part of `Mx` into a
weak-axis moment (twist creates lateral bending), and lateral curvature turns part of
`Mx` into a torque (bending creates more twist). These terms are on by default since
v0.5.0; `BeamModel(...; moment_coupling = false)` recovers the paper's Eqs. (2)–(4)
exactly (load-height term only).

are discretized with a two-node element using cubic Hermite interpolation for the
lateral deflection `u(z)`, vertical deflection `v(z)`, and twist `ϕ(z)`. Each node
carries six degrees of freedom — translations `u, v`, rotations `u', v'`, twist `ϕ`,
and warping `ϕ'`.

* **Elastic stiffness** `elastic_stiffness(model)`: bending (`EIy`, `EIx`, and the
  `EIxy` coupling for point-symmetric sections such as Z's), warping (`ECw`) and
  St. Venant (`GJ`) torsion, and the continuous bracing terms `kx`, `kϕ`.
* **Geometric (load) stiffness** `geometric_stiffness(model)`, symmetric:
  * load height — the twist-dependent load-position terms on the right-hand side of
    Eq. (4), `(qx ax - qy ay) ϕ`: softening for gravity load applied above the shear
    center, stiffening for uplift;
  * moment coupling — the u–ϕ and v–ϕ blocks `-∫ Mx N''ᵀN dz` and `∫ My N''ᵀN dz`
    (and transposes), from the second-order energy `∫ (-Mx ϕ u'' + My ϕ v'') dz`.
    `Mx`, `My` are the first-order (pre-buckling) moments from a linear solve with
    `Ke` alone (`prebuckling_moments(model)`), so they are exact for continuous
    spans and fixed ends and include the bracing reactions.

  u and v are taken at the shear center, which is the centroid for I- and
  Z-sections. The Wagner term `½ βx Mx ϕ'²` is not included (it vanishes for those
  sections and for a C-section bending about its axis of symmetry).
* **Solution**: `Kg` depends on the load `q` but not on the displacements, so for a
  given load `[Ke + Kg(q)] d = F(q)` is a linear system. `K` is assembled in sparse
  storage (`sparse_stiffness(model)`) and solved with one sparse LU factorization, so
  memory and run time grow linearly with the mesh (an error is raised if `K` is
  singular).
  The load–deformation curves are nonlinear in `q` only because `Kg` scales with `q`.
* **Critical load** `critical_load(model)`: since `Kg` is proportional to `q`, the
  lateral-torsional buckling load is `λ q`, with `λ` the smallest positive factor
  making `Ke + λ Kg(q)` singular (one symmetric eigenvalue problem).

Boundary conditions follow the paper: `:pinned` (`u = v = ϕ = 0`, free to warp) and
`:fixed` (`u = u' = v = v' = ϕ = ϕ' = 0`). Interior supports (`u = v = ϕ = 0`) model
two-span beams; by symmetry each span of a symmetric two-span beam behaves as a
pinned-fixed single span (verified in the tests).

## Usage

Consistent N–mm units: `E`, `G`, `kx` in N/mm², `I`, `J` in mm⁴, `Cw` in mm⁶,
`kϕ` in N/rad, `q` in N/mm (= kN/m), `θ` in rad.

```julia
using PlautBeamElement: Section, Material, Restraint, BeamModel, solve_beam,
                              twist, lateral_deflection, deflections, element_end_forces,
                              critical_load

# Example 1 of the paper: 9CS2.5×059 C-section stud, L = 7620 mm
section   = Section(Ix = 4.287e6, Iy = 290500.0, J = 424.56,
                    Cw = 3.1956e9, ax = 57.67, ay = 114.3)   # Ixy = 0 default
material  = Material(E = 200e3)                              # G = E/2.6 default
restraint = Restraint(kx = 0.1, kϕ = 1000.0)

model = BeamModel(L = 7620.0, nel = 40,
                  section = section, material = material, restraint = restraint,
                  q = 1.0,                # gravity; q < 0 for uplift;
                                          # q = (q1, q2) for a trapezoidal load
                                          # varying linearly from z = 0 to z = L
                  θ = 0.0,                # roof slope angle (Eq. 1, Fig. 2)
                  bc_left = :pinned, bc_right = :pinned)

sol = solve_beam(model)

twist(sol, 3810.0)                 # ϕ(L/2) ≈ 0.058 rad
lateral_deflection(sol, 3810.0)    # u(L/2) ≈ -6.1 mm
deflections(sol, 3810.0)           # (u = ..., v = ..., ϕ = ...)
critical_load(model)               # q_cr ≈ 4.22 N/mm (load factor for q = 1)
```

With `moment_coupling = false` the same model reproduces the paper's Fig. 7
(`ϕ(L/2) ≈ 0.072` rad, `u(L/2) ≈ -8.5` mm, `q_cr ≈ 9.15` N/mm).

Internal forces, moments, torsion, and bimoment at the element ends are recovered
from the element deformation field by the matrix equilibrium method,
`g = [ke + kg(q)] dₑ - fₑ`, which is exact at element ends whenever the nodal
solution is exact (no differentiation of the shape functions):

```julia
forces = element_end_forces(sol)   # one entry per element

forces[1].end1        # (z, Vx, My, Vy, Mx, T, B) at the left support:
forces[1].end1.Vy     #   vertical shear  = +qL/2
forces[20].end2.Mx    #   midspan moment  = -qL²/8 for the uncoupled v field
forces[1].end1.T      #   torque (St. Venant + warping), GJϕ' - ECwϕ'''
forces[1].end1.B      #   bimoment ECwϕ'' (zero at ends free to warp)
```

All actions are reported on the cross-section face with outward normal +z,
conjugate to the DOFs (u, u', v, v', ϕ, ϕ'): shears `Vx`, `Vy`; bending moments
`My = E(Iy u'' + Ixy v'')` and `Mx = E(Ix v'' + Ixy u'')` about the centroidal
x-y axes; torque `T`; bimoment `B`.

With moment coupling, `element_end_forces(sol)` reports actions about the fixed
(undeformed) axes, in exact nodal equilibrium; `element_end_forces(sol; axes =
:section)` reports them about the twisted section's own axes (e.g. a weak-axis
moment `My ≈ Mx ϕ` from the rotated strong-axis moment), which is what stress and
capacity checks need.

A two-span beam with a central support:

```julia
two_span = BeamModel(L = 2 * 7620.0, nel = 80, section = section,
                     material = material, restraint = restraint,
                     q = 1.0, interior_supports = [7620.0])
```

## Verification

`test/runtests.jl` verifies the element against the paper's two examples
(C-section 9CS2.5×059 and Z-section 8ZS2.25×059). The paper-reference checks run with
`moment_coupling = false`, since they test Eqs. (2)–(4) as published:

* exact Fourier sine-series solutions of Eqs. (2)–(4) for pinned-pinned beams,
  covering gravity/uplift, all restraint combinations, and roof slope;
* classical bending benchmarks for the uncoupled vertical deflection;
* Fig. 7 twist magnitudes; Fig. 3 boundary-condition ordering (pinned-pinned >
  pinned-fixed > fixed-fixed) and the pinned-fixed peak near `z = 0.42 L`;
* the wavy Z-section twist shape of Fig. 14 and the deflection sign conclusions of
  Section 6;
* two-span ≡ pinned-fixed span equivalence;
* gravity softening / uplift stiffening from the geometric stiffness, and the
  critical load `q_cr = (ECw λ₁⁴ + GJ λ₁²)/ay` of the unbraced C-section;
* element end forces: bending statics (`qL/2`, `qL²/8`, `qL²/12`, propped-cantilever
  reactions `3qL/8`/`5qL/8`), continuity of all six actions across element
  boundaries, torque/moment/bimoment against exact series solutions, bimoment-free
  pinned ends vs. warping-restrained fixed ends, and global torque balance
  `T(0) - T(L) = ∫ q (ax + ay ϕ) dz`;
* moment coupling: an independent sine-series Galerkin solution of the coupled
  equations (unbraced and braced Z, C-section, including the bracing-induced `My`),
  the classical `C1 ≈ 1.13` critical moment of a doubly symmetric I-beam under
  uniform load, pre-buckling moments (simple span and the `qL²/8` hogging moment of a
  two-span beam), symmetry and load-proportionality of `Kg`, end-force equilibrium,
  and the Z8x2.5 Abaqus shell model of the 2026 validation study (`q_cr` 10.55 vs
  11.2 lb/ft; the coupling sign moves u, v, ϕ toward the shell results, the opposite
  sign away from them);
* trapezoidal loads: exact reduction to the uniform case for equal end intensities,
  triangular-load statics (`wL/6`/`wL/3` reactions, `wL²/16` midspan moment, peak
  `wL²/(9√3)` at `z = L/√3`), superposition without the geometric stiffness, torque
  balance, and a spectral (sine-basis) Galerkin reference solution that captures the
  mode coupling produced by the linearly varying geometric-stiffness coefficient;
* mesh convergence.
