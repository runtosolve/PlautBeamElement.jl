// Z-purlin lateral-torsional deformation study.
// Numbers and figures come from generate_report_data.jl (data.typ, figures/).
// Compile from this folder:  typst compile purlin_report.typ

#import "data.typ": inputs, cases

#let za = cases.z8x2
#let zb = cases.z8x25

// ------------------------------------------------------------------
// Formatting helpers
// ------------------------------------------------------------------

// n significant figures; values with |x| < tiny print as ≈ 0
#let num(x, n: 4, tiny: none) = {
  if x == none { return [—] }
  if tiny != none and calc.abs(x) < tiny { return [$approx 0$] }
  if x == 0 { return [0] }
  let e = int(calc.floor(calc.log(calc.abs(x))))
  let d = calc.max(0, n - 1 - e)
  let r = calc.round(x, digits: d)
  if d == 0 { r = int(r) }
  [#r]
}
// fixed number of decimals
#let fx(x, d: 3) = [#calc.round(x, digits: d)]

#let plf(q) = num(q * 12000)

// ------------------------------------------------------------------
// Document setup
// ------------------------------------------------------------------

#set document(title: "Lateral-Torsional Response of Unbraced Single-Span Z-Purlins")
#set page(paper: "us-letter", margin: (x: 1in, y: 1in), numbering: "1",
          header: context if counter(page).get().first() > 1 [
            #set text(8pt, fill: luma(110))
            Unbraced single-span Z-purlins: Z8x2 and Z8x2.5, 16 ga #h(1fr) PlautBeamElement.jl
          ])
#set text(size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show heading.where(level: 1): it => { v(0.6em); it; v(0.3em) }
#show figure.where(kind: table): set figure.caption(position: top)
#set table(stroke: none, inset: (x: 6pt, y: 3.5pt), align: (x, y) => if x == 0 { left } else { center })
#show table: set text(size: 9.5pt)

#let rule-table(columns: auto, ..cells) = table(
  columns: columns,
  table.hline(stroke: 0.8pt),
  ..cells,
  table.hline(stroke: 0.8pt),
)

// ------------------------------------------------------------------
// Title
// ------------------------------------------------------------------

#align(center)[
  #text(17pt, weight: "bold")[Lateral-Torsional Response of \ Unbraced Single-Span Z-Purlins]
  #v(0.4em)
  #text(11pt)[Z8x2 16 ga (0.5 in lips) and Z8x2.5 16 ga (0.75 in lips) \
  25 ft simple span, 50 plf gravity load on the top flange]
  #v(0.4em)
  #text(9.5pt, fill: luma(90))[Analysis with PlautBeamElement.jl v0.4.0 #sym.dot.c
  #datetime.today().display("[month repr:long] [day], [year]")]
]
#v(1em)

// ==================================================================
= Introduction
// ==================================================================

This study computes the deflections, twist and internal forces of two unbraced
cold-formed steel Z-purlins (Z8x2 and Z8x2.5, 16 ga) under a 50 plf gravity load on
the top flange, and checks whether that load is below the lateral-torsional
buckling load. Because $I_(x y) != 0$ and the load is eccentric to the shear
center, a vertical load produces lateral deflection and twist as well as vertical
deflection. The finite element (FE) formulation of Plaut and Moen @plaut2021
captures these coupled effects.

// ==================================================================
= Problem Statement
// ==================================================================

== Purlins and geometry

Dimensions are from the RunToSolve purlin databases (@tab-geometry). Centerline
dimensions are $h = H - t$ and $b = B - t$; corner radii are ignored.

#figure(
  rule-table(columns: 3,
    table.header([], [*#za.label*], [*#zb.label*])
    ,table.hline(stroke: 0.5pt),
    [Database designation], [#za.db], [#zb.db],
    [Database source], text(8.5pt)[#za.source], text(8.5pt)[#zb.source],
    [Thickness $t$ (in)], [#za.t], [#zb.t],
    [Depth $H$, out-to-out (in)], [#za.H], [#zb.H],
    [Flange width $B$, out-to-out (in)], [#za.B], [#zb.B],
    [Lip length $D$ (in)], [#za.D], [#zb.D],
    [Lip angle to flange], [#za.lipangle°], [#zb.lipangle°],
    [Centerline web depth $h$ (in)], [#fx(za.h)], [#fx(zb.h)],
    [Centerline flange width $b$ (in)], [#fx(za.b)], [#fx(zb.b)],
  ),
  caption: [Purlin geometry.],
) <tab-geometry>

#figure(
  grid(columns: 2, gutter: 1em,
    image("figures/section_z8x2.svg", width: 85%),
    image("figures/section_z8x25.svg", width: 85%)),
  caption: [Centerline models. $x$ points toward the top flange tip; $y$ is
  positive downward.],
) <fig-sections>

== Span, supports and loading

#figure(
  rule-table(columns: 2,
    table.header([*Input*], [*Value*]),
    table.hline(stroke: 0.5pt),
    [Span], [$L = 25$ ft $= #inputs.L$ in, single simple span],
    [Load], [$q = #plf(inputs.q)$ plf $= #num(inputs.q)$ kip/in, gravity, uniform over the full span],
    [Load location], [Centerline of the top flange ($a_x$, $a_y$ from the shear center, @tab-props)],
    [Roof slope], [$theta = 0$ (load acts parallel to the web), so $q_x = 0$ and $q_y = q$],
    [Supports at $z = 0$ and $z = L$], [Pinned: $u = v = phi = 0$; $u'$, $v'$ and warping $phi'$ free],
    [Axial support], [Axial translation restrained at a single node at midspan],
    [Continuous bracing], [None: $k_x = 0$, $k_phi = 0$],
    [Material], [$E = #inputs.E$ ksi, $G = #inputs.G$ ksi],
    [Mesh], [#inputs.nel equal two-node elements, $L_e = #inputs.Le$ in],
  ),
  caption: [Analysis inputs (kip, in).],
) <tab-inputs>

End supports use the DOF tuple `("fixed","free","fixed","free","fixed","free")`
for $(u, u', v, v', phi, phi')$. The midspan axial restraint is not modeled
(@sec-axial).

== Section properties

Computed as in @sec-props (worked in @app-props). For a Z-section the shear
center coincides with the centroid.

#figure(
  rule-table(columns: 4,
    table.header([*Property*], [*Units*], [*#za.label*], [*#zb.label*]),
    table.hline(stroke: 0.5pt),
    [Area $A$], [in²], [#num(za.props.A)], [#num(zb.props.A)],
    [$I_x$], [in⁴], [#num(za.props.Ix)], [#num(zb.props.Ix)],
    [$I_y$], [in⁴], [#num(za.props.Iy)], [#num(zb.props.Iy)],
    [$I_(x y)$], [in⁴], [#num(za.props.Ixy)], [#num(zb.props.Ixy)],
    [$J$], [in⁴], [#num(za.props.J)], [#num(zb.props.J)],
    [$C_w$], [in⁶], [#num(za.props.Cw)], [#num(zb.props.Cw)],
    [Shear center $(x_s, y_s)$], [in], [(0, 0)], [(0, 0)],
    [$a_x$ (load, $+x$ of shear center)], [in], [#num(za.props.ax)], [#num(zb.props.ax)],
    [$a_y$ (load, above shear center)], [in], [#num(za.props.ay)], [#num(zb.props.ay)],
  ),
  caption: [Section properties about the centroidal $x$–$y$ axes (centerline model,
  sharp corners).],
) <tab-props>

// ==================================================================
= Analysis
// ==================================================================

== Section properties <sec-props>

Each section is five straight centerline segments of thickness $t$. For linear
$f$, $g$ over a segment of length $ell$ from node $i$ to $j$:

$ integral_"seg" f g dif A = (t ell) / 6 (2 f_i g_i + f_i g_j + f_j g_i + 2 f_j g_j). $ <eq-seg>

This gives $I_x$, $I_y$, $I_(x y)$; $J = sum ell t^3 slash 3$; and
$C_w = integral omega_n^2 dif A$, with $omega_n$ the normalized sectorial
coordinate about the shear center.

== Governing equations

Plaut and Moen @plaut2021:

$ E I_y u'''' + E I_(x y) v'''' + k_x (u + a_y phi) &= q_x \
  E I_x v'''' + E I_(x y) u'''' &= q_y \
  E C_w phi'''' - G J phi'' + k_x a_y (u + a_y phi) + k_phi phi &= q_x (a_y - a_x phi) + q_y (a_x + a_y phi) $ <eq-gov>

With $q_x = 0$, $q_y = q$ and $k_x = k_phi = 0$:

$ E I_y u'''' + E I_(x y) v'''' = 0, quad
  E I_x v'''' + E I_(x y) u'''' = q, quad
  E C_w phi'''' - G J phi'' - q a_y phi = q a_x. $ <eq-reduced>

Bending is coupled through $I_(x y)$; torsion is uncoupled from bending since
$k_x = 0$. $q a_x$ is the eccentric torque and $-q a_y phi$ is the load-height
(second-order) effect.

== Finite element discretization

#inputs.nel two-node elements with cubic Hermite interpolation of $u$, $v$,
$phi$; nodal DOFs $(u, u', v, v', phi, phi')$, with $xi = (z - z_1) slash L_e$:

$ bold(N) = [1 - 3 xi^2 + 2 xi^3, quad L_e (xi - 2 xi^2 + xi^3), quad 3 xi^2 - 2 xi^3, quad L_e (xi^3 - xi^2)]. $

Stiffness kernels: $integral bold(N)''^T bold(N)''$ ($E I$, $E C_w$) and
$integral bold(N)'^T bold(N)'$ ($G J$). The load-height term gives the geometric
stiffness

$ bold(k)_g = integral_0^(L_e) (q_x a_x - q_y a_y) bold(N)^T bold(N) dif z = -q a_y integral_0^(L_e) bold(N)^T bold(N) dif z, $

(softening for gravity load). Loads: $integral q bold(N)^T$ on $v$ and
$integral q a_x bold(N)^T$ on $phi$ (@app-element).

== Solution: first- and second-order analyses

$ [bold(K)_e + gamma bold(K)_g (q)] bold(d) = bold(F)(q), $ <eq-system>

solved by sparse LU factorization. First order: $gamma = 0$. Second order:
$gamma = 1$, valid only below the critical load.

== Stability check

$q_"cr"$ is the smallest $mu$ making $bold(K)_e + mu bold(K)_g (q = 1)$
singular. Closed form ($lambda_1 = pi slash L$):

$ q_"cr" = (E C_w lambda_1^4 + G J lambda_1^2) / a_y. $ <eq-qcr>

== Internal forces

Recovered at element ends by equilibrium:

$ bold(g) = [bold(k)_e + gamma bold(k)_g] bold(d)_e - bold(f)_e, quad
  "end 1:" -bold(g)_(1..6), quad "end 2:" +bold(g)_(7..12). $ <eq-recovery>

giving $V_x$, $M_y = E(I_y u'' + I_(x y) v'')$, $V_y$,
$M_x = E(I_x v'' + I_(x y) u'')$, $T = G J phi' - E C_w phi'''$ and
$B = E C_w phi''$. Sagging $M_x$ is negative ($y$ down).

== Axial restraint at midspan <sec-axial>

The formulation has no axial DOF. With no axial load, the restraint only
prevents rigid-body sliding and does not affect any result.

// ==================================================================
= Results
// ==================================================================

== Stability

#figure(
  rule-table(columns: 3,
    table.header([], [*#za.label*], [*#zb.label*]),
    table.hline(stroke: 0.5pt),
    [Applied load $q$ (plf)], [#plf(inputs.q)], [#plf(inputs.q)],
    [$q_"cr"$, closed form @eq-qcr (plf)], [#plf(za.qcr_cf)], [#plf(zb.qcr_cf)],
    [$q_"cr"$, FE eigenvalue (plf)], [#plf(za.qcr_fe)], [#plf(zb.qcr_fe)],
    [$q slash q_"cr"$], [#num(inputs.q / za.qcr_fe, n: 3)], [#num(inputs.q / zb.qcr_fe, n: 3)],
    [$bold(K)_e + bold(K)_g$ positive definite at $q$?],
      [#if za.stable [Yes] else [*No*]], [#if zb.stable [Yes] else [*No*]],
  ),
  caption: [Lateral-torsional stability of the unbraced purlins.],
) <tab-stability>

*Unbraced, neither purlin can carry 50 plf; both buckle first.* The Z8x2.5's
larger $C_w$ raises $q_"cr"$ by #calc.round((zb.qcr_fe / za.qcr_fe - 1) * 100) %.

#pagebreak(weak: true)
== Deformations

=== Z8x2 16 ga (0.5 in lips)

#figure(
  image("figures/deflections_z8x2.svg", width: 82%),
  caption: [Z8x2 16 ga (0.5 in lips): vertical deflection, lateral deflection and
  twist along the span.],
) <fig-defl-a>

#pagebreak(weak: true)
=== Z8x2.5 16 ga (0.75 in lips)

#figure(
  image("figures/deflections_z8x25.svg", width: 82%),
  caption: [Z8x2.5 16 ga (0.75 in lips): vertical deflection, lateral deflection and
  twist along the span.],
) <fig-defl-b>

=== Discussion of deformations

#figure(
  rule-table(columns: 5,
    table.header([], table.cell(colspan: 2)[*#za.label*], table.cell(colspan: 2)[*#zb.label*]),
    table.header([*Midspan*], [1st order], [2nd order], [1st order], [2nd order]),
    table.hline(stroke: 0.5pt),
    [$v$ (in)], [#num(za.first.v)], [#num(za.second.v)], [#num(zb.first.v)], [#num(zb.second.v)],
    [$u$ (in)], [#num(za.first.u)], [#num(za.second.u)], [#num(zb.first.u)], [#num(zb.second.u)],
    [$phi$ (rad)], [#num(za.first.phi)], [#num(za.second.phi)], [#num(zb.first.phi)], [#num(zb.second.phi)],
    [$L slash v$], [#calc.round(inputs.L / za.first.v)], [#calc.round(inputs.L / za.second.v)],
                   [#calc.round(inputs.L / zb.first.v)], [#calc.round(inputs.L / zb.second.v)],
  ),
  caption: [Midspan deformations.],
) <tab-defl>

$v$ and $u$ are the same in both analyses and match the closed form
(@app-deform). $u slash v = -I_(x y) slash I_y$
(#num(-za.props.Ixy / za.props.Iy, n: 3) and #num(-zb.props.Ixy / zb.props.Iy, n: 3)).
The Z8x2.5 reduces $u$ by #calc.round((1 - zb.first.u / za.first.u) * 100) % and
$v$ by #calc.round((1 - zb.first.v / za.first.v) * 100) %.

The twist is not reliable: first-order values exceed 1 rad, and the second-order
twist reverses sign because $q > q_"cr"$ (@app-deform).

#pagebreak(weak: true)
== Internal forces

=== Z8x2 16 ga (0.5 in lips)

#figure(
  image("figures/forces_z8x2.svg", width: 82%),
  caption: [Z8x2 16 ga (0.5 in lips): internal forces along the span.],
) <fig-force-a>

#pagebreak(weak: true)
=== Z8x2.5 16 ga (0.75 in lips)

#figure(
  image("figures/forces_z8x25.svg", width: 82%),
  caption: [Z8x2.5 16 ga (0.75 in lips): internal forces along the span.],
) <fig-force-b>

=== Discussion of internal forces

#figure(
  rule-table(columns: 5,
    table.header([], table.cell(colspan: 2)[*#za.label*], table.cell(colspan: 2)[*#zb.label*]),
    table.header([*Quantity*], [1st order], [2nd order], [1st order], [2nd order]),
    table.hline(stroke: 0.5pt),
    [$M_x (L slash 2)$ (kip·in)], [#num(za.first.mid.Mx, n: 5)], [#num(za.second.mid.Mx, n: 5)], [#num(zb.first.mid.Mx, n: 5)], [#num(zb.second.mid.Mx, n: 5)],
    [$V_y (0)$ (kip)], [#num(za.first.endf.Vy)], [#num(za.second.endf.Vy)], [#num(zb.first.endf.Vy)], [#num(zb.second.endf.Vy)],
    [$T (0)$ (kip·in)], [#num(za.first.endf.T)], [#num(za.second.endf.T)], [#num(zb.first.endf.T)], [#num(zb.second.endf.T)],
    [$B (L slash 2)$ (kip·in²)], [#num(za.first.mid.B)], [#num(za.second.mid.B)], [#num(zb.first.mid.B)], [#num(zb.second.mid.B)],
    [max $|M_y|$ (kip·in)], [#num(za.first.My_max, tiny: 1e-6)], [#num(za.second.My_max, tiny: 1e-6)], [#num(zb.first.My_max, tiny: 1e-6)], [#num(zb.second.My_max, tiny: 1e-6)],
    [max $|V_x|$ (kip)], [#num(za.first.Vx_max, tiny: 1e-6)], [#num(za.second.Vx_max, tiny: 1e-6)], [#num(zb.first.Vx_max, tiny: 1e-6)], [#num(zb.second.Vx_max, tiny: 1e-6)],
  ),
  caption: [Key internal forces. $M_x < 0$ is sagging in the $y$-down convention.],
) <tab-forces>

$M_x$ and $V_y$ are the simple-beam values ($-q L^2 slash 8$, $q L slash 2$) in
all cases. $M_y = V_x = 0$ because $q_x = 0$. First-order $T(0) = q a_x L slash 2$,
and $B$ peaks at midspan. Second-order $T$ and $B$ are post-critical and not
meaningful.

== Comparison and conclusions

- Unbraced, neither purlin carries 50 plf: $q_"cr"$ = #plf(za.qcr_fe) plf (Z8x2),
  #plf(zb.qcr_fe) plf (Z8x2.5).
- $v$, $u$, $M_x$ and $V_y$ are valid. The Z8x2.5 is stiffer ($v$ =
  #num(zb.first.v, n: 3) vs #num(za.first.v, n: 3) in; $u$ = #num(zb.first.u, n: 3)
  vs #num(za.first.u, n: 3) in).
- Twist, torque and bimoment are not valid magnitudes at this load.
- Deck bracing ($k_x$, $k_phi$) must be included for realistic torsional results.

#bibliography("refs.yml", title: "References")

// ==================================================================
// Appendix
// ==================================================================

#pagebreak()
#counter(heading).update(0)
#set heading(numbering: "A.1", supplement: [Appendix])

= Sample Computations

Worked for the Z8x2; checks given for both sections. Units: kip, in.

== Section properties <app-props>

#let node-table(c) = rule-table(columns: 6,
  table.header([*Node*], [*$x$ (in)*], [*$y$ (in)*], [*$X = x - x_c$*], [*$omega_C$ (in²)*], [*$omega_n$ (in²)*]),
  table.hline(stroke: 0.5pt),
  ..c.nodes.enumerate().map(((i, n)) => (
    [#(i + 1)], [#fx(n.x, d: 4)], [#fx(n.y, d: 4)], [#fx(c.omega.at(i).X, d: 4)],
    [#fx(c.omega.at(i).wc, d: 4)], [#fx(c.omega.at(i).wn, d: 4)])).flatten())

#let seg-names = ("bottom lip", "bottom flange", "web", "top flange", "top lip")
#let seg-table(c) = rule-table(columns: 7,
  table.header([*Segment*], [*$ell$ (in)*], [*$A$ (in²)*], [*$I_x$ (in⁴)*], [*$I_y$ (in⁴)*], [*$I_(x y)$ (in⁴)*], [*$J$ (in⁴)*]),
  table.hline(stroke: 0.5pt),
  ..c.segments.enumerate().map(((i, s)) => (
    [#(i + 1) #seg-names.at(i)], [#fx(s.L, d: 4)], [#fx(s.A, d: 4)], [#fx(s.Ix, d: 4)],
    [#fx(s.Iy, d: 4)], [#fx(s.Ixy, d: 4)], [#num(s.J, n: 4)])).flatten(),
  table.hline(stroke: 0.5pt),
  [*Total*], [#fx(c.segments.map(s => s.L).sum(), d: 4)], [*#fx(c.props.A, d: 4)*], [*#fx(c.props.Ix, d: 4)*],
  [*#fx(c.props.Iy, d: 4)*], [*#fx(c.props.Ixy, d: 4)*], [*#num(c.props.J, n: 4)*])

*Nodes* (@tab-nodes-a): $h = #fx(za.h)$ in, $b = #fx(za.b)$ in; lip tips at
$(plus.minus (b + D cos 55°), plus.minus (h slash 2 - D sin 55°))$.

*Area and centroid.* $A = t sum ell = #za.t times #fx(za.segments.map(s => s.L).sum(), d: 4) = #fx(za.props.A, d: 4)$ in².
$x_c = y_c = 0$.

*Moments of inertia* (@tab-seg-a):
- Web ($x = 0$, $y$ from $h slash 2$ to $-h slash 2$):
  $I_x = t h^3 slash 12 = #za.t times #fx(za.h)^3 slash 12 = #fx(za.segments.at(2).Ix, d: 4)$ in⁴.
- Top flange ($y = -h slash 2$, $x$ from 0 to $b$):
  $I_x = t b (h slash 2)^2 = #fx(za.segments.at(3).Ix, d: 4)$ in⁴,
  $I_y = t b^3 slash 3 = #fx(za.segments.at(3).Iy, d: 4)$ in⁴,
  $I_(x y) = t (b^2 slash 2)(-h slash 2) = #fx(za.segments.at(3).Ixy, d: 4)$ in⁴.

*Torsion constant.* $J = t^3 slash 3 sum ell = #za.t^3 slash 3 times #fx(za.segments.map(s => s.L).sum(), d: 4) = #num(za.props.J)$ in⁴.

#figure(seg-table(za), caption: [Z8x2: segment contributions about the centroid.]) <tab-seg-a>

*Warping constant.* Each segment adds $Delta omega_C = X_i Y_j - X_j Y_i$.
$I_(omega x) = I_(omega y) approx 0$, so the shear center is at the centroid.
Subtracting the mean gives $omega_n$ (@tab-nodes-a):

$ C_w = integral omega_n^2 dif A = sum_"seg" (t ell) / 3 (omega_i^2 + omega_i omega_j + omega_j^2) = #num(za.props.Cw) " in"^6. $

#figure(node-table(za), caption: [Z8x2: centerline nodes and sectorial coordinates.]) <tab-nodes-a>

*Load location.* $a_x = b slash 2 = #fx(za.props.ax)$ in,
$a_y = h slash 2 = #fx(za.props.ay)$ in. Z8x2.5: @tab-seg-b, @tab-nodes-b.

#figure(seg-table(zb), caption: [Z8x2.5: segment contributions about the centroid.]) <tab-seg-b>
#figure(node-table(zb), caption: [Z8x2.5: centerline nodes and sectorial coordinates.]) <tab-nodes-b>

== Element matrices and load vector <app-element>

$L_e = #inputs.L slash #inputs.nel = #inputs.Le$ in. Z8x2, $v$–$v$ block with
$E I_x = #num(za.element1.EIx, n: 5)$ kip·in²:

$ bold(k)_(v v) = (E I_x) / L_e^3 mat(12, 6 L_e, -12, 6 L_e; 6 L_e, 4 L_e^2, -6 L_e, 2 L_e^2; -12, -6 L_e, 12, -6 L_e; 6 L_e, 2 L_e^2, -6 L_e, 4 L_e^2),
  quad (12 E I_x) / L_e^3 = #num(za.element1.kvv, n: 5) "kip/in". $

Other blocks use $E I_y$, $E I_(x y)$, $E C_w$; $G J$ adds
$G J slash (30 L_e) [36, 3 L_e, -36, 3 L_e; ...]$. Consistent loads:

$ bold(f)_v = q [L_e / 2, L_e^2 / 12, L_e / 2, -L_e^2 / 12] = [#za.element1.fv.map(x => num(x)).join(", ")], $
$ bold(f)_phi = q a_x [L_e / 2, L_e^2 / 12, L_e / 2, -L_e^2 / 12] = [#za.element1.fphi.map(x => num(x)).join(", ")]. $

Geometric stiffness: $bold(k)_g = -q a_y (L_e slash 420) [156, 22 L_e, 54, -13 L_e; ...]$,
leading term #num(-inputs.q * za.props.ay * 156 * inputs.Le / 420) kip·in.

== Deformations: closed-form check <app-deform>

*Bending* ($Delta = I_x I_y - I_(x y)^2$):

$ v(L/2) = (5 q L^4) / (384 E) I_y / Delta, quad u(L/2) = -(5 q L^4) / (384 E) I_(x y) / Delta. $

Z8x2: $Delta = #fx(za.props.Ix, d: 4) times #fx(za.props.Iy, d: 4) - (#fx(za.props.Ixy, d: 4))^2 = #fx(za.props.Delta, d: 4)$ in⁸ and
$5 q L^4 slash (384 E) = #num(5 * inputs.q * calc.pow(inputs.L, 4) / (384 * inputs.E))$ in⁵, so
$v(L slash 2) = #num(za.closed.v)$ in and $u(L slash 2) = #num(za.closed.u)$ in.

*Twist.* $phi(z) = sum_(n "odd") phi_n sin(lambda_n z)$, $lambda_n = n pi slash L$:

$ phi_n = (4 q a_x slash (n pi)) / (k_n - gamma q a_y), quad k_n = E C_w lambda_n^4 + G J lambda_n^2, $ <eq-phin>

Z8x2: $q a_y = #num(inputs.q * za.props.ay)$ kip (@tab-series).

#figure(
  rule-table(columns: 5,
    table.header([*$n$*], [*$k_n$ (kip)*], [*$k_n - q a_y$ (kip)*], [*$phi_n sin(n pi slash 2)$, 1st*], [*$phi_n sin(n pi slash 2)$, 2nd*]),
    table.hline(stroke: 0.5pt),
    ..za.closed.terms.map(t => ([#t.n], [#num(t.k)], [#num(t.k - inputs.q * za.props.ay)], [#num(t.t1)], [#num(t.t2)])).flatten(),
    table.hline(stroke: 0.5pt),
    [Series sum], [], [], [#num(za.closed.phi1)], [#num(za.closed.phi2)],
    [FE], [], [], [#num(za.first.phi)], [#num(za.second.phi)],
  ),
  caption: [Z8x2: sine-series terms of the midspan twist.],
) <tab-series>

Since $q > q_"cr" = k_1 slash a_y$, $k_1 - q a_y < 0$ and the second-order
twist reverses sign.

#figure(
  rule-table(columns: 5,
    table.header([], table.cell(colspan: 2)[*#za.label*], table.cell(colspan: 2)[*#zb.label*]),
    table.header([*Midspan*], [Closed form], [FE], [Closed form], [FE]),
    table.hline(stroke: 0.5pt),
    [$v$ (in)], [#num(za.closed.v, n: 6)], [#num(za.first.v, n: 6)], [#num(zb.closed.v, n: 6)], [#num(zb.first.v, n: 6)],
    [$u$ (in)], [#num(za.closed.u, n: 6)], [#num(za.first.u, n: 6)], [#num(zb.closed.u, n: 6)], [#num(zb.first.u, n: 6)],
    [$phi$ 1st order (rad)], [#num(za.closed.phi1, n: 5)], [#num(za.first.phi, n: 5)], [#num(zb.closed.phi1, n: 5)], [#num(zb.first.phi, n: 5)],
    [$phi$ 2nd order (rad)], [#num(za.closed.phi2, n: 5)], [#num(za.second.phi, n: 5)], [#num(zb.closed.phi2, n: 5)], [#num(zb.second.phi, n: 5)],
  ),
  caption: [Deformations: closed form vs. FE.],
) <tab-check-defl>

== Internal forces: element-end recovery <app-forces>

@eq-recovery for Z8x2 element 1, first order:

#let dof-names = ("u_1", "u'_1", "v_1", "v'_1", "phi_1", "phi'_1", "u_2", "u'_2", "v_2", "v'_2", "phi_2", "phi'_2")
#let dof-units = ("in", "rad", "in", "rad", "rad", "rad/in", "in", "rad", "in", "rad", "rad", "rad/in")
#let force-names = ("V_x", "M_y", "V_y", "M_x", "T", "B", "", "", "", "", "", "")

#figure(
  rule-table(columns: 6,
    table.header([*DOF*], [*$d_e$*], [*$(bold(k)_e bold(d)_e)$*], [*$f_e$*], [*$g = k d - f$*], [*End 1 force $= -g$*]),
    table.hline(stroke: 0.5pt),
    ..range(12).map(i => (
      [#eval(dof-names.at(i), mode: "math") (#dof-units.at(i))],
      [#num(za.element1.d.at(i), tiny: 1e-12)],
      [#num(za.element1.kd.at(i), tiny: 1e-9)],
      [#num(za.element1.f.at(i), tiny: 1e-12)],
      [#num(za.element1.g.at(i), tiny: 1e-9)],
      if i < 6 {
        let gi = -za.element1.g.at(i)
        [#eval(force-names.at(i), mode: "math") #if calc.abs(gi) < 1e-9 [$approx 0$] else [$=$ #num(gi)]]
      } else [],
    )).flatten(),
  ),
  caption: [Z8x2, element 1, first order: recovery of the end forces (kip, in).],
) <tab-elem1>

Statics: $V_y (0) = q L slash 2 = #num(za.closed.Vy_end)$ kip;
$T(0) = q a_x L slash 2 = #num(za.closed.T_end1)$ kip·in; $M_x (0) = B(0) = 0$;
$M_x (L slash 2) = -q L^2 slash 8 = #num(za.closed.Mx_mid, n: 5)$ kip·in.
Torque and bimoment series:

$ T(z) = sum_(n "odd") phi_n lambda_n (G J + E C_w lambda_n^2) cos(lambda_n z), quad
  B(z) = -E C_w sum_(n "odd") phi_n lambda_n^2 sin(lambda_n z). $

#figure(
  rule-table(columns: 5,
    table.header([], table.cell(colspan: 2)[*#za.label*], table.cell(colspan: 2)[*#zb.label*]),
    table.header([*Quantity*], [Closed form], [FE], [Closed form], [FE]),
    table.hline(stroke: 0.5pt),
    [$M_x (L slash 2)$ (kip·in)], [#num(za.closed.Mx_mid, n: 6)], [#num(za.first.mid.Mx, n: 6)], [#num(zb.closed.Mx_mid, n: 6)], [#num(zb.first.mid.Mx, n: 6)],
    [$V_y (0)$ (kip)], [#num(za.closed.Vy_end, n: 6)], [#num(za.first.endf.Vy, n: 6)], [#num(zb.closed.Vy_end, n: 6)], [#num(zb.first.endf.Vy, n: 6)],
    [$T (0)$, 1st order (kip·in)], [#num(za.closed.T1, n: 5)], [#num(za.first.endf.T, n: 5)], [#num(zb.closed.T1, n: 5)], [#num(zb.first.endf.T, n: 5)],
    [$T (0)$, 2nd order (kip·in)], [#num(za.closed.T2, n: 3)], [#num(za.second.endf.T, n: 3)], [#num(zb.closed.T2, n: 3)], [#num(zb.second.endf.T, n: 3)],
    [$B (L slash 2)$, 1st order (kip·in²)], [#num(za.closed.B1, n: 5)], [#num(za.first.mid.B, n: 5)], [#num(zb.closed.B1, n: 5)], [#num(zb.first.mid.B, n: 5)],
    [$B (L slash 2)$, 2nd order (kip·in²)], [#num(za.closed.B2, n: 5)], [#num(za.second.mid.B, n: 5)], [#num(zb.closed.B2, n: 5)], [#num(zb.second.mid.B, n: 5)],
  ),
  caption: [Internal forces: closed form vs. FE.],
) <tab-check-force>

== Critical load <app-qcr>

Z8x2, $lambda_1 = pi slash #inputs.L = #num(calc.pi / inputs.L)$ in⁻¹:

$ E C_w lambda_1^4 = #num(inputs.E * za.props.Cw * calc.pow(calc.pi / inputs.L, 4)) "kip", quad
  G J lambda_1^2 = #num(inputs.G * za.props.J * calc.pow(calc.pi / inputs.L, 2)) "kip", $
$ q_"cr" = (E C_w lambda_1^4 + G J lambda_1^2) / a_y = #num(za.qcr_cf) "kip/in" = #plf(za.qcr_cf) "plf". $

Z8x2.5: $q_"cr" = #plf(zb.qcr_cf)$ plf.
