# Installations ----------------------------------------------------------------
#=
using Pkg

Pkg.add(url="https://github.com/aptmcl/KhepriFrame3DD.jl")
Pkg.add(url="https://github.com/ines-pereira/Metaheuristics.jl")
Pkg.add("Surrogates")
Pkg.add("Plots")
=#

using Pkg
Pkg.activate("Study_Frame3dd")
Pkg.status()
# Truss Geometry ---------------------------------------------------------------
using KhepriFrame3DD

truss_nodes(ps) =
  map(truss_node, ps)

truss_bars(ps, qs) =
  map(truss_bar, ps, qs)

function fixedpoint(f, x0; residualnorm = (x -> norm(x,Inf)), tol = 1E-10, maxiter=100)
  residual = Inf
  iter = 1
  xold = x0
  while residual > tol && iter < maxiter
    xnew = f(xold)
    residual = residualnorm(xold - xnew)
    xold = xnew
    iter += 1
  end
  return xold
end

catenary_points(p, h, r, ϕ, n) =
  let a = -fixedpoint(a->r/acosh(h/a+1), 1.0)
    map_division(ρ -> add_cyl(p, ρ, ϕ, h + a*(cosh(ρ/a) - 1)),
                 0, r, n)
  end

truss_rib(as, bs) =
  let fixed_truss_node_family =
        truss_node_family_element(default_truss_node_family(),
                                  support=truss_node_support(ux=true, uy=true, uz=true)),
      c = intermediate_loc(as[end], bs[end]),
      as = as[1:end-1],
      bs = bs[1:end-1]
    truss_nodes(as)
    truss_nodes(bs)
    truss_node(c, fixed_truss_node_family)
    truss_bars([c,c], [as[end], bs[end]])
    truss_bars(as, bs)
    truss_bars(as[1:end-1], as[2:end])
    truss_bars(bs[1:end-1], bs[2:end])
    truss_bars(as[1:2:end], bs[2:2:end])
    truss_bars(bs[2:2:end], as[3:2:end])
  end

dome_truss_ribs(p, h, Δ, qs, n) =
  let
    for q in qs
      v = q - p
      ρ = pol_rho(v)
      ϕ = pol_phi(v)
      truss_rib(catenary_points(p, h, ρ, ϕ, n),
                catenary_points(p, h - Δ, ρ - Δ, ϕ, n))
    end
  end

dome_truss_ribs(p, h, Δ, qs, rs, n) =
  let
    for (q, r) in zip(qs, rs)
      v = q - p
      ρ = pol_rho(v)
      ϕ = pol_phi(v)
      with_truss_node_family(radius=r*2.4) do
        with_truss_bar_family(radius=r, inner_radius=4/5*r) do
          truss_rib(catenary_points(p, h, ρ, ϕ, n),
                    catenary_points(p, h - Δ, ρ - Δ, ϕ, n))
        end
      end
    end
  end

random_frontier(p, rmin, rmax, n) =
  map_division(ϕ->add_pol(p, random_range(rmin, rmax), ϕ), 0, 2π, n, false)

random_seed(12345)
frontier = random_frontier(u0(), 10, 14, 20)

# Multi-Objective Optimization -------------------------------------------------
using Metaheuristics
using Surrogates 
using Random

## Helper Functions ------------------------------------------------------------
int2float(x, min, step=0.01) = 
  min + step*x

## Initial Population ----------------------------------------------------------
create_initial_population(bounds, pop_size) =
  let n = size(bounds, 2),
      lb = bounds[1, :],
      ub = bounds[2, :]
    Random.seed!(12345)
    samples = Surrogates.sample(pop_size, lb, ub, LatinHypercubeSample())
    int_samples = n==1 ? [(floor.(Int, s), ) for s in samples] : [floor.(Int, s) for s in samples]
    vcat(permutedims.(collect.(int_samples))...)
  end

## Materials Young's Modulus and Cost -------------------------------------
materials_e = [164090000000.0, # Cast Iron ASTM A536
               180000000000.0, # Steel, stainless AISI 302
               200000000000.0, # Carbon Steel, Structural ASTM A516
               206840000000.0, # Alloy Steel, ASTM A242
               204770000000.0, # Alloy Steel, AISI 4140
               193000000000.0, # Stainless Steel AISI 201
              ]

materials_cost = [460.00, # Cast Iron ASTM A536
                  940.00, # Steel, stainless AISI 302
                  650.00, # Carbon Steel, Structural ASTM A516
                  750.00, # Alloy Steel, ASTM A242
                  1750.00, # Alloy Steel, AISI 4140
                  1225.00, # Stainless Steel AISI 201
                 ]

## Objectives ------------------------------------------------------------------
#=
The objectives for the optimization are:
  (1) minimizing the maximum displacement,
  (2) minimizing the material cost of the truss structure.
=#
#n_objs = 2

displacement(d) =
  let d_max_aceitavel = 0.2,
      peso_d_max = 0.1
    d > d_max_aceitavel ? d - d_max_aceitavel : peso_d_max*(d - d_max_aceitavel)
  end

cost(truss_volume, material) =
    truss_volume * materials_cost[Int(material)]

objectives(material, x_center, y_center, rs...) =
    let rs = int2float.(rs, 0.05),
        x_c = int2float(x_center, -3),
        y_c = int2float(y_center, -3),
        load = vxz(1e3, -1e4)
      set_backend_family(default_truss_bar_family(),
           frame3dd,
           frame3dd_truss_bar_family(
             E=materials_e[Int(material)], # (Young's modulus)
             G=81000000000.0, # (Kirchoff's or Shear modulus)
             p=0.0, # Roll angle.
             d=77010.0)) # Density
        KhepriFrame3DD.with(current_backend, frame3dd) do
            delete_all_shapes()
            dome_truss_ribs(xy(x_c, y_c), 9, 1, frontier, rs, 10)
            truss_volume = truss_bars_volume()
            d = max_displacement(truss_analysis(load))
            [displacement(d), cost(truss_volume, material)]
        end
    end

objectives(2, 5, 5, [random_range(0, 10) for p in frontier]...)

problem(x) = 
  (objectives(x[1], x[2], x[3], x[4:end]...), [0.0], [0.0])

## Variables -------------------------------------------------------------------
#n_vars = 2
material = 1:6
x_center = 0:600 # min= -3, max=3, step=0.01
y_center = 0:600 # min= -3, max=3, step=0.01
rs = [0:10 for p in frontier]

rs_lb = [rs[1][1] for p in frontier]
rs_ub = [rs[1][end] for p in frontier]

vars_bounds = [material[1] x_center[1] y_center[1] rs_lb...;
               material[end] x_center[end] y_center[end] rs_ub...]

## Optimization Setup | Global Variables ---------------------------------------
#n_runs = 1
max_evals = 600
n_iterations = 25
pop_size = ceil(Int, 600/25)
pop_ini = create_initial_population(vars_bounds, pop_size)

#problem(pop_ini[:, 1])

# Run Optimization -------------------------------------------------------------
options = Options(iterations=n_iterations)
spea2 = SPEA2(N=pop_size, options=options)
set_user_solutions!(spea2, pop_ini, problem)

result = optimize(problem, vars_bounds, spea2)

## Results Visualization--------------------------------------------------------
using Plots

final_pop = fvals(result)
pf = pareto_front(result)

scatter(final_pop[:, 2], final_pop[:, 1],
        label="Final Population", color=:gray,
        markersize=3, markerstrokewidth=0,
        title="Results for SPEA2 Optimization", 
        xlabel="Material Cost", ylabel="Maximum Displacement")

scatter!(pf[:,2], pf[:, 1], 
        label="Non-Dominated Solutions", color=:orange,
        markersize=4, markerstrokewidth=1)
