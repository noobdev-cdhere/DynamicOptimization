using Pkg
Pkg.add("Optim")
Pkg.add("Plots")
using Optim
using Plots
using HardTestProblems

#NelderMead(; parameters = AdaptiveParameters(), initial_simplex = AffineSimplexer())

struct MySimplexer <: Optim.Simplexer 
end

Optim.simplexer(S::MySimplexer, initial_x) = [rand(length(initial_x)) for i = 1:length(initial_x)+1]

f, conf = HardTestProblems.get_RW_MOP_problem(1)

function scalarization_f(x)
    x_clamped = clamp.(x, conf[:xmin], conf[:xmax])
    objs = f(x_clamped)
    P = sum(o[1] for o in objs)

    println(P)
    return P
        
end

run(`clear`)
initial_guess = zeros(length(conf[:xmin]))

optimize(scalarization_f, initial_guess, NelderMead(initial_simplex = MySimplexer()))


global iter_points = []

function f_logged(x)
    push!(iter_points, copy(x))
    return f(x)
end

result = optimize(f_logged, [0.0, 0.0], NelderMead(initial_simplex = MySimplexer()))

xs = [p[1] for p in iter_points]
ys = [p[2] for p in iter_points]

xrange = -2:0.05:2
yrange = -1:0.05:3
z = [f([x, y]) for y in yrange, x in xrange]  # note: reversed order for mesh

contour(xrange, yrange, z, levels=50, linewidth=0.7, title="Rosenbrock Function with Optimization Path")
plot!(xs, ys, lw=2, marker=:circle, label="Optimization Path", c=:red)
scatter!([xs[end]], [ys[end]], markersize=8, label="Final Point", c=:green)
