using Pkg
Pkg.activate(".")
Pkg.add("Optim")
Pkg.add("Plots")
Pkg.add("HardTestProblems") 
using Optim, Plots, HardTestProblems

# Custom simplexer
struct MySimplexer <: Optim.Simplexer end
Optim.simplexer(S::MySimplexer, initial_x) = [rand(length(initial_x)) for _ in 1:length(initial_x)+1]

# Load CEC 2020 problem (single-objective by default?)
f, conf = HardTestProblems.get_cec2020_problem(1, n=2)

# Scalarization function (adjust if f(x) is multi-objective)
function scalarization_f(x)
    x_clamped = clamp.(x, conf[:xmin], conf[:xmax])
    P = f(x_clamped)  # Assumes f returns a scalar
    println("Current value: ", P)
    return P
end

initial_guess = zeros(length(conf[:xmin]))
result = optimize(scalarization_f, initial_guess, NelderMead(initial_simplex=MySimplexer()))

# Logging optimization path
global iter_points = []
function f_logged(x)
    push!(iter_points, copy(x))
    return scalarization_f(x)  # Reuse scalarization
end

result_logged = optimize(f_logged, initial_guess, NelderMead(initial_simplex=MySimplexer()))

# Plotting (only works for 2D problems!)
if length(initial_guess) == 2
    xs = [p[1] for p in iter_points]
    ys = [p[2] for p in iter_points]
    
    xrange = range(conf[:xmin][1], conf[:xmax][1], length=100)
    yrange = range(conf[:xmin][2], conf[:xmax][2], length=100)
    z = [f([x, y]) for y in yrange, x in xrange]  # f must return a scalar
    
    contour(xrange, yrange, z, levels=50, title="Optimization Path")
    plot!(xs, ys, label="Path", color=:red, marker=:circle)
    scatter!([xs[end]], [ys[end]], label="Final Point", color=:green)
else
    println("Plotting only supported for 2D problems.")
end