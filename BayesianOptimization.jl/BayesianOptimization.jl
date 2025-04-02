using BayesianOptimization, GaussianProcesses, Distributions
using Metaheuristics, HardTestProblems, Random
using Plots

function evaluate_nsga2(N, η_cr, p_cr, η_m, id)
    # Get the optimization problem
    f, conf = HardTestProblems.get_RW_MOP_problem(id)

    # Extract bounds
    xmin = conf[:xmin]
    xmax = conf[:xmax]
    bounds = [xmin xmax]

    # Convert parameters to appropriate types
    N = round(Int, N)
    η_cr = round(Int, η_cr)
    η_m = round(Int, η_m)

    # Set up NSGA2
    method = NSGA2(N=N, η_cr=η_cr, p_cr = p_cr, η_m=η_m)

    # Run NSGA2
    result = optimize(f, bounds, method)
    
    best_solution = result.best_sol 

    println("Best solution: ", best_solution)
    # Extract Pareto front solutions
    pareto_solutions = [sol for sol in result.population if sol.rank == 1]  # Keep only Pareto-optimal solutions

    # Extract objective values
    f1_values = [sol.f[1] for sol in pareto_solutions]
    f2_values = [sol.f[2] for sol in pareto_solutions]

    return f1_values, f2_values  # Return both objectives for plotting later
end


# Bounds for hyperparameters: (lower, upper)
param_bounds = [
    (0, 500),  # N
    (0, 100),  # η_cr
    (0, 1),    # p_cr, continuous between 0 and 1
    (0, 100)   # η_m
]

# Convert bounds to usable format for BayesianOptimization
lower_bounds = [b[1] for b in param_bounds]
upper_bounds = [b[2] for b in param_bounds]

# Initialize Bayesian Optimization
model = ElasticGPE(length(param_bounds), 
                   mean=MeanConst(0.), 
                   kernel=SEArd(zeros(length(param_bounds)), 5.), 
                   logNoise=0., 
                   capacity=3000)

set_priors!(model.mean, [Normal(1, 2)])

modeloptimizer = MAPGPOptimizer(every=50, 
                                noisebounds=[-4, 3], 
                                kernbounds=[[fill(-1, length(param_bounds))..., 0], 
                                            [fill(4, length(param_bounds))..., 10]], 
                                maxeval=40)

# Select a problem ID from HardTestProblems
problem_id = 1  # Change this based on the desired problem

# Store Pareto front data
pareto_f1 = []
pareto_f2 = []

opt = BOpt(
    x -> begin
        f1, f2 = evaluate_nsga2(x[1], x[2], x[3], x[4], problem_id)
        append!(pareto_f1, f1)
        append!(pareto_f2, f2)
        return minimum(f1)  # Optimize f1 in Bayesian Optimization
    end,
    model,
    UpperConfidenceBound(), 
    modeloptimizer,
    lower_bounds, upper_bounds,
    repetitions=1,
    maxiterations=10,  # Use fewer iterations for testing
    sense=Min,
    acquisitionoptions=(method=:LD_LBFGS, restarts=5, maxtime=0.1, maxeval=1000),
    verbosity=Progress
)

result = boptimize!(opt)

# 🔹 Plot the final Pareto front after Bayesian Optimization
scatter(pareto_f1, pareto_f2, xlabel="Objective 1 (f1)", ylabel="Objective 2 (f2)",
        title="Final Pareto Front", legend=false)
