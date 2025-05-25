using NLopt

function objective(x::Vector, grad::Vector)
    return sum(x.^2)  # Minimize x₁² + x₂² + ...
end

n = 3
opt = Opt(:GN_DIRECT, n)


fieldnames(Opt)

# Standard NLopt parameters
opt.lower_bounds = [-1.0, -1.0, -1.0]
opt.upper_bounds = [1.0, 1.0, 1.0]
opt.xtol_rel = 1e-4
opt.maxeval = 500

# DIRECT-specific hyperparameters
opt.params["fglper"] = 0.1
opt.params["volper"] = 0.05

opt.min_objective = objective
(minf, minx, ret) = optimize(opt, rand(n))

println("Minimum: ", minf, " at x = ", minx)