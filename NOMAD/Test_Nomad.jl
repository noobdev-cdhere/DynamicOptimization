using Pkg
Pkg.activate("NOMAD")
Pkg.add("CxxWrap")
using HardTestProblems
using NOMAD

f, conf = HardTestProblems.get_RW_MOP_problem(1)

pwd()

names(NOMAD)
getfield(NOMAD, :NOMAD)
fieldnames(NOMAD.NomadProblem)

# Constraints
function c(x)
  g = -(abs(cos(x[1])) + 0.1) * sin(x[1]) + 2
  ε = 0.05 + 0.05 * (1 - 1 / (1 + abs(x[1] - 11)))
  constraints = [g - ε - x[2]; x[2] - g - ε]
  return constraints
end

# Evaluator
function bb(x)
  bb_outputs = [f(x); c(x)]
  success = true
  count_eval = true
  return (success, count_eval, bb_outputs)
end

# Define Nomad Problem
p = NomadProblem(2, 3, ["OBJ"; "EB"; "EB"], bb,
                lower_bound=[0.0;0.0],
                upper_bound=[20.0;4.0])

# Fix some options
p.options.max_bb_eval = 1000 # total number of evaluations
p.options.display_stats = ["BBE", "EVAL", "SOL", "OBJ", "CONS_H"] # some display options

# Solution
result = solve(p, [0.0;2.0])
println("Solution: ", result.x_best_feas)