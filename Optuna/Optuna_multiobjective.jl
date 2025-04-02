using Pkg
Pkg.activate(@__DIR__)
include("/home/afonso-meneses/Desktop/THESIS_ALGORITHM/Hyperoptimization_intervals.jl")
using .Hyperoptimization_intervals
include("/home/afonso-meneses/Desktop/THESIS_ALGORITHM/utils.jl")
using .utils 
include("/home/afonso-meneses/Desktop/THESIS_ALGORITHM/HyperTuning/Aux_func_Hypertuning.jl")
using .Aux_func_Hypertuning
using PyCall
using Metaheuristics
using Metaheuristics: TestProblems, optimize, SPEA2, get_non_dominated_solutions, pareto_front, Options
import Metaheuristics.PerformanceIndicators: hypervolume
using HardTestProblems
using DataStructures


#=
Grid Search implemented in GridSampler
Random Search implemented in RandomSampler
Tree-structured Parzen Estimator algorithm implemented in TPESampler
CMA-ES based algorithm implemented in CmaEsSampler
Gaussian process-based algorithm implemented in GPSampler
Algorithm to enable partial fixed parameters implemented in PartialFixedSampler
Nondominated Sorting Genetic Algorithm II implemented in NSGAIISampler
A Quasi Monte Carlo sampling algorithm implemented in QMCSampler
=#

# Import Optuna
optuna = pyimport("optuna")

# Create an Optuna study
study = optuna.create_study(sampler=optuna.samplers.RandomSampler())

# List available samplers
function get_samplers(optuna)
    Samplers_list = []

    for samplers in keys(optuna[:samplers])
        samplers_txt = string(samplers)
        if occursin("Sampler", samplers_txt)
            push!(Samplers_list, samplers)
        end
    end
    return Samplers_list
end

Samplers_list = get_samplers(optuna)

# Retrieve problem instance
function getproblem(id)
    f, conf = HardTestProblems.get_RW_MOP_problem(id)
    reference_point = conf[:nadir]  
    bounds = hcat(conf[:xmin], conf[:xmax])
    return f, bounds, reference_point 
end


# Detect search spaces
last_index = 1
Algorithm_structure = utils.Algorithm(:none, OrderedDict(), OrderedDict())
Algorithm_structure, last_index = utils.detect_searchspaces(last_index)

# Correct function to set Optuna configuration
function set_configuration_optuna(trial)
    params = Dict()

    for (hyperparam, range_vals) in Algorithm_structure.Parameters_ranges
        lb, hb = range_vals[1:2]  # Extract lower and upper bounds
        num_elements = length(range_vals) > 2 ? Int(range_vals[3]) : nothing
        param_type = typeof(Algorithm_structure.Parameters[hyperparam])

        params[hyperparam] = if param_type == Float64
            trial.suggest_float(hyperparam, lb, hb)
        elseif param_type == Int64
            trial.suggest_int(hyperparam, lb, hb)
        elseif param_type == Bool
            trial.suggest_categorical(hyperparam, ["false", "true"])
        else
            error("Unsupported parameter type: $param_type")
        end
    end
    return params
end


run(`clear`)


HyperTuning_configuration = utils.HyperTuning_Problems_config(1,50,100)

path = Aux_func_Hypertuning.make_folder()

function objective(trial, current_instance)
    params = set_configuration_optuna(trial)
    try
        problem_name = HardTestProblems.NAME_OF_PROBLEMS_RW_MOP_2021[current_instance]
        println("Optimizing problem: ", problem_name)

        f, searchspace, reference_point = try
            getproblem(current_instance)  
        catch e
            @warn "Error retrieving problem for trial $trial: $e"
            return -Inf
        end

        if haskey(Aux_func_Hypertuning.ref_points_offset, current_instance)
            reference_point = Aux_func_Hypertuning.ref_points_offset[current_instance]
        end

        open(joinpath(path, "reference_points.txt"), "a") do file
            write(file, "$problem_name :::: $reference_point\n")
        end

        hv_values = Aux_func_Hypertuning.run_optimization(
            current_instance, problem_name, f, searchspace, reference_point,
            string(Algorithm_structure.Name), params, Algorithm_structure, path)
        println(@isdefined path)
        println("HV Values: ", hv_values)

        return isempty(hv_values) ? -Inf : -maximum(values(hv_values))
    catch e
        @warn "Unexpected error in objective function: $e"
        return -Inf
    end
end

run(`clear`)

db_path = joinpath(path, "db.sqlite3")

for current_instance in HyperTuning_configuration.lb_instaces:HyperTuning_configuration.max_instace


    study = optuna.create_study( storage="sqlite:///$db_path",study_name="$(HardTestProblems.NAME_OF_PROBLEMS_RW_MOP_2021[current_instance])", sampler=optuna.samplers.RandomSampler())

    study.optimize(trial -> objective(trial, current_instance), n_trials=100)

    #study.optimize(objective, n_trials=100)
    print("Best value: $(study.best_value) (params: $(study.best_params))")
end


print(study.best_trial.value, study.best_trial.params)
