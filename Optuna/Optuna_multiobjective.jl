using Pkg
Pkg.activate(@__DIR__)
include("/home/afonso-meneses/Desktop/GitHub/DynamicOptimization/Hyperoptimization_intervals.jl")
using .Hyperoptimization_intervals
include("/home/afonso-meneses/Desktop/GitHub/DynamicOptimization/utils.jl")
using .utils 
include("/home/afonso-meneses/Desktop/GitHub/DynamicOptimization/Aux_functions.jl")
using .Aux_functions
using PyCall
using Metaheuristics
using Metaheuristics: TestProblems, optimize, SPEA2, get_non_dominated_solutions, pareto_front, Options
import Metaheuristics.PerformanceIndicators: hypervolume
using HardTestProblems
using DataStructures
using Surrogates
using CSV
using DataFrames


run(`clear`)

#base = pwd()
#base_dir =joinpath(base, "Optuna")
#base_dir



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

optuna = pyimport("optuna");

# Retrieve problem instance
function getproblem(id)
    f, conf = HardTestProblems.get_RW_MOP_problem(id)
    reference_point = conf[:nadir]  
    bounds = hcat(conf[:xmin], conf[:xmax])
    return f, bounds, reference_point 
end;

#methods(HardTestProblems.process_synthesis)



# Detect search spaces
last_index = 1
Algorithm_structure = utils.Algorithm(:none, OrderedDict(), OrderedDict())
Algorithm_structure, last_index = utils.detect_searchspaces(last_index)

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

path = Aux_functions.make_folder()

iteration_counts = [100]

function objective(trial, current_instance)
    params = set_configuration_optuna(trial)
    try
        problem_name = HardTestProblems.NAME_OF_PROBLEMS_RW_MOP_2021[current_instance]
        println("Optimizing problem: ", problem_name)
        
        #Surrogate_flag = true
        
        f, searchspace, reference_point = try
            getproblem(current_instance)  
        catch e
            @warn "Error retrieving problem for trial $trial: $e"
            return -Inf
        end
        #f = apply_surrogate(f, bounds)
        #if haskey(Aux_functions.ref_points_offset, current_instance)
        #    reference_point = Aux_functions.ref_points_offset[current_instance]
        #end

        #if Surrogate_flag === true
        #    f_surrogate = apply_surrogate(f, searchspace, RadialBasis)
        #else
        #    f_surrogate = f
        #end

        #open(joinpath(path, "reference_points.txt"), "a") do file
        #    write(file, "$problem_name :::: $reference_point\n")
        #end

        
        hv_values = Aux_functions.run_optimization(
            current_instance, iteration_counts, problem_name, f, searchspace, reference_point,
            string(Algorithm_structure.Name), params, Algorithm_structure, path)
        
        println("HV Values: ", hv_values)

        return isempty(hv_values) ? -Inf : maximum(values(hv_values))
    catch e
        @warn "Unexpected error in objective function: $e"
        return -Inf
    end
end

run(`clear`)

print(path)

db_path = joinpath(path, "db_$(string(Algorithm_structure.Name))_$(string(iteration_counts[1])).sqlite3")

results = []

#rm(joinpath(pwd(), "$(db_path)"), recursive=true)

current_instance = 1

for current_instance in HyperTuning_configuration.lb_instaces:HyperTuning_configuration.max_instace
    name = HardTestProblems.NAME_OF_PROBLEMS_RW_MOP_2021[current_instance]

    study = optuna.create_study(
        storage="sqlite:///$db_path",
        study_name=name,
        direction="maximize", 
        sampler=optuna.samplers.RandomSampler()
    )

    try
    study.optimize(trial -> objective(trial, current_instance), n_trials=100)
    catch e
        @warn "Optuna crashed during optimization: $e"
        continue
    end



    println("[$name] Best value: $(study.best_value) (params: $(study.best_params))")
    
    push!(results, (
        algorithm_name = Algorithm_structure.Name,
        sampler = study[:sampler][:__class__][:__name__], 
        name = name,
        hv_value = study.best_value,
        params = study.best_params
    ))
end


result_df = DataFrame(
    algorithm_name = Symbol[],
    name = String[],
    hv_value = Float64[],
    params = String[] 
)

result_df

run(`clear`)

println("\n📊 Summary of best trials:")

length(results)

for r in range(1, length(results))
    println(r)
    println("🧪 $(results[r].algorithm_name) :: $(results[r].name): value = $(results[r].hv_value), params = $(results[r].params)")
    if occursin("Dict{Any, Any}", string(results[r][:params]))
        
        params_str = replace(string(results[r][:params]), r"Dict\{Any, Any\}\(" => "", ")" => "", "\"" => "")
    end
    push!(result_df, (results[r][:algorithm_name],results[r][:name], results[r][:hv_value], params_str))
    CSV_NAME = "$(Algorithm_structure.Name)_$(results[r].sampler)_nadir.csv"
    CSV.write(CSV_NAME, result_df)


end 
 