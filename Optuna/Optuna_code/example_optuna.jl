#using Pkg
#Pkg.activate("Test_optuna")
include(joinpath(@__DIR__, "Hyperoptimization_intervals.jl"))
include(joinpath(@__DIR__, "example_utils.jl"))
using PyCall
using Metaheuristics
using Metaheuristics: TestProblems, optimize, SPEA2, get_non_dominated_solutions, pareto_front, Options
import Metaheuristics.PerformanceIndicators: hypervolume
using HardTestProblems
using DataStructures
using CSV
using DataFrames


run(`clear`)

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

#/usr/bin/python3
#ENV["PYTHON"] = "C:\\ProgramData\\Anaconda3\\python.exe"
#Pkg.build("PyCall")


Algorithm_structure = detect_searchspaces("NSGA2_searchspace")

optuna = pyimport("optuna")

HyperTuning_configuration4 = HyperTuning_Problems_config(1,2,100)

Algorithm_structure

function objective(trial, current_instance)
    
    params = set_configuration_optuna(trial, Algorithm_structure)
    problem_name = HardTestProblems.NAME_OF_PROBLEMS_RW_MOP_2021[current_instance]
    println("Optimizing problem: ", problem_name)
    f, searchspace, reference_point = getproblem(current_instance)  

    hv_values, PF = run_optimization(f, searchspace, reference_point, params, Algorithm_structure)
    
    PF = [sol.f for sol in PF]  

    pf_matrix = hcat(PF...)'

    isempty(hv_values) && return -Inf

    hv_max = maximum(values(hv_values))

    trial.set_user_attr("problem_name", problem_name)
    trial.set_user_attr("PF", pf_matrix)
    trial.set_user_attr("hv_max", hv_max)

    return hv_max
end

iteration_counts = [100]

PyCall.python

results_path = joinpath(splitdir(@__DIR__)[1], "Results")

db_path = joinpath(results_path, "db_$(string(Algorithm_structure.Name))_$(string(iteration_counts[1])).sqlite3")

results = []

run(`clear`)

for current_instance in HyperTuning_configuration4.lb_instaces:HyperTuning_configuration4.max_instace


    opt_results_df = DataFrame(
        algorithm_name = Symbol[],
        sampler = String[],
        solutions = Vector{Any}[],
    )

    problem_name = HardTestProblems.NAME_OF_PROBLEMS_RW_MOP_2021[current_instance]
    study = optuna.create_study(
        #storage = "db_path",
        study_name = problem_name,
        direction = "maximize",
        sampler = optuna.samplers.TPESampler(seed = 42)
        #RandomSampler(seed = 42)
    )

    initial_time = time()

    study.optimize(trial -> objective(trial, current_instance), n_trials=5)
    
    final_time = time() - initial_time


    println("[$problem_name] Best value: $(study.best_value) (params: $(study.best_params))")
    
    PF_best = study.best_trial.user_attrs["PF"]

    problem_folder_name = "Problem_$(current_instance)_$(problem_name)"
    
    problem_dir, iter_dir = create_directories(String(Algorithm_structure.Name), iteration_counts, problem_folder_name, results_path)
    cd(problem_dir)

     
    for row in eachrow(PF_best)
        push!(opt_results_df, (algorithm_name = Symbol(problem_name), sampler = study[:sampler][:__class__][:__name__],
         solutions = row,))
    end


    cd(problem_dir)
    println("Writing pareto front....")
    CSV_NAME = "$(Symbol(problem_name))_$(current_instance)_$(study[:sampler][:__class__][:__name__])_obtained_solutions.csv"
    CSV.write(CSV_NAME, opt_results_df)
    println("Pareto front was sucessefully writen")

    push!(results, (
        algorithm_name = Algorithm_structure.Name,
        sampler = study[:sampler][:__class__][:__name__],
        problem_name = problem_name,
        current_instance = current_instance,
        hv_value = study.best_value,
        params = study.best_params, 
        
    ))
    
end


result_df = DataFrame(
    algorithm_name = Symbol[],
    sampler = String[],
    current_instance = Int[],
    problem_name = String[],
    hv_value = Float64[],
    params = String[], 
)

result_df



println("\n📊 Summary of best trials:")


length(results)




for r in range(1, length(results))
    
    CSV_NAME = ""
    println(r)
    println("🧪 $(results[r].algorithm_name) :: $(results[r].problem_name): value = $(results[r].hv_value), params = $(results[r].params)")
    
    if occursin("Dict{Any, Any}", string(results[r][:params]))    
        params_str = replace(string(results[r][:params]), r"Dict\{Any, Any\}\(" => "", ")" => "", "\"" => "", "=>" => "=")
    end

    println("params_str:: $params_str")

    typeof(results[r][:current_instance])
    push!(result_df, (results[r][:algorithm_name], results[r][:sampler], results[r][:current_instance],
     results[r][:problem_name], results[r][:hv_value], params_str ))

    problem_folder_name = "Problem_$(results[r][:current_instance])_$(results[r].problem_name)"
    problem_dir, iter_dir = create_directories(String(results[r][:algorithm_name]), iteration_counts, problem_folder_name, results_path)
    cd(iter_dir)


    CSV_NAME = "$(Algorithm_structure.Name)_$(results[r].sampler)_nadir.csv"
    CSV.write(CSV_NAME, result_df)


end 
 