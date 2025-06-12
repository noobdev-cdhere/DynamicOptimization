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
using Statistics
using Distances
using BayesOpt
using JSON


run(`clear`)

base_dir = pwd()

alg =["MOEAD_DE_searchspace"]


function return_results(approx_front, reference_point, All_dist, All_HV)
    if length(approx_front) == 1
        data = unravel_PF(approx_front) 
        d = euclidean(data[1], reference_point)
        println("Distance:: $d")
        push!(All_dist, d)
        println("Returning euclidian Distance")
        return All_dist
    else
        push!(All_HV, hypervolume(approx_front, reference_point))
        println("Returning Hypervolume")

        #front_objectives = [sol.f for sol in approx_front]
        #push!(all_pareto_fronts, front_objectives)
        return All_HV
    end

end




for searchspace in alg

    CSV_RUNS_FILE_NAME = check_CSV(searchspace; test = true)


    Algorithm_structure = detect_searchspaces(searchspace)

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

    global All_HV = Float64[]

    optuna = pyimport("optuna")

    HyperTuning_configuration4 = HyperTuning_Problems_config(1,50,100)

    Algorithm_structure
    

    function objective(trial, current_instance)
        
        params = set_configuration_optuna(trial, Algorithm_structure)
        problem_name = HardTestProblems.NAME_OF_PROBLEMS_RW_MOP_2021[current_instance]
        println("Optimizing problem: ", problem_name)
        f, searchspace, reference_point = getproblem(current_instance)  
        println(searchspace)

        hv_value, All_HV = run_optimization(f, searchspace, reference_point, params, Algorithm_structure, problem_name)
        

        #println("data_for_csv:: $(typeof(data_for_csv))")
        isempty(hv_value) && return -Inf

        hv_max = maximum(values(hv_value))

        #trial.set_user_attr("problem_name", problem_name)
        #trial.set_user_attr("PF", data_for_csv)
        trial.set_user_attr("All_HV", All_HV)

        return hv_max
    end

    iteration_counts = [100] #preguntar á Ines como modificar isto 

    results_path = joinpath(splitdir(@__DIR__)[1], "Results")

    db_path = joinpath(results_path, "db_$(string(Algorithm_structure.Name))_$(string(iteration_counts[1])).sqlite3")

    results = []

    run(`clear`)

    #########

    ######### GET RUNS WITH DEFAULT PARAMS

    ############
    for current_instance in HyperTuning_configuration4.lb_instaces:HyperTuning_configuration4.max_instace
        
        if  current_instance == 11
            hv_values = Dict()
            num_ite = 100
            problem_name = HardTestProblems.NAME_OF_PROBLEMS_RW_MOP_2021[current_instance]
            println("Optimizing problem: ", problem_name)
            f, searchspace, reference_point = getproblem(current_instance) 
            
            if haskey(ref_points_offset, current_instance)
                reference_point = ref_points_offset[current_instance]
            end

            algorithm_instance = Algorithm_structure.Name

            println("Using algorithm: $algorithm_instance")
            metaheuristic = set_up_algorithm(algorithm_instance, num_ite)

            result_dir = "/home/afonso-meneses/Desktop/GitHub/DynamicOptimization/Optuna/Results"

            if pwd() !== result_dir
                cd(result_dir)
            end

            All_HV = Float64[]
            All_dist = Float64[]
            num_runs = 100

            for i in 1:num_runs
                println("Starting task...")
                status = optimize(f, searchspace, metaheuristic)
                println("Task Finished...")
                approx_front = get_non_dominated_solutions(status.population)
                results = return_results(approx_front, reference_point, All_dist, All_HV)
            end

            hv_values[num_ite] = mean(results)


            println("Hypervolume: $(hv_values[num_ite])")
            
            get_minimum_runs(results, problem_name, current_instance, CSV_RUNS_FILE_NAME)
        end
    end
end



###########
########### HPO
###########
#=
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
        sampler = optuna.samplers.TPESampler(seed = 80)
        #RandomSampler(seed = 42)
    )

    study.optimize(trial -> objective(trial, current_instance), n_trials=100)
    
    println("[$problem_name] Best value: $(study.best_value) (params: $(study.best_params))")
    

    All_HV = study.best_trial.user_attrs["All_HV"]

    
    #PF_best = study.best_trial.user_attrs["PF"]
    
    problem_folder_name = "Problem_$(current_instance)_$(problem_name)"
    
    problem_dir, iter_dir = create_directories(String(Algorithm_structure.Name), iteration_counts, problem_folder_name, results_path)
    cd(problem_dir)

    #= 
    for row in eachrow(PF_best)
        
        push!(opt_results_df, (algorithm_name = Symbol(problem_name), sampler = study[:sampler][:__class__][:__name__],
         solutions = row,))
    end

    cd(problem_dir)
    println("Writing pareto front....")
    CSV_NAME = "$(Symbol(problem_name))_$(current_instance)_$(study[:sampler][:__class__][:__name__])_obtained_solutions.csv"
    CSV.write(CSV_NAME, opt_results_df)
    println("Pareto front was sucessefully writen")
    =#
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
 =#
