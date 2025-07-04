
using Distributed
addprocs(3)
@everywhere begin
    
    using Pkg
    #ENV["PYTHON"] = "/home/afonso-meneses/Desktop/GitHub/python_env/bin/python" 
    #Pkg.build("PyCall")
    using Metaheuristics
    using Metaheuristics: optimize, get_non_dominated_solutions, pareto_front, Options
    import Metaheuristics.PerformanceIndicators: hypervolume
    using HardTestProblems
    using DataStructures
    using CSV
    using DataFrames
    using Statistics
    using JSON
    using PyCall
    include(joinpath(@__DIR__, "Hyperoptimization_intervals.jl"))
    include(joinpath(@__DIR__, "example_utils.jl"))
    optuna = pyimport("optuna")
    optuna_configuration = Optimization_configuration(1,5,5) ##[LOW BOUND PROBLEM INSTANCE, HIGHER BOUND PROBLEM INSTANCE, NUM OF HP CONFIGURATIONS TO BE TESTED]
    
end

PyCall.python


basedir = pwd()
if !occursin("Results", basedir)
     results_path = joinpath(basedir, "Optuna/Results") #Change according the path 
    cd(results_path)
end




@everywhere begin
    alg = ["MOEAD_DE_searchspace","NSGA2_searchspace", "SPEA2_searchspace", "SMS_EMOA_searchspace"]
    n_trials =  optuna_configuration.max_trials
    All_Algorithm_structure = Any[]
    runs_dicts = Dict()

end
 

    
    for searchspace in alg
       
        Algorithm_structure = detect_searchspaces(searchspace)
        push!(All_Algorithm_structure, Algorithm_structure)
        df = CSV.read("minimum_runs_$(Algorithm_structure.Name).csv", DataFrame; header=false)
        runs_dicts = get_df_column_values(df, 6, 10, Algorithm_structure.Name, runs_dicts)

    end


@everywhere begin

    function run_trial(alg_instance::Int, All_Algorithm_structure, sampler_constructor, sampler_name, results_path::String, iteration_counts, problem_instance, runs_dicts)
    println("Current sampler :: $(sampler_name)")
    Algorithm_structure = All_Algorithm_structure[alg_instance]
    println("$Algorithm_structure :: $alg_instance")
    length_of_runs_array = runs_dicts[Algorithm_structure.Name]
    problem_name = HardTestProblems.NAME_OF_PROBLEMS_RW_MOP_2021[problem_instance]
    study = optuna.create_study(
        study_name = problem_name,
        direction = "maximize",
        sampler = sampler_constructor()
    )

    study.optimize(trial -> objective(trial, problem_instance, Algorithm_structure, length_of_runs_array), n_trials = n_trials)

    if isnan(study.best_value) || study.best_value == -Inf || !haskey(study.best_trial.user_attrs, "All_HV")
        println("No valid result for $problem_name")
        return nothing
    end

    All_HV = study.best_trial.user_attrs["All_HV"]
    PF_best = study.best_trial.user_attrs["PF"]

    opt_results_df = DataFrame(
        algorithm_name = Symbol[],
        sampler = String[],
        solutions = Vector{Any}[],
    )
    for (key, PF) in PF_best
    
        push!(opt_results_df, (algorithm_name = Symbol(key), sampler = study[:sampler][:__class__][:__name__], solutions = PF,))
    end

    problem_folder_name = "Problem_$(problem_instance)_$(problem_name)"
    problem_dir, iter_dir = create_directories(String(Algorithm_structure.Name), iteration_counts, problem_folder_name, results_path)
    cd(problem_dir)

    CSV_NAME = "$(Symbol(problem_name))_$(problem_instance)_$(study[:sampler][:__class__][:__name__])_$(Algorithm_structure.Name)_obtained_solutions.csv"
    CSV.write(CSV_NAME, opt_results_df)

    return (
        algorithm_name = Algorithm_structure.Name,
        sampler = study[:sampler][:__class__][:__name__],
        problem_name = problem_name,
        problem_instance = problem_instance,
        hv_value = study.best_value,
        params = study.best_params,
        #all_hv = [JSON.json(All_HV)]
    )

    end



    function run_HPO(optuna_sampler_dict,   optuna_configuration, iteration_counts, All_Algorithm_structure, results_path, runs_dicts)

        results = []
        problem_instances_array, algo_instances_array = init_parallel_arrays(optuna_configuration, All_Algorithm_structure)
        #@distributed
        #println("Currently Testing : $sampler_name")
                        
    
        #@sync @distributed
        for (sampler_name, sampler_constructor) in collect(optuna_sampler_dict)        
            
            println("Currently Testing : $sampler_name")
            task = (alg, prob) -> run_trial(alg, All_Algorithm_structure, sampler_constructor, sampler_name, results_path, iteration_counts, prob, runs_dicts)        
            pmap(task, algo_instances_array, problem_instances_array)
        end

        return results
    end



    iteration_counts = [100]
    #=
    optuna_sampler_dict = Dict(
        "NSGAIIISampler" => optuna.samplers.NSGAIIISampler,
    )
    =#
    
    optuna_sampler_dict = Dict(
    "NSGAIISampler" => optuna.samplers.NSGAIISampler,
    "CmaEsSampler" => optuna.samplers.CmaEsSampler,
    "TPESampler" => optuna.samplers.TPESampler,
    "RandomSampler" => optuna.samplers.RandomSampler,
    "QMCSampler" => optuna.samplers.QMCSampler,
    "NSGAIIISampler" => optuna.samplers.NSGAIIISampler,
    "GPSampler" => optuna.samplers.GPSampler,
    #"BruteForceSampler" => optuna.samplers.BruteForceSampler,
    "GridSampler" => () -> optuna.samplers.GridSampler(Algorithm_structure.Parameters_ranges)
    )
    
    #sampler_instances = 1:length(optuna_sampler_dict)

    function init_parallel_arrays(optuna_configuration, All_Algorithm_structure)

        problem_instances = optuna_configuration.lb_instaces:optuna_configuration.max_instace
        algo_instances = 1:length(All_Algorithm_structure)
        algo_instances_array = vcat([algo_instances for _  in problem_instances]...)    
        problem_instances_array = [prob_i for prob_i in problem_instances for _ in algo_instances]
        println("algo_instances_array ::: $algo_instances_array")
        println("problem_instances_array ::: $problem_instances_array")
    
        return problem_instances_array, algo_instances_array
    end
    #problem_instance = 1

end


    ###########
    ########### HPO
    ###########


    results = []

    @time results = run_HPO(optuna_sampler_dict, optuna_configuration, iteration_counts, All_Algorithm_structure, results_path, runs_dicts)
    # pmap -- 25.687724 seconds (2.98 M allocations: 203.145 MiB, 0.33% gc time, 8.33% compilation time: 4% of which was recompilation)
    # map -- 38.523086 seconds (38.27 M allocations: 14.290 GiB, 5.12% gc time, 25.31% compilation time: 18% of which was recompilation)
    results



    result_df = DataFrame(
        algorithm_name = Symbol[],
        sampler = String[],
        problem_instance = Int[],
        problem_name = String[],
        hv_value = Float64[],
        params = String[], 
    )

    println("Summary of best trials:")

    for r in range(1, length(results))
       
        if !isnothing(results[r]) 
            CSV_NAME = ""

            result_df = DataFrame(
                algorithm_name = Symbol[],
                sampler = String[],
                problem_instance = Int[],
                problem_name = String[],
                hv_value = Float64[],
                params = String[], 
            )


            println(results[r])

            println("Separation------------$r")
            println("$(results[r].algorithm_name) :: $(results[r].problem_name): value = $(results[r].hv_value), params = $(results[r].params)")
            if occursin("Dict{Any, Any}", string(results[r][:params]))    
                params_str = replace(string(results[r][:params]), r"Dict\{Any, Any\}\(" => "", ")" => "", "\"" => "", "=>" => "=")
            end

            println("params_str:: $params_str")

            push!(result_df, (results[r][:algorithm_name], results[r][:sampler], results[r][:problem_instance],
            results[r][:problem_name], results[r][:hv_value], params_str ))

            problem_folder_name = "Problem_$(results[r][:problem_instance])_$(results[r].problem_name)"
            problem_dir, iter_dir = create_directories(String(results[r][:algorithm_name]), iteration_counts, problem_folder_name, results_path)
            cd(iter_dir)
            
            CSV_NAME = "$(results[r][:algorithm_name])_$(results[r].sampler)_nadir.csv"
            

            write_header = !isfile(CSV_NAME)
            println(write_header)
            if occursin(string(results[r].sampler),CSV_NAME)
                println("$(results[r].sampler) HERE")
                
                CSV.write(CSV_NAME, result_df, append = true, writeheader = write_header)
            end
        end

    end 


#=
a = 1:4

b = vcat([a for _ in 1:3]...)

c = [prob_i for prob_i in 1:3 for _ in 1:4]

length(b)

=#