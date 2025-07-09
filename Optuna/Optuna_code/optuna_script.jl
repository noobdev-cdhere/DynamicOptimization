
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
    include(joinpath(@__DIR__, "optuna_utils.jl"))
    optuna = pyimport("optuna")
end  

PyCall.python 

run(`clear`)
result_dir = normpath(@__DIR__,"..","Results")
cd(result_dir)

@everywhere begin
    alg = ["MOEAD_DE_searchspace","NSGA2_searchspace", "SPEA2_searchspace", "SMS_EMOA_searchspace"]
    n_trials = 2
    All_Algorithm_structure = initialize_algorithm_structures(alg)
    iteration_counts = 100
end


@everywhere begin

    function run_trial(sampler_instance::Int, Algorithm_structure, sampler_vector, result_dir::String, iteration_counts, problem_instance, runs_dicts)
       
        length_of_runs_array = runs_dicts[Algorithm_structure.Name]
        problem_name = HardTestProblems.NAME_OF_PROBLEMS_RW_MOP_2021[problem_instance]
        sampler_name = sampler_vector[sampler_instance]
        println("sampler_name :: $sampler_name")
        println("Problem being tested ", problem_name)
        sampler_module = optuna.samplers
        sampler_func = getproperty(sampler_module, Symbol(sampler_name))

        if sampler_name == "GridSampler"
            sampler_constructor = sampler_func(Algorithm_structure.Parameters_ranges)
        else
            sampler_constructor = sampler_func()
        end
        
        println("sampler_func  " ,sampler_func)
        study = optuna.create_study(
            study_name = problem_name,
            direction = "maximize",
            sampler = sampler_constructor
        )
        


        study.optimize(trial -> objective(trial, problem_instance, sampler_func, Algorithm_structure, length_of_runs_array, result_dir), n_trials = n_trials)

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
        println()
        problem_dir, iter_dir = create_directories(String(Algorithm_structure.Name), iteration_counts, problem_folder_name, result_dir)
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



    function run_HPO(sampler_vector, iteration_counts, result_dir, All_Algorithm_structure)
        
        results = []
        runs_dicts = initialize_runs_dicts(All_Algorithm_structure)
        problem_instances_array, sampler_instances_array = init_parallel_arrays(;sampler_vector, lb_instaces = 1, hb_instaces = 2) 
        results = @distributed (vcat) for Algorithm_structure in collect(All_Algorithm_structure)
            println("Currently Testing : $(Algorithm_structure.Name)")
            task = (sampler_instance, prob) -> run_trial(sampler_instance, Algorithm_structure, sampler_vector, result_dir, iteration_counts, prob, runs_dicts)
            pmap_results = pmap(task, sampler_instances_array, problem_instances_array)
            [pmap_results]
        end
   

        return results
    end



    
    optuna_sampler_dict = Dict(
    "NSGAIISampler" => optuna.samplers.NSGAIISampler,
    "CmaEsSampler" => optuna.samplers.CmaEsSampler,
    "TPESampler" => optuna.samplers.TPESampler,
    "RandomSampler" => optuna.samplers.RandomSampler,
    "QMCSampler" => optuna.samplers.QMCSampler,
    "NSGAIIISampler" => optuna.samplers.NSGAIIISampler,
    "GPSampler" => optuna.samplers.GPSampler,
    "BruteForceSampler" => optuna.samplers.BruteForceSampler,
    "GridSampler" => () -> optuna.samplers.GridSampler(Algorithm_structure.Parameters_ranges)
    )


    sampler_vector =  collect(keys(optuna_sampler_dict))



end

   
    ###########
    ########### HPO
    ###########

    results = []
    run(`clear`)
    @time results = run_HPO(sampler_vector, iteration_counts,result_dir, All_Algorithm_structure)
    # pmap -- 25.687724 seconds (2.98 M allocations: 203.145 MiB, 0.33% gc time, 8.33% compilation time: 4% of which was recompilation)
    # map -- 38.523086 seconds (38.27 M allocations: 14.290 GiB, 5.12% gc time, 25.31% compilation time: 18% of which was recompilation)

   
    #println("Summary of best trials:")
   
   function write_data_into_csv(results)
        temp = results
        
        for i in range(1, length(results))
            temp = results[i]
            for r in range(1, length(temp))
            
                if !isnothing(temp[r]) 
                    CSV_NAME = ""

                    result_df = DataFrame(
                        algorithm_name = Symbol[],
                        sampler = String[],
                        problem_instance = Int[],
                        problem_name = String[],
                        hv_value = Float64[],
                        params = String[], 
                    )


            
                    println("$(temp[r].algorithm_name) :: $(temp[r].problem_name): value = $(temp[r].hv_value), params = $(temp[r].params)")
                    if occursin("Dict{Any, Any}", string(temp[r][:params]))    
                        params_str = replace(string(temp[r][:params]), r"Dict\{Any, Any\}\(" => "", ")" => "", "\"" => "", "=>" => "=")
                    end

                    println("params_str:: $params_str")

                    push!(result_df, (temp[r][:algorithm_name], temp[r][:sampler], temp[r][:problem_instance],
                    temp[r][:problem_name], temp[r][:hv_value], params_str ))

                    problem_folder_name = "Problem_$(temp[r][:problem_instance])_$(temp[r].problem_name)"
                    problem_dir, iter_dir = create_directories(String(temp[r][:algorithm_name]), iteration_counts, problem_folder_name, result_dir)
                    cd(iter_dir)
                    
                    CSV_NAME = "$(temp[r][:algorithm_name])_$(temp[r].sampler).csv"
                    

                    write_header = !isfile(CSV_NAME)

                    if occursin(string(temp[r].sampler),CSV_NAME)
                        CSV.write(CSV_NAME, result_df, append = true, writeheader = write_header)
                    end
                end

            end 
        end
    end


write_data_into_csv(results)

#=
a = 1:4

b = vcat([a for _ in 1:3]...)

c = [prob_i for prob_i in 1:3 for _ in 1:4]

length(b)

=#