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

main_script_name = basename(abspath(@__FILE__))


run(`clear`)

base_dir = pwd()

### For CCMO add prefix and check flag in run_optimization

alg = ["SMS_EMOA_searchspace"]



for searchspace in alg

    CSV_RUNS_FILE_NAME, CSV_LENGTH_RESULTS_NAME = check_CSV(searchspace, main_script_name; test = false)

    #if occursin("CCMO_", searchspace)
    #   searchspace = String(searchspace[2])
    #    println("Searchspace is $(searchspace[2])")
    #end
    Algorithm_structure = detect_searchspaces(searchspace)

    HyperTuning_configuration4 = HyperTuning_Problems_config(1,50,100)

    iteration_counts = [100] #preguntar á Ines como modificar isto 

    results_path = joinpath(splitdir(@__DIR__)[1], "Results")

    db_path = joinpath(results_path, "db_$(string(Algorithm_structure.Name))_$(string(iteration_counts[1])).sqlite3")

    results = []

    run(`clear`)

    #########

    ######### GET RUNS WITH DEFAULT PARAMS

    ############

    count = 0
    for current_instance in HyperTuning_configuration4.lb_instaces:HyperTuning_configuration4.max_instace
        
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

            global All_HV = Dict(:Hypervolumes => Float64[])
            global All_dist = Dict(:Distances => Float64[])

            num_runs = 100

            for i in 1:num_runs
                println("Starting task...")
                status = optimize(f, searchspace, metaheuristic)
                println("Task Finished...")
                approx_front = get_non_dominated_solutions(status.population)
                All_HV = return_results(approx_front, reference_point)
            end


            df = DataFrame(
                problem_name = problem_name,
                current_instance = current_instance,
                #Pareto_front_length = length(approx_front),
                length_HV = length(All_HV[:Hypervolumes])
            )

            if count == 0
                CSV.write(CSV_LENGTH_RESULTS_NAME, df; append=true, writeheader = true )
                count = 1
            else
                CSV.write(CSV_LENGTH_RESULTS_NAME, df; append=true)
            end
            #println(All_HV)
            #diff_leng = length(All_dist[:Distances]) - length(All_HV[:Hypervolumes])
            
            #results = diff_leng > 50 ? All_dist : All_HV
            results = All_HV
            type_of_result = first(keys(results))                

            
            #println(diff_leng)

            #=
            for i in All_HV[:Hypervolumes]
            
                normalized_HV = (i - minimum(All_HV[:Hypervolumes])) / (maximum(All_HV[:Hypervolumes]) - minimum(All_HV[:Hypervolumes]))
                println("Hypervolume :: $i :: $normalized_HV")
                push!(results, normalized_HV)
            end

            for i in All_dist[:Distances]
                
                normalized_D = 1 - ((i - minimum(All_dist[:Distances])) / (maximum(All_dist[:Distances]) - minimum(All_dist[:Distances])))
                println("Distances :: $i :: $normalized_D")
                push!(results, normalized_D)
            end
            =#

            println("Results::$results")

            hv_values[num_ite] = mean(results[type_of_result])


            println("Hypervolume: $(hv_values[num_ite])")
            
            get_minimum_runs(results, problem_name, current_instance, CSV_RUNS_FILE_NAME)
    end
end
