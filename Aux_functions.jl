module Aux_functions

    using Plots
    using Metaheuristics
    using Metaheuristics: optimize, get_non_dominated_solutions, pareto_front, Options
    import Metaheuristics.PerformanceIndicators: hypervolume
    using DataStructures

    export ref_points_offset, run_optimization, make_folder

    ref_points_offset = Dict(
        1 => [370000, -7330],
        2 => [1, 1000], 
        4 => [40, 0.2],
        7 => [4, 50],
        9 =>[3200, 0.05],
        11 =>  [93450.511, 2000, 4.853469e6, 9.620032e6, 25000.0],
        12 => [1200, 0.1],
        13 => [6500, 1300.0, 2000],
        14 => [2, 0.02],
        15 => [35, 190000],
        16 => [20, 0.0020408763],
        17 => [1000000, 1000000, 10000],
        20 =>[10000, 5e-5],
        24 => [8000, 400000, 100000000],
        27 => [1, 0],
        28 => [300, 50],
        29 => [10, 25],
        30 => [10, 30],
        31 => [10, 30],
        32 => [10, 30],
        33 => [10, 30],
        34 => [10, 30],
        35 => [5, 30],
        36 => [1000, 3000],
        37 => [1000, 3000],
        38 => [3000, 3000],
        39 => [1000, 3000, 3000],
        40 => [40, 40],
        41 => [40, 40, 40],
        42 => [40, 40],
        43 => [40, 40],
        44 => [40, 40, 60], #[5.720257, 1.0231384, 4.0471196]
        45 => [40, 40, 40],
        46 => [80, 80, 80, 80],  #[5.7381081, 2.3578333, 8.1909894, 3.9032935] 
        47 => [100, 100],
        48 => [100, 100],
        49 => [100, 100, 100],
        50 => [100000, 2000]
    )

    function make_folder()
        global path
    
        if @isdefined(path)
            return path
        end
    
        base_dir = pwd()
        folder_dir = ""
    
        try
            if isdefined(Main, :optuna) && !occursin("Optuna", base_dir)
                folder_dir = joinpath(base_dir, "Optuna")
            elseif isdefined(Main, :HyperTuning) && !occursin("HyperTuning", base_dir)
                folder_dir = joinpath(base_dir, "HyperTuning")
            else
                error("No optimization library detected!")
            end
        catch e
            println("You can't have two optimization libraries in use in the same script")
            return
        end
    
        folder_dir = joinpath(folder_dir, "result_optimized")
        if isdir(folder_dir)
            rm(folder_dir; recursive=true, force=true)
        end
        mkpath(folder_dir)
        cd(folder_dir)
    
        path = folder_dir  # Store globally
        return path
    end
    
    function create_directories(metaheuristic_str, iteration_counts, problem_folder_name, path)
        
        algorithm_dir = joinpath(path, metaheuristic_str)
        mkpath(algorithm_dir)

        iter_dir = joinpath(algorithm_dir, string(iteration_counts))
        mkpath(iter_dir)

        problem_dir = joinpath(iter_dir, problem_folder_name)
        mkpath(problem_dir)

        return problem_dir
    end


    function write_results(file_path, problem_name, current_instance, num_ite, hv, params, metaheuristic_str)
        open(file_path, "a") do file
            write(file, "$problem_name problem number $current_instance ---- $metaheuristic_str with $num_ite iterations\n")
            write(file, "Hypervolume: $hv\n")
            write(file, "$params\n\n")
        end
    end

    function run_optimization(current_inst, problem_name, f, searchspace, reference_point, metaheuristic_str, params, Algorithm_structure, path)
        
        hv_values = Dict()
        problem_folder_name = "Problem $(current_inst):  $problem_name"
        iteration_counts = [100]
        
        println(path)
        problem_dir = create_directories(metaheuristic_str, iteration_counts, problem_folder_name, path)
        
        open(joinpath(path, "reference_points.txt"), "a") do file
             write(file, "$problem_name :::: $reference_point\n")
        end


        p = plot(title="Pareto Front for $problem_name", xlabel="Objective 1", ylabel="Objective 2")
        
        i = 1
        for num_ite in iteration_counts
            i+=1 
            println("Running $metaheuristic_str with $num_ite iterations for $problem_name: problem number: $(current_inst)")

            algorithm_instance = Algorithm_structure.Name
            options = Metaheuristics.Options(; iterations = num_ite)
            metaheuristic = getproperty(Metaheuristics, Symbol(algorithm_instance))(; params..., options)

            metaheuristic_str = string(metaheuristic)
            status = optimize(f, searchspace, metaheuristic)
            approx_front = get_non_dominated_solutions(status.population)
            hv_values[num_ite] = hypervolume(approx_front, reference_point)

            println("Hypervolume so far: $(hv_values)")
            write_results(joinpath(problem_dir, "results.txt"), problem_name, current_inst, num_ite, hv_values[num_ite], params, metaheuristic_str)

            plot_pareto_front!(p, approx_front, reference_point, i, num_ite)
        end

        finalize_plot(p, reference_point, problem_dir, params)

        return hv_values
    end

    function plot_pareto_front!(p, approx_front, reference_point, color_index, num_ite)
        if isempty(approx_front)
            println("Warning: No Pareto front found for iteration $num_ite")
            return
        end

        pareto_matrix = hcat([sol.f for sol in approx_front]...)'
        if length(reference_point) == 2
            scatter!(p, pareto_matrix[:, 1], pareto_matrix[:, 2], label="Iterations: $num_ite", markersize=3, markerstrokewidth=0, color=color_index)
        elseif length(reference_point) == 3
            scatter3d!(p, pareto_matrix[:, 1], pareto_matrix[:, 2], pareto_matrix[:, 3], label="Iterations: $num_ite", markersize=3, markerstrokewidth=0, color=color_index)
        end
    end

    function finalize_plot(p, reference_point, problem_dir, params)
        if length(reference_point) == 2
            scatter!(p, [reference_point[1]], [reference_point[2]], label="Reference Point", markershape=:star, markersize=6, color=:red)
        elseif length(reference_point) == 3
            scatter3d!(p, [reference_point[1]], [reference_point[2]], [reference_point[3]], label="Reference Point", markershape=:star, markersize=6, color=:red)
        end
        savefig(joinpath(problem_dir, "pareto_front_comparison_$(params).png"))
    end

    iteration_counts = [100, 500, 1000, 2000, 3000]

    function fit_HardTestProblems_default_values_()
        
        result_path = joinpath("/home/afonso-meneses/Desktop/THESIS_ALGORITHM", "results_path")
        if !isdir(result_path)
            mkdir(result_path)
        end
        cd(result_path)

        for (index, problem_name) in enumerate(HardTestProblems.NAME_OF_PROBLEMS_RW_MOP_2021)
            try
                println("$problem_name ---- NSGA2 with different iterations")

                f, conf = HardTestProblems.get_RW_MOP_problem(index)
                bounds = hcat(conf[:xmin], conf[:xmax])

                problem_dir = joinpath(result_path, problem_name)
                mkpath(problem_dir)

                log_file_path = joinpath(problem_dir, "warnings_log.txt")
                epsilon = 0.2
                reference_point = conf[:nadir] .* (1 .+ epsilon)  
                #reference_point = [maximum(conf[:xmax]), maximum(conf[:xmax])] 
            
                p = plot(title="Pareto Front for $problem_name", xlabel="Objective 1", ylabel="Objective 2")

                for (i, num_ite) in enumerate(iteration_counts)
                    println("Running NSGA-II with $num_ite iterations for $problem_name")

                    nsga2 = NSGA2(options=Options(iterations=num_ite))
                    status = optimize(f, bounds, nsga2)
                    approx_front = get_non_dominated_solutions(status.population)

                    hv = hypervolume(approx_front, reference_point)
                    println("Hypervolume ($num_ite iterations): ", hv)
                    
                    result_file = joinpath(problem_dir, "results.txt")
                    open(result_file, "a") do file
                        write(file, "$problem_name problem number $index ---- NSGA2 with $num_ite iterations\n")
                        write(file, "Hypervolume:  $hv\n\n")
                    end

                    if hv == 0.0
                        pareto_error_file = joinpath(problem_dir, "pareto_front_$(num_ite)_hv_0.0.txt")
                        open(pareto_error_file, "a") do file
                        write(file, "approx_front:::   $approx_front \n")
                        write(file, "reference_point:::   $reference_point \n")
                        end
                    end

                    #open(log_file_path, "a") do log_file
                    #    redirect_stderr(log_file)
                    #end


                    if !isempty(approx_front)
                        pareto_values = [sol.f for sol in approx_front]  # Extract objectives
                        pareto_matrix = hcat(pareto_values...)'  # Convert to matrix format

                        scatter!(p, pareto_matrix[:, 1], pareto_matrix[:, 2], label="Iterations: $num_ite", 
                                markersize=3, markerstrokewidth=0, color=i)
                    else
                        println("Warning: No Pareto front found for $problem_name with $num_ite iterations")
                    end
                end
                scatter!(p, [reference_point[1]], [reference_point[2]], label="Reference Point", markershape=:star, markersize=6, color=:red)
                savefig(joinpath(problem_dir, "pareto_front_comparison.png"))

            catch e
                println("Error in problem $problem_name: ", e)
                continue 
            end
        end

    end

    
end