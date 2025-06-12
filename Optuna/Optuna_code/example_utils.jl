
include(joinpath(@__DIR__, "Hyperoptimization_intervals.jl"))
using DataStructures
using Metaheuristics
using Metaheuristics: pareto_front
using HardTestProblems


    ref_points_offset = Dict(
        #1 => [370000, -7330],
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
        #22 => [-9000.90871, -20000.0],
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

mutable struct Algorithm
    Name::Symbol
    Parameters::OrderedDict{Symbol, Any}
    Parameters_ranges::OrderedDict{Symbol, Any}    
end


mutable struct HyperTuning_Problems_config
    lb_instaces::Int
    max_instace::Int
    max_trials::Int
end

function getproblem(id)
    f, conf = HardTestProblems.get_RW_MOP_problem(id)
    reference_point = conf[:nadir]  
    bounds = hcat(conf[:xmin], conf[:xmax])
    if id == 25
        bounds = bounds[[2, 1], :]
    end
    return f, bounds, reference_point 
end

function remove_existing_csv(base_dir, CSV_RUNS_FILE_NAME)
    if ispath(joinpath(base_dir,"Optuna/Results/$CSV_RUNS_FILE_NAME"))
        println("It exists")
        rm(joinpath(base_dir,"Optuna/Results/$CSV_RUNS_FILE_NAME"))
    end
end

function check_CSV(searchspace; test)
    
    println("Currently using $searchspace")
    split_str = split(searchspace, "_searchspace")
    if test == true
        println("Making test CSV for $(split_str[1])")
        CSV_RUNS_FILE_NAME = "minimum_runs_test_$(split_str[1]).csv"
        remove_existing_csv(base_dir, CSV_RUNS_FILE_NAME)
        return CSV_RUNS_FILE_NAME
    else
        CSV_RUNS_FILE_NAME = "minimum_runs_$(split_str[1]).csv"
        remove_existing_csv(base_dir, CSV_RUNS_FILE_NAME)
        return CSV_RUNS_FILE_NAME
    end
end

function init_algorithm_structure(Name_algorithm::String)
    Algorithm_structure = Algorithm(:none, OrderedDict(), OrderedDict())

    Algorithm_structure.Name = Symbol(Name_algorithm)
    algorithm_instance = getfield(Metaheuristics, Symbol(Algorithm_structure.Name))
    println(algorithm_instance)
    Algorithm_structure.Parameters = get_default_kwargs(algorithm_instance)
    return Algorithm_structure
end

function get_default_kwargs(algorithm)
    if algorithm == MOEAD_DE
        weights = set_up_weights_MOEAD_DE(algorithm)
        instance = algorithm(weights;) # TODO:: Melhorar estes unecessary checks 
        return OrderedDict(fields => getfield(instance.parameters, fields) for fields in fieldnames(algorithm))
    else
        try
            instance = algorithm()
            if :parameters in fieldnames(typeof(instance))
                params = getfield(instance, :parameters)
                return OrderedDict(field => getfield(params, field) for field in fieldnames(typeof(params)))
            else
                return OrderedDict()
            end
        catch err
            println("Error creating instance: ", err)
            return OrderedDict()
        end
    end
end

function create_directories(metaheuristic_str, iteration_counts, problem_folder_name, path)
algorithm_dir = joinpath(string(path), metaheuristic_str)
mkpath(algorithm_dir)

iter_dir = joinpath(algorithm_dir, string(iteration_counts))
mkpath(iter_dir)

problem_dir = joinpath(iter_dir, problem_folder_name)
mkpath(problem_dir)

return problem_dir, iter_dir
end

function detect_searchspaces(searchspace::String)
    if occursin("_searchspace", string(searchspace))
        Algorithm_structure = init_algorithm_structure(string(split(string(searchspace), "_searchspace")[1]))
        current_searchspace = getfield(@__MODULE__, Symbol(searchspace))
        println( "Algorithm to be used for optimization::: $(Algorithm_structure.Name)")
        println("Searchspace in consideration::")
        for (key, value) in current_searchspace
            symbol = Symbol(key)
            println("Key: $(key), Value: $(value)")
            Algorithm_structure.Parameters_ranges[symbol] = [value[1], value[2], value[3]]
        end
        
        println(Algorithm_structure.Parameters_ranges)
    end
    Algorithm_structure.Parameters_ranges
    
    return Algorithm_structure
end


function set_configuration_optuna(trial, Algorithm_structure)
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


minimum_runs(z, stdev, ϵ) = 
    ceil((z * stdev / ϵ)^2)


function get_minimum_runs(All_HV, problem_name, current_instance, CSV_RUNS_FILE_NAME)
    mean_hv = mean(All_HV)
    all_std = std(All_HV)
    
    if all_std == 0
        @warn "All HV values are identical (std=0), using default runs"
        all_std = eps() # Use smallest positive float to avoid division by zero
    end
    
    CONFIDENCE_LEVELS = [
        (90, 1.645),
        (95, 1.96),
        (98, 2.33),
        (99, 2.575)
    ]
    
    function soft_penalty(scaling_factor::Float64)
        ϵ = scaling_factor * mean_hv
        upper_threshold = 40.0
        total_penalty = 0.0
        
        for (_, z) in CONFIDENCE_LEVELS
            runs = minimum_runs(z, all_std, ϵ)
            total_penalty += max(runs - upper_threshold, 0.0)
        end
        
        return total_penalty 
    end

    config = ConfigParameters()
    set_kernel!(config, "kMaternARD5")
    config.n_iterations = 50
    config.sc_type = SC_MAP  

    optimizer, optimum = bayes_optimization(
        x -> soft_penalty(x[1]), 
        [0.0], 
        [0.1], 
        config
    )
    
    optimal_scaling = optimizer[1]
    ϵ = optimal_scaling * mean_hv
    
    # Calculate all runs once with safety checks
    runs_dict = Dict{Int,Union{Int,String}}()  # Allow both Int and String values
    for (level, z) in CONFIDENCE_LEVELS
        runs = minimum_runs(z, all_std, ϵ)
        if isfinite(runs)
            runs_dict[level] = Int(floor(runs))  # Convert to Int safely
        else
            runs_dict[level] = "Inf"  # Store as string if infinite
            @warn "Infinite runs calculated for $level% confidence level"
        end
    end

    println("Optimum: $optimum")
    println("Optimizer: $optimizer")
    println("Desired margin of error (ϵ): $ϵ")
    for (level, _) in CONFIDENCE_LEVELS
        println("- $(level)% confidence interval: $(runs_dict[level])")
    end
    println()

    df = DataFrame(
        problem_name = problem_name,
        best_HV = maximum(All_HV), 
        error = ϵ/mean_hv,
        current_instance = current_instance,
        confidence_interval_90 = runs_dict[90],
        confidence_interval_95 = runs_dict[95],
        confidence_interval_98 = runs_dict[98],
        confidence_interval_99 = runs_dict[99],
    )
   
    write_header = !isfile(CSV_RUNS_FILE_NAME)
    CSV.write(CSV_RUNS_FILE_NAME, df; append=true, writeheader=write_header)
    
    All_HV_df = DataFrame(All_HV = [JSON.json(All_HV)])
    CSV.write(CSV_RUNS_FILE_NAME, All_HV_df, append=true)
    
    empty_row = DataFrame(
        problem_name = [""],
        error = [""],
        confidence_interval_90 = [""],
        confidence_interval_95 = [""],
        confidence_interval_98 = [""],
        confidence_interval_99 = [""]
    )
    CSV.write(CSV_RUNS_FILE_NAME, empty_row, append=true)

    println("min: $(minimum(All_HV))")
    println("max: $(maximum(All_HV))")
    println("std: $(all_std)")
    println("mean: $(mean_hv)")
end


function set_up_weights_MOEAD_DE(algorithm_instance)

    if algorithm_instance == MOEAD_DE || algorithm_instance == Symbol("MOEAD_DE") 
        nobjectives = 2   # TODO: improve this 
        npartitions = 50
        weights = gen_ref_dirs(nobjectives, npartitions)
        return weights
    end
     
end


function set_up_algorithm(algorithm_instance, num_ite; params = NamedTuple(), HPO=false)
    options = Metaheuristics.Options(; iterations = num_ite, time_limit = 4.0)
    weights = set_up_weights_MOEAD_DE(algorithm_instance)
    if HPO == true
        if @isdefined weights
            metaheuristic = getproperty(Metaheuristics, Symbol(algorithm_instance))(weights; params..., options)
        else
            metaheuristic = getproperty(Metaheuristics, Symbol(algorithm_instance))(; params..., options)
        end
    else
        if @isdefined weights
            metaheuristic = getproperty(Metaheuristics, Symbol(algorithm_instance))(weights; options)
        else
            metaheuristic = getproperty(Metaheuristics, Symbol(algorithm_instance))(; options)
        end
    end
    return metaheuristic
    
end


function unravel_PF(PF::Vector{Metaheuristics.xFgh_solution{Vector{Float64}}}) # So far is only usable for length(PF) == 1 
    println("PF:: $(typeof(PF))")
    data = []

    for (idx, sol) in enumerate(PF)
        push!(data, sol.f)  
    end

    return data
end


function run_optimization(f, searchspace, 
                    reference_point, params, 
                    Algorithm_structure, problem_name)

hv_values = Dict()

num_ite = 100
 

algorithm_instance = Algorithm_structure.Name

println("Using algorithm: $algorithm_instance")
metaheuristic = set_up_algorithm(algorithm_instance, num_ite; params, HPO = true)

result_dir = "/home/afonso-meneses/Desktop/GitHub/DynamicOptimization/Optuna/Results"

if pwd() !== result_dir
    cd(result_dir)
end
All_HV = Float64[]

all_pareto_fronts = []
num_runs = csv_data

for i in 1:num_runs
    println("Starting task...")
    status = optimize(f, searchspace, metaheuristic)
    println("Task Finished...")
    approx_front = get_non_dominated_solutions(status.population)
    push!(All_HV, hypervolume(approx_front, reference_point))
    #front_objectives get_non_dominated_solutions= [sol.f for sol in approx_front]
    #push!(all_pareto_fronts, front_objectives)
end

hv_values[num_ite] = mean(All_HV)


println("Hypervolume: $(hv_values[num_ite])")


return hv_values, All_HV

end

