

include(joinpath(@__DIR__, "Hyperoptimization_intervals.jl"))
using DataStructures
using Metaheuristics
using Metaheuristics: pareto_front
using HardTestProblems
using DataFrames



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
        24 => [8000, 400000, 100000000], #--
        27 => [1, 0],
        28 => [300, 50],
        29 => [10, 25],
        30 => [10, 30],
        31 => [10, 30],
        32 => [10, 30],
        33 => [10, 30], # -- 
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


mutable struct Optimization_configuration
    lb_instaces::Int
    max_instace::Int
    max_trials::Int
end


function getproblem(id::Int)
    f, conf = HardTestProblems.get_RW_MOP_problem(id)
    reference_point = conf[:nadir]  
    bounds = hcat(conf[:xmin], conf[:xmax])
    if id == 25
        bounds = bounds[[2, 1], :]
    end
    return f, bounds, reference_point 
end

function init_algorithm_structure(Name_algorithm::String)
    Algorithm_structure = Algorithm(:none, OrderedDict(), OrderedDict())
    Algorithm_structure.Name = Symbol(Name_algorithm)
    algorithm_instance = getfield(Metaheuristics, Symbol(Algorithm_structure.Name))
    Algorithm_structure.Parameters = get_default_kwargs(algorithm_instance)
    return Algorithm_structure
end

function get_default_kwargs(algorithm)
    if algorithm == MOEAD_DE
        weights = set_up_weights_MOEAD_DE()
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
            length(value) > 2 ? arr_range = [value[1], value[2], value[3]] : arr_range = [value[1], value[2]]
            Algorithm_structure.Parameters_ranges[symbol] = arr_range
        end
        
        println(Algorithm_structure.Parameters_ranges)
    end
    Algorithm_structure.Parameters_ranges
    
    return Algorithm_structure
end


function set_configuration_optuna(trial, Algorithm_structure, sampler_constructor,reference_point)
    params = Dict()
    MOEAD_HP = Dict() 
    weights = nothing

    for (hyperparam, range_vals) in Algorithm_structure.Parameters_ranges
        
        lb, hb = range_vals[1:2]

        if hyperparam == :npartitions
            MOEAD_HP[hyperparam] = trial.suggest_int(hyperparam, lb, hb)
        else
            param_type = typeof(Algorithm_structure.Parameters[hyperparam])
            params[hyperparam] = if param_type == Float64
                if !(sampler_constructor == optuna.samplers.BruteForceSampler)            
                    trial.suggest_float(hyperparam, lb, hb)
                else
                    step = range_vals[3]   
                    trial.suggest_float(hyperparam, lb, hb, step = step)
                end
            elseif param_type == Int64
                trial.suggest_int(hyperparam, lb, hb)
            elseif param_type == Bool
                trial.suggest_categorical(hyperparam, ["false", "true"])
            else
                error("Unsupported parameter type: $param_type")
            end
        end
    end
    if !isempty(MOEAD_HP)
        weights = set_up_weights_MOEAD_DE(length(reference_point), MOEAD_HP[:npartitions])
        
    end
    return params, weights
end

function get_df_column_values(df::DataFrame, column::Int, array_position::Int, Alg_Name, run_dict)
    
    if 1 ≤ column ≤ length(names(df))
        collumn_array = []

        for i in df[2:end, column] 
            if !ismissing(i)
                str = String(i)
                if !occursin("Inf", str)
                    array = eval(Meta.parse(str)) 
                    push!(collumn_array, array[array_position])
                    
                else
                    push!(collumn_array, i)
                end
            end
        end
    else
        error("Invalid column index: $column")
    end
    run_dict[Alg_Name] = collumn_array
    return run_dict

end

function set_up_weights_MOEAD_DE(nobjectives = nothing , npartitions = nothing)
    if  isnothing(nobjectives) && isnothing(npartitions)
        nobjectives = 2  
        npartitions = 50
    end
    weights = gen_ref_dirs(nobjectives, npartitions)
    println("Weights generated")
    return weights
   
end

function set_up_algorithm(algorithm_instance, num_ite; params=Dict(), HPO=false, CCMO=false, MOEAD_WEIGHTS=nothing)
    options = Metaheuristics.Options(; iterations=num_ite, time_limit=4.0)
    base_algo = getproperty(Metaheuristics, Symbol(algorithm_instance))
    
    if CCMO
        base_algo = CCMO(base_algo)
    end

    if algorithm_instance == :MOEAD_DE
        if MOEAD_WEIGHTS === nothing
            MOEAD_WEIGHTS = set_up_weights_MOEAD_DE()
        end
        
        if HPO
            
            T = max(3, round(Int, 0.2*length(MOEAD_WEIGHTS)))
            n_r = max(2, round(Int, 0.05*length(MOEAD_WEIGHTS)))

            println("T :: $T")
            println("n_r :: $n_r")

            metaheuristic = base_algo(MOEAD_WEIGHTS; params..., T, n_r, options)

        else
            metaheuristic = base_algo(MOEAD_WEIGHTS; options)
        end
    else
        kwargs = HPO ? (; params..., options) : (; options)
        metaheuristic = base_algo(; kwargs...)
    end

    return metaheuristic
end

function run_optimization(f, searchspace, 
                    reference_point, params,
                    Algorithm_structure, current_instance, 
                    problem_name, length_of_runs_array; MOEAD_WEIGHTS = nothing)

    hv_values = Dict()
    All_HV = Float64[]
    all_pareto_fronts = Dict()
    num_ite = 100

    algorithm_instance = Algorithm_structure.Name

    println("Using algorithm: $algorithm_instance")
    metaheuristic = set_up_algorithm(algorithm_instance, num_ite; params, HPO = true, MOEAD_WEIGHTS)#, CCMO = true) ### CCMO parameter

    result_dir = "/home/afonso-meneses/Desktop/GitHub/DynamicOptimization/Optuna/Results"

    if pwd() !== result_dir
        cd(result_dir)
    end
 

    num_runs = length_of_runs_array[current_instance]
    println(num_runs)
    if length_of_runs_array[current_instance] == "Inf" || length_of_runs_array[current_instance] > 100
        println("Skipping instance $current_instance due to invalid run length.")
        return -Inf, -Inf, -Inf
    end

    println("$(Algorithm_structure.Name) :: $(current_instance)")
    for i in 1:num_runs
        println("Starting task...")
        status = optimize(f, searchspace, metaheuristic)
        println("Task Finished...")
        approx_front = get_non_dominated_solutions(status.population)
        push!(All_HV, hypervolume(approx_front, reference_point))
        front_objectives = get_non_dominated_solutions([sol.f for sol in approx_front])   
        field = "Run_$(i)_$(problem_name)"
        all_pareto_fronts[Symbol(field)] = front_objectives
    end

    #println(all_pareto_fronts)

    hv_values[num_ite] = mean(All_HV)


    println("Hypervolume: $(hv_values[num_ite])")


    return hv_values, All_HV, all_pareto_fronts

end

function objective(trial, current_instance, sampler_constructor,Algorithm_structure, length_of_runs_array)
    

    problem_name = HardTestProblems.NAME_OF_PROBLEMS_RW_MOP_2021[current_instance]
    println("Optimizing problem: ", problem_name)
    f, searchspace, reference_point = getproblem(current_instance)  
    println(searchspace)

    if haskey(ref_points_offset, current_instance)
            reference_point = ref_points_offset[current_instance]
    end
    
    params, weights = set_configuration_optuna(trial, Algorithm_structure, sampler_constructor, reference_point)
    
    hv_value, All_HV, all_pareto_fronts = run_optimization(f, searchspace, reference_point, params, 
                                                            Algorithm_structure, current_instance, 
                                                            problem_name, length_of_runs_array; MOEAD_WEIGHTS = weights)

                                                            
    if hv_value == -Inf && All_HV == -Inf
        return -Inf
    end

    isempty(hv_value) && return -Inf

    hv_max = maximum(values(hv_value))

    trial.set_user_attr("problem_name", problem_name)
    trial.set_user_attr("PF", all_pareto_fronts)
    trial.set_user_attr("All_HV", All_HV)

    return hv_max
end
