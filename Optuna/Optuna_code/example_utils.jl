
include(joinpath(@__DIR__, "Hyperoptimization_intervals.jl"))
using DataStructures
using Metaheuristics
using Metaheuristics: pareto_front
using HardTestProblems

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
        nobjectives = 2   # TODO: improve this 
        npartitions = 50
        weights = gen_ref_dirs(nobjectives, npartitions)
        instance = algorithm(weights;)
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


function run_optimization(f, searchspace, 
                    reference_point, params, 
                    Algorithm_structure)

hv_values = Dict()

num_ite = 100


algorithm_instance = Algorithm_structure.Name



println("Using algorithm: $algorithm_instance")
options = Metaheuristics.Options(; iterations = num_ite)
metaheuristic = getproperty(Metaheuristics, Symbol(algorithm_instance))(; params..., options)

hv_values_mean = []
data_for_csv = []
all_pareto_fronts = []
num_runs = 5

for i in 1:num_runs
    println("Starting task...")
    status = optimize(f, searchspace, metaheuristic)
    println("Task Finished...")
    approx_front = get_non_dominated_solutions(status.population)
    push!(hv_values_mean, hypervolume(approx_front, reference_point))
    front_objectives = [sol.f for sol in approx_front]
    push!(all_pareto_fronts, front_objectives)
end


hv_values[num_ite] = mean(hv_values_mean)


println("Hypervolume: $(hv_values[num_ite])")


return hv_values, all_pareto_fronts 

end

