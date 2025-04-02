using Pkg
Pkg.activate("/home/afonso-meneses/Desktop/THESIS_ALGORITHM/HyperTuning")
import Metaheuristics
using Metaheuristics
using Metaheuristics: TestProblems, optimize, SPEA2, get_non_dominated_solutions, pareto_front, Options
include("/home/afonso-meneses/Desktop/THESIS_ALGORITHM/utils.jl")
using .utils 
using DataStructures
using HyperTuning
import HardTestProblems

dirpath = "/home/afonso-meneses/Desktop/THESIS_ALGORITHM/HyperTuning/"
fname = "Hyperparameters_single_optimization.txt"


mutable struct HyperTuning_Problems_config
    lb_instaces::Int
    max_instace::Int
    max_trials::Int
end




HyperTuning_configuration = HyperTuning_Problems_config(1,10,10)


if isfile(joinpath(dirpath, fname)) !== true 

    for name_alg in utils.single_opt
        Algorithm_structure = utils.init_algorithm_structure(name_alg)
        utils.txt_path = utils.create_file(Algorithm_structure, dirpath, fname)
    end
    print(utils.txt_path)
else
    txt_path = joinpath(dirpath, fname)
    print(txt_path)
end


last_position = 0
#for i in utils.single_opt
Algorithm_structure = utils.Algorithm(:none, OrderedDict(), OrderedDict())
last_position, Algorithm_structure = utils.create_intervals(Algorithm_structure, txt_path, last_position)
println("$(Algorithm_structure.Name)---- $Algorithm_structure.Parameters_ranges")


#end

Algorithm_structure.Parameters_ranges

function getproblem(id)
    
    #=
    HardTestProblems:
        RW-MOP-2021 Real world multi-objective Constrained Optimization Test Suite.
        CEC2020-BC-SO Bound-constrained test problems for single-objective optimization.
        PMM Pseudo-feasible solutions in evolutionary bilevel optimization: Test problems and performance assessment
        SMD Scalable test problems for single-objective bilevel optimization.
        CEC2017 Competition on Constrained Real-Parameter Optimization.

    =#

    f, conf = HardTestProblems.get_cec2020_problem(id)
    fmin = conf[:minimum]

    return f, [conf[:xmin] conf[:xmax]], fmin
end




function objective(trial)
    # Define a global variable to store trial (not ideal)
    global last_trial = trial  

    params = (;trial.values...)
    
    # Get instance information
    f, searchspace, fmin = getproblem(get_instance(trial))
    println(fmin)

    # Get provided seed for RNG
    seed = get_seed(trial)

    # Construct `@unpack` dynamically as a string
    unpack_txt = "@unpack " * join(keys(Algorithm_structure.Parameters_ranges), ", ") * " = last_trial"

    eval(Meta.parse(unpack_txt))

    # Now, the parameters should be available
    algorithm_instance = Algorithm_structure.Name
    options = Metaheuristics.Options(; seed)

    # Construct the metaheuristic
    metaheuristic = getproperty(Metaheuristics, Symbol(algorithm_instance))(; params..., options)
    
    # Metaheuristic loop
    while !Metaheuristics.should_stop(metaheuristic)
        Metaheuristics.optimize!(f, searchspace, metaheuristic)
        fmin_approx = Metaheuristics.minimum(metaheuristic.status)
        report_value!(trial, fmin_approx)
        should_prune(trial) && return
    end

    fmin_approx = Metaheuristics.minimum(metaheuristic.status)
    fmin_approx - fmin < 1e-8 && report_success!(trial)

    println(fieldnames(typeof(trial)))

    for val in fieldnames(typeof(trial))
        v = String(val)
        if v !== "record"
            println("$val ------------ $(getfield(trial, val))")
        end
    end


    fmin_approx
end


function set_configuration()
    config_parts = []
    
    for (hyperparam, range_vals) in Algorithm_structure.Parameters_ranges
        lb, hb = range_vals[1:2]  # Extract lower and upper bounds
        num_elements = length(range_vals) > 2 ? Int(range_vals[3]) : nothing
        param_type = typeof(Algorithm_structure.Parameters[hyperparam])

        param_value = if param_type == Float64
            "LinRange($lb, $hb, $num_elements)"
        elseif param_type == Int64
            "round.(Int, LinRange($lb, $hb, $num_elements))"
        elseif param_type == Bool
            "[true, false]"
        else
            continue  # Skip unsupported types
        end

        push!(config_parts, "$hyperparam = $param_value")
    end

    # Construct the scenario configuration text
    configuration_txt = """
    scenario = Scenario($(join(config_parts, ", ")),
        instances = $(HyperTuning_configuration.lb_instaces):$(HyperTuning_configuration.max_instace),
        max_trials = $(HyperTuning_configuration.max_trials),
        verbose = true,
        pruner = MedianPruner(prune_after = $(HyperTuning_configuration.max_trials))
    )
    """
    
    print(configuration_txt)

    try
        scenario = Meta.parse(configuration_txt)
        eval(scenario)
        println("\nParsed successfully!")
    catch e
        println("\nParsing failed: ", e)
    end

    HyperTuning.optimize(objective, scenario)

    return scenario
end


run(`clear`) 


scenario = set_configuration()

@info "Top parameters"
# show the list of top parameters regarding success, accuracy, and time
display(top_parameters(scenario))

@info "History"
# display all evaluated trials
display(history(scenario))

# obtain the hyperparameters values
#@unpack N, K, η_max = scenario;

#Metaheuristics.optimize --> logger


