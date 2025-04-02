using Pkg
Pkg.activate("/home/afonso-meneses/Desktop/THESIS_ALGORITHM/HyperTuning")
import Metaheuristics
using Metaheuristics
using Metaheuristics: TestProblems, optimize, SPEA2, get_non_dominated_solutions, pareto_front, Options
import Metaheuristics.PerformanceIndicators: hypervolume
include("/home/afonso-meneses/Desktop/THESIS_ALGORITHM/utils.jl")
include("/home/afonso-meneses/Desktop/THESIS_ALGORITHM/HyperTuning/Aux_func_Hypertuning.jl")
using .Aux_func_Hypertuning
using .utils 
using DataStructures
using HyperTuning
import HardTestProblems
using Statistics
using Plots
include("/home/afonso-meneses/Desktop/THESIS_ALGORITHM/Hyperoptimization_intervals.jl")
using .Hyperoptimization_intervals


HyperTuning_configuration = utils.HyperTuning_Problems_config(1,50,100)

last_index = 1
#for i in utils.single_opt
Algorithm_structure = utils.Algorithm(:none, OrderedDict(), OrderedDict())
#last_position, Algorithm_structure = utils.create_intervals(Algorithm_structure, txt_path, last_position)

Algorithm_structure, last_index = utils.detect_searchspaces(last_index)
println("$(Algorithm_structure.Name)---- $(Algorithm_structure.Parameters_ranges)")

Algorithm_structure
#end

run(`clear`)


function getproblem(id)
 
    f, conf = HardTestProblems.get_RW_MOP_problem(id)

    reference_point = conf[:nadir]  
    return f, [conf[:xmin] conf[:xmax]], reference_point  # Removed fmin

end

#dont forget to eliminate the Trials dir

# 44, 46

function objective(trial)
    try
        params = (;trial.values...)
        println(params)
        current_instance = trial.instance_id
        problem_name = HardTestProblems.NAME_OF_PROBLEMS_RW_MOP_2021[current_instance]

        f, searchspace, reference_point = try
            getproblem(get_instance(trial))  
        catch e
            @warn "Error retrieving problem for trial $trial: $e"
            return -Inf
        end

        if haskey(Aux_func_Hypertuning.ref_points_offset, current_instance)
            reference_point = Aux_func_Hypertuning.ref_points_offset[current_instance]
        end
               
        hv_values = Aux_func_Hypertuning.run_optimization(trial.instance_id, problem_name, f, searchspace, reference_point, string(Algorithm_structure.Name), params, Algorithm_structure)

        return isempty(hv_values) ? -Inf : -maximum(values(hv_values))
    catch e
        @warn "Unexpected error in objective function: $e"
        return -Inf
    end
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
        verbose = true
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

