module utils

using Metaheuristics
using Metaheuristics.PerformanceIndicators: hypervolume, Δₚ
using Metaheuristics: TestProblems, optimize, SPEA2, get_non_dominated_solutions, pareto_front, Options
using Plots
using Plots; gr()
using Nonconvex
using Hyperopt
using DataStructures
include("/home/afonso-meneses/Desktop/GitHub/DynamicOptimization/Hyperoptimization_intervals.jl")
using .Hyperoptimization_intervals


export Problem, Algorithm, create_file, warning_txt_file, create_intervals, set_parameters_hyperopt
export init_algorithm_structure

# ---------------------
# Define Problem Struct
# ---------------------
mutable struct Problem
    func::Function
    bounds::Matrix{Float64}
    pareto_solutions::Vector{Metaheuristics.xFgh_solution{Vector{Float64}}}
end

# ---------------------
# Define Algorithm Struct
# ---------------------
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




single_opt = ["ECA", "DE", "PSO", "ABC", "CGSA", "SA", "WOA", "MCCGA", "GA", "εDE" ] #"BRKGA não está a funcionar"
multi_opt = ["MOEAD_DE", "NSGA2", "NSGA3", "SMS_EMOA", "SPEA2", "CCMO"]  #AINDA FALTA O BCA

#
# ---------------------
# Initialize Algorithm Structure
# ---------------------
function init_algorithm_structure(Name_algorithm::String)
    Algorithm_structure = Algorithm(:none, OrderedDict(), OrderedDict())

    Algorithm_structure.Name = Symbol(Name_algorithm)
    algorithm_instance = getfield(Metaheuristics, Symbol(Algorithm_structure.Name))
    println(algorithm_instance)
    Algorithm_structure.Parameters = get_default_kwargs(algorithm_instance)
    return Algorithm_structure
end


# ---------------------
# Get Default Hyperparameter Values
# ---------------------
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

#Algorithm_structure.Parameters = get_default_kwargs(algorithm_instance)

# ---------------------
# Create Hyperparameter File
# ---------------------
function create_file(Algorithm_structure::Algorithm, dirpath::String, fname:: String)
    txt_path = joinpath(dirpath, fname)
    open(txt_path, "a") do file2
        write(file, string(Algorithm_structure.Name), "\n")
        for field in keys(Algorithm_structure.Parameters)
            write(file, "$field = \n")
        end
        write(file,"\n")
    end
    println("Hyperparameters saved to: ", txt_path)
    #warning_txt_file(txt_path)
    return txt_path
end

function getproblem(id)
    f, conf = HardTestProblems.get_RW_MOP_problem(id)
    reference_point = conf[:nadir]  
    bounds = hcat(conf[:xmin], conf[:xmax])
    return f, bounds, reference_point 
end

# ---------------------
# Get Hyperparameters values from Hyperoptimization_intervals
# ---------------------

function detect_searchspaces(last_index:: Int)


    for (i, searchspace) in enumerate(names(Hyperoptimization_intervals))
        
        if i ≤ last_index
            continue 
        end
        
        if occursin("_searchspace", string(searchspace))
            Algorithm_structure = utils.init_algorithm_structure(string(split(string(searchspace), "_searchspace")[1]))
            current_searchspace = getproperty(Hyperoptimization_intervals, Symbol(searchspace))  
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
        
        last_index = i
        return Algorithm_structure, last_index
        break
        
    end

end


# ---------------------
# Hyperparameter Optimization Setup
# ---------------------
mutable struct Hyperopt_text_form
    Text::String
    Samplers::Vector{String}
end


hv_values = Float64[]  

logger(st) = begin
    A = fvals(st)
    scatter(A[:, 1], A[:,2], label="NSGA-II", title="Gen: $(st.iteration)")
    plot!(pf[:, 1], pf[:,2], label="Pareto Front", lw=2)
    display(current())  # Show the plot
    sleep(0.1)

    reference_point = [1.1, 1.1]  
    hv = hypervolume(A, reference_point)
    
    push!(hv_values, hv)  
    println("Generation $(st.iteration): Hypervolume = $hv")

end


max_hv_index = findfirst(item -> item == maximum(hv_values), hv_values)


# ---------------------
# Parse Hyperopt
# ---------------------

function set_parameters_hyperopt()
    Hyperopt_structure = Hyperopt_text_form("""ho = @hyperopt for i in number_of_samples, """, 
                                            ["RandomSampler(), ", "LHSampler(),", "CLHSampler(),", "Hyperband(R = 50),"])

    println("Choose the sampler:")
    println("[1] RandomSampler()")
    println("[2] LHSampler()")
    println("[3] CLHSampler()")
    println("[4] Hyperband(R = 50)")

    input = tryparse(Int, readline())
    if input === nothing || input > 4 || input < 1
        println("Invalid input, please enter a number between 1-4.")
        return set_parameters_hyperopt()  
    end

    sampler_txt = "sampler = " * Hyperopt_structure.Samplers[input]
    Hyperopt_structure.Text *= sampler_txt
    println(Hyperopt_structure.Text)

    hyper_params = collect(keys(Algorithm_structure.Parameters_ranges))
    last_index = length(hyper_params)

    for (i, field) in enumerate(hyper_params)
        value = Algorithm_structure.Parameters_ranges[field]
        hyp_txt = "$field = $value"
        if i === last_index
            hyp_txt *= " end"
        else
            hyp_txt *= ", "
        end
        Hyperopt_structure.Text *= hyp_txt      
    end
    println(Hyperopt_structure.Text)
end


#set_parameters_hyperopt()



#expr = Meta.parse(hyperopti_in_text) 
#eval(expr)  


end


