#using KhepriAutoCAD
#using Distributed
#addprocs(2)
#@everywhere begin
    #using Pkg
    #Pkg.add(url="https://github.com/aptmcl/KhepriFrame3DD.jl")
    using KhepriFrame3DD

    include(joinpath(@__DIR__, "Hyperoptimization_intervals.jl"))
    include(joinpath(@__DIR__, "utils_minimum_runs.jl"))
    include(joinpath(@__DIR__, "optuna_utils.jl"))
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
    optuna = pyimport("optuna")
    optuna_configuration = Optimization_configuration(1,50,100) ##[LB PROBLEM INSTANCE, HB PROBLEM INSTANCE, NUM OF HP CONFIGURATIONS TO BE TESTED]
#end







# Truss Geometry ---------------------------------------------------------------
#@everywhere begin
    

my_free_truss_node_family = truss_node_family_element(default_truss_node_family(), support=false)
free_node(pt) = truss_node(pt, family=my_free_truss_node_family)
fixed_node(pt) = truss_node(pt, family=fixed_truss_node_family)

space_frame(ptss) =
    let ais = ptss[1],
        bis = ptss[2],
        cis = ptss[3]

        fixed_node(ais[1])
        free_node.(ais[2:end-1])
        fixed_node(ais[end])
        free_node.(bis)
        truss_bar.(ais, cis)
        truss_bar.(bis, ais[1:end-1])
        truss_bar.(bis, cis[1:end-1])
        truss_bar.(bis, ais[2:end])
        truss_bar.(bis, cis[2:end])
        truss_bar.(ais[2:end], ais[1:end-1])
        truss_bar.(bis[2:end], bis[1:end-1])
        if ptss[4:end] == []
            fixed_node(cis[1])
            free_node.(cis[2:end-1])
            fixed_node(cis[end])
            truss_bar.(cis[2:end], cis[1:end-1])
        else
            truss_bar.(bis, ptss[4])
            space_frame(ptss[3:end])
        end
    end



parametric_truss(x11, y11, z11, x12, y12, z12, x13, y13, z13, x21, y21, z21, x22, y22, z22, x31, y31, z31, x32, y32, z32, x33, y33, z33, x41, y41, z41, x42, y42, z42, x51, y51, z51, x52, y52, z52, x53, y53, z53) =
    let p11 = xyz(x11, y11, z11),
        p12 = xyz(x12, y12, z12),
        p13 = xyz(x13, y13, z13),
        p21 = xyz(x21, y21, z21),
        p22 = xyz(x22, y22, z22),
        p31 = xyz(x31, y31, z31),
        p32 = xyz(x32, y32, z32),
        p33 = xyz(x33, y33, z33),
        p41 = xyz(x41, y41, z41),
        p42 = xyz(x42, y42, z42),
        p51 = xyz(x51, y51, z51),
        p52 = xyz(x52, y52, z52),
        p53 = xyz(x53, y53, z53)

        space_frame([[p11, p12, p13],
            [p21, p22],
            [p31, p32, p33],
            [p41, p42],
            [p51, p52, p53]])
    end

#delete_all_shapes()
#parametric_truss(0, 0, 0, 1, 0, 0, 2, 0, 0, 0.5, 0.5, 1, 1.5, 0.5, 1, 0, 1, 0, 1, 1, 0, 2, 1, 0, 0.5, 1.5, 1, 1.5, 1.5, 1, 0, 2, 0, 1, 2, 0, 2, 2, 0)

fixed_parametric_truss(
    x12, y12, z12,
    x21, y21, z21,
    x22, y22, z22,
    x32, y32, z32,
    x41, y41, z41,
    x42, y42, z42,
    x52, y52, z52) =
    begin
        delete_all_shapes()
        parametric_truss(
            0, 0, 0, x12, y12, z12, 20, 0, 0,
            x21, y21, z21, x22, y22, z22,
            0, 10, 0, x32, y32, z32, 20, 10, 0,
            x41, y41, z41, x42, y42, z42,
            0, 20, 0, x52, y52, z52, 20, 20, 0)
    end

#delete_all_shapes()
#fixed_parametric_truss(
#    1.3, 0, 0,
#    0.5, 0.5, 1, 1.5, 0.5, 1,
#    1, 1, 0,
#    0.5, 1.5, 1, 1.5, 1.5, 1,
#    1, 2, 0)

#=
random_fixed_parametric_truss(r) =
    let rnd(v) = random_range(v - r, v + r)*10
        fixed_parametric_truss(
            rnd(1), rnd(0), rnd(0),
            rnd(0.5), rnd(0.5), rnd(1),
            rnd(1.5), rnd(0.5), rnd(1),
            rnd(1), rnd(1), rnd(0),
            rnd(0.5), rnd(1.5), rnd(1),
            rnd(1.5), rnd(1.5), rnd(1),
            rnd(1), rnd(2), rnd(0))
    end
=#

#delete_all_shapes()
#random_fixed_parametric_truss(0.4)

# Truss Analysis and Optimization ----------------------------------------------
using Metaheuristics

## Helper Functions
int2float(x, min, step=0.01) =
    min + step * x

bounds_coordinates(v, r=0.3) = (v - r, v + r) .* 10

## Materials Young's Modulus and Cost
materials_e = [
    1.6409e11, # Cast Iron ASTM A536
    1.86e11,   # Steel, stainless AISI 302
    2.e11,     # Carbon Steel, Structural ASTM A516
    2.0684e11, # Alloy Steel, ASTM A242
    2.047e11,  # Alloy Steel, AISI 4140
    1.93e11,   # Stainless Steel AISI 201
]

materials_cost = [
    460.0,     # Cast Iron ASTM A536
    1480.0,    # Steel, stainless AISI 302
    860.0,     # Carbon Steel, Structural ASTM A516
    950.0,     # Alloy Steel, ASTM A242
    2750.0,    # Alloy Steel, AISI 4140
    1825.0,    # Stainless Steel AISI 201
]

## Objectives
#=
The objectives for the optimization are:
  (1) minimizing the maximum displacement,
  (2) minimizing the material cost of the truss structure.
=#
n_objs = 2

cost(truss_volume, material) =
    truss_volume * materials_cost[Int(material)]

objectives(
    material, bar_radius,
    x12, y12, z12,
    x21, y21, z21,
    x22, y22, z22,
    x32, y32, z32,
    x41, y41, z41,
    x42, y42, z42,
    x52, y52, z52) =
    let b_radius = int2float(bar_radius, 0.035, 0.005),
        load = vz(-3500.0) * 20 * 20 # total load applied to the truss
        set_backend_family(
            default_truss_bar_family(),
            frame3dd,
            frame3dd_truss_bar_family(
                E=materials_e[Int(material)], # (Young's modulus)
                #G=8.1e10,                    # (Kirchoff's or Shear modulus)
                G=7.95e10,                    # (Kirchoff's or Shear modulus)
                p=0.0,                        # Roll angle
                d=7850.0))                    # Density
                #d=77010.0))                   # Density
        with_truss_node_family(radius=b_radius * 2.4) do
            with_truss_bar_family(radius=b_radius, inner_radius=b_radius - 0.02) do
                fixed_parametric_truss(x12, y12, z12, x21, y21, z21, x22, y22, z22, x32, y32, z32, x41, y41, z41, x42, y42, z42, x52, y52, z52)
                free_ns = length(filter(!KhepriBase.truss_node_is_supported, frame3dd.truss_nodes))
                truss_volume = truss_bars_volume()
                analysis = truss_analysis(load / free_ns)
                max_disp = max_displacement(analysis)
                # show_truss_deformation(analysis, autocad, factor=100) # to visualize in AutoCAD
                [max_disp, cost(truss_volume, material)]
            end
        end
    end
#=
problem(x) =
    (objectives(
        x[1], x[2], x[3], x[4], x[5],
        x[6], x[7], x[8], x[9], x[10],
        x[11], x[12], x[13], x[14], x[15],
        x[16], x[17], x[18], x[19], x[20],
        x[21], x[22], x[23]),
    [0.0], [0.0])
=#


problem(x) =
(objectives(
        x[:integer][1], x[:integer][2],
        x[:continuous][1], x[:continuous][2], x[:continuous][3],
        x[:continuous][4], x[:continuous][5], x[:continuous][6],
        x[:continuous][7], x[:continuous][8], x[:continuous][9],
        x[:continuous][10], x[:continuous][11], x[:continuous][12], 
        x[:continuous][13], x[:continuous][14], x[:continuous][15],
        x[:continuous][16], x[:continuous][17], x[:continuous][18],
        x[:continuous][19], x[:continuous][20], x[:continuous][21]),
    [0.0], [0.0])

## Variables
n_vars = 23

material_idx = 1:6
bar_radius = 0:8 # min=0.035, max=0.075, step=0.005

r = 0.4
x12 = bounds_coordinates(1, r)
y12 = bounds_coordinates(0, r)
z12 = bounds_coordinates(0, r)
x21 = bounds_coordinates(0.5, r)
y21 = bounds_coordinates(0.5, r)
z21 = bounds_coordinates(1, r)
x22 = bounds_coordinates(1.5, r)
y22 = bounds_coordinates(0.5, r)
z22 = bounds_coordinates(1, r)
x32 = bounds_coordinates(1, r)
y32 = bounds_coordinates(1, r)
z32 = bounds_coordinates(0, r)
x41 = bounds_coordinates(0.5, r)
y41 = bounds_coordinates(1.5, r)
z41 = bounds_coordinates(1, r)
x42 = bounds_coordinates(1.5, r)
y42 = bounds_coordinates(1.5, r)
z42 = bounds_coordinates(1, r)
x52 = bounds_coordinates(1, r)
y52 = bounds_coordinates(2, r)
z52 = bounds_coordinates(0, r)

points = [x12, y12, z12,
          x21, y21, z21,
          x22, y22, z22,
          x32, y32, z32,
          x41, y41, z41,
          x42, y42, z42,
          x52, y52, z52]

points_lb = [p[1] for p in points]
points_ub = [p[end] for p in points]

integer_space = BoxConstrainedSpace([material_idx[1], bar_radius[1]], [material_idx[end], bar_radius[end]])
continuous_space = BoxConstrainedSpace(points_lb, points_ub)
mixed_space = MixedSpace(:integer => integer_space, :continuous => continuous_space)
#end 
#=
vars_bounds =
    [material_idx[1] bar_radius[1] x12[1] y12[1] z12[1] [
     x21[1]] y21[1] z21[1] x22[1] y22[1] z22[1] [
     x32[1]] y32[1] z32[1] x41[1] y41[1] z41[1] [
     x42[1]] y42[1] z42[1] x52[1] y52[1] z52[1];
     material_idx[end] bar_radius[end] x12[end] y12[end] z12[end] [
     x21[end]] y21[end] z21[end] x22[end] y22[end] z22[end] [
     x32[end]] y32[end] z32[end] x41[end] y41[end] z41[end] [
     x42[end]] y42[end] z42[end] x52[end] y52[end] z52[end]]
=#

## Test Optimization


#@everywhere begin
    base_dir = pwd()
    main_script_name = basename(abspath(@__FILE__))
    searchspace = ["SMS_EMOA_searchspace"]
    CSV_RUNS_FILE_NAME, CSV_LENGTH_RESULTS_NAME = check_CSV(searchspace[1], main_script_name; test = false)
    Algorithm_structure = detect_searchspaces(searchspace[1])
    results_path = joinpath(splitdir(@__DIR__)[1], "Results")
    pop_size = Algorithm_structure.Parameters[:N]
    n_iterations = 20
    max_evals = pop_size * n_iterations
    results = []
    hv_values = Dict()
#end





run(`clear`)





#for current_instance in optuna_configuration.lb_instaces:optuna_configuration.max_instace
    

#options = Options(iterations = n_iterations, f_calls_limit = 3 * max_evals)
#nsga2 = MixedInteger(NSGA2(N = pop_size, options = options))
#results = optimize(problem, mixed_space, nsga2)

#@everywhere begin
    
 algorithm_instance = Algorithm_structure.Name

 println("Using algorithm: $algorithm_instance")
 options = Metaheuristics.Options(iterations = n_iterations, f_calls_limit = 3 * max_evals)
 #options = Metaheuristics.Options(; iterations=num_ite, time_limit=4.0)
 metaheuristic = set_up_algorithm(algorithm_instance, options)
 metaheuristic = MixedInteger(metaheuristic)
 result_dir = "/home/afonso-meneses/Desktop/GitHub/DynamicOptimization/Optuna/Results"
 All_HV = Dict(:Hypervolumes => Float64[])
 num_runs = 100
 reference_point = [10000, 30000]
#end

if pwd() !== result_dir
    cd(result_dir)
end




#All_HV = @sync @distributed (vcat)
  @time for i in 1:num_runs
        println("Starting task...")
        status = optimize(problem, mixed_space, metaheuristic)
        println("Task Finished... iteration : $i")
        approx_front = get_non_dominated_solutions(status.population)
        HV = hypervolume(approx_front, reference_point) 
        push!(All_HV[:Hypervolumes], HV) 
    end

results = All_HV
type_of_result = first(keys(results))                


println("Results::$results")

hv_values[n_iterations] = mean(results[type_of_result])


println("Hypervolume: $(hv_values[n_iterations])")

problem_name = "parametric_truss_example"
current_instance = 0
get_minimum_runs(results, problem_name, current_instance, CSV_RUNS_FILE_NAME)

#end









pop_size = 15
n_iterations = 20
max_evals = pop_size * n_iterations

options = Options( f_calls_limit = 20)
nsga2 = MixedInteger(NSGA2(N = pop_size, options = options))

results = optimize(problem, mixed_space, nsga2)