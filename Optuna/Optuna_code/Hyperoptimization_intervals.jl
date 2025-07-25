

    using DataStructures
    


    NSGA2_searchspace = OrderedDict(
        "N" => [5, 500],        # Changed lower from 1 to 10
        "η_cr" => [1.0, 100.0, 1.0],       # Changed lower from 0 to 1
        "p_cr" => [0.1, 1, 0.05],       # Changed lower from 0 to 0.5
        "η_m" => [1.0, 100.0, 1.0]        # Changed lower from 0 to 1
    )

    NSGA3_searchspace = OrderedDict(
        "N" => [5, 500, 100],
        "η_cr" => [1.0, 100.0, 1.0],
        "p_cr" =>[0.1, 1, 0.05],
        "η_m" => [0, 50, 100], 
        #p_m => 1.0 / D,
        "partitions" => [0, 500, 10],
        #reference_points => Vector{Float64}[],
        #information => Information(),
        #options => Options(),
    );

    SMS_EMOA_searchspace = OrderedDict(
        "N" => [5, 500],
        "η_cr" => [1.0, 100.0, 1.0],
        "p_cr" => [0.1, 1, 0.05],
        "η_m" =>[1.0, 100.0, 1.0],
        #p_m => 1.0 / D,
        "n_samples" => [0, 1000000],
        #information => Information(),
        #options => Options(),
    );

    SPEA2_searchspace = OrderedDict(
    "N" => [5, 500],
    "η_cr" => [1.0, 100.0, 1.0],
    "p_cr" => [0.1, 1, 0.05],
    "η_m" => [1.0, 100.0, 1.0]
    #p_m = 1.0 / D,
    #information = Information(),
    #options = Options(),
    );

    MOEAD_DE_searchspace = OrderedDict( 
        "npartitions" => [5, 30],
        "F" => [0.1, 1.0, 0.05],
        "CR" => [0.1, 1.0, 0.05],
        #"λ" = Array{Vector{Float64}}[], # ref. points
        "η" => [5.0, 50.0, 1.0],
        "p_m" => [0.0, 1.0, 0.1],
        #"T" = round(Int, 0.2*length(weights)),
        "δ" =>[0.1, 1.0, 0.05], 
        #"n_r" = round(Int, 0.02*length(weights)),
        #"z" => zeros(0),
        #"B" = Array{Int}[],
        "s1" => [0.001, 0.1, 0.0005],
        "s2" => [1.0, 50.0, 1.0] 
        #information = Information(),
        #options = Options())
    );


    #NelderMead_searchspace  = OrderedDict(;
    # parameters = AdaptiveParameters(),
    # initial_simplex = AffineSimplexer()
    # )
