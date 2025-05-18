using Pkg
Pkg.activate(@__DIR__)
using PyCall

optuna = pyimport("optuna")

function objective(trial)
    x = trial.suggest_float("x", -10, 10)
    return (x - 2)^2
end

study = optuna.create_study(sampler=optuna.samplers.RandomSampler())

study.optimize(trial -> objective(trial), n_trials=50)

study.best_params  