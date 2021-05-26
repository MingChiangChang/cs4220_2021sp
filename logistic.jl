using DelimitedFiles
using Plots
using Random

function likelihood(xs, ys, β)
    l = 0
    for i in 1:size(xs)[1]
        l += exp(xs[i,:]'*β)-ys[i]*(xs[i,:]'*β)
    end
    l
end

function grad_poisson(xs, ys, β)
    grad = zeros(size(xs)[2])
    for i in 1:size(xs)[1]
        grad += (xs[i, :].*exp(xs[i, :]'*β).-ys[i].*xs[i, :])
    end
    grad
end

function hessian_poisson(xs,β)
    d = size(xs)[2]
    h = zeros((d,d))
    for i in 1:size(xs)[1]
        h += xs[i,:]*xs[i,:]'.*exp(xs[i,:]'*β)
    end
    h
end

function newton_step(xs, ys, β)
    grad = grad_poisson(xs, ys, β)
    hessian = hessian_poisson(xs, β)
    β -= 0.1*(hessian \ grad)
    β
end

function optimize(xs, ys, iter)
    l = Vector{Float64}()
    β = 0.0001.*rand(size(xs)[2])
    append!(l, likelihood(xs, ys, β))
    for i in 1:iter
        β = newton_step(xs, ys, β)
        append!(l, likelihood(xs, ys, β))
    end
    return β, l
end

data = readdlm("bike_rentals.csv", ',')

xs = data[2:end, 1:7]
ys = data[2:end, 8]
xs=map(x->convert(Float16, x), xs)
ys=map(x->convert(Float16, x), ys)


β, l = optimize(xs, ys, 1000)
println(β)
scatter(0:1000, l)
