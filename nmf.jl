using LinearAlgebra
using Plots
using MLDatasets
using ProgressBars

function loss(A, W, Z, β)
    norm(A-W*Z')^2 + β*norm(W)^2 + β*norm(Z)^2
end

function grad_w(A, W, Z, β)
    -2(A-W*Z')Z + 2*β*W
end

function grad_z(A, W, Z, β)
    -2(A-W*Z')'W + 2*β*Z
end

function backtrack_w(A, W, Z, β, init_step, stop_step, default)
    step = init_step
    i = 0
    current_loss = loss(A, W, Z, β)
    grad = grad_w(A, W, Z, β)
    temp = similar(W)
    while true
        temp = W - step*grad
        temp[temp.<0].=0
        if current_loss > loss(A, temp, Z, β)
            return step
        end
        step > stop_step|| break
        step /= 2
    end
    return default
end

function backtrack_z(A, W, Z, β, init_step, stop_step, default)
    step = init_step
    i = 0
    current_loss = loss(A, W, Z, β)
    grad = grad_z(A, W, Z, β)
    temp = similar(Z)
    while true
        temp = Z - step*grad
        temp[temp.<0].=0
        if current_loss > loss(A, W, temp, β)
            return step
        end
        step > stop_step || break
        step /= 2
    end
    return default
end

function gd(A, W, Z, β, init_step, stop_step, default)
    W -= backtrack_w(A, W, Z, β, init_step, stop_step, default)*grad_w(A, W, Z, β)
    Z -= backtrack_z(A, W, Z, β, init_step, stop_step, default)*grad_z(A, W, Z, β)
    return W, Z
end

function NMF(A, W, Z, β, init_step, stop_step, default, iter=100)
    l = Vector{Float64}()
    append!(l, loss(A, W, Z, β))
    for _ in tqdm(1:iter)
        W, Z = gd(A, W, Z, β, init_step, stop_step, default)
        W[W.<0].=0
        Z[Z.<0].=0
        append!(l, loss(A, W, Z, β))
    end
    p = plot(l)
    display(p)
    return W, Z
end

# rk = 10
# β = 0.1
# A = rand(100, 100)
# W = rand(100, rk)
# Z = rand(100, rk)
# NMF(A, W, Z, β, 0.001, 0.00001, 0.00001)

train_x, _ = FashionMNIST.traindata()
test_x, _ = FashionMNIST.testdata()
x = cat(train_x, test_x, dims=3)
x = reshape(x, (784, 70000))'

rk = 5
β = 0.0001
W = rand(70000, rk)
Z = rand(784, rk)
println(size(x))
W, Z = NMF(x, W, Z, β, 0.01, 0.00001, 0.00001, 100)

Z = reshape(Z, (28, 28, 5))
for i in 1:6
    heatmap(Z[:,:,i])
    png("$i")
end
