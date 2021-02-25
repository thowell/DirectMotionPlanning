using StaticArrays, LinearAlgebra, ForwardDiff, Plots

mutable struct Hopper{T}
    mb::T
    ml::T
    Jb::T
    Jl::T
    r::T
    μ::T
    g::T
end

# Dimensions
nq = 5 # configuration dim
nu = 2 # control dim
nc = 1 # number of contact points
nf = 2 # number of faces for friction cone pyramid

# Parameters
g = 9.81 # gravity
Δt = 0.1 # time step
μ = 0.5  # coefficient of friction
mb = 10. # body mass
ml = 1.  # leg mass
Jb = 2.5 # body inertia
Jl = 0.25 # leg inertia
r = 0.7 # nominal leg length
model = Hopper(mb,ml,Jb,Jl,r,μ,g)

# from "Dynamically Stable Legged Locomotion p. 144"

function gen_z(func,t,t_flight,t_stance)
    step = 0
    z = zero(t)

    for (i,tt) in enumerate(t)
        if tt > (step+0.5)*(t_flight + t_stance)
            step += 1
        end

        if step == 0
            z[i] = func(tt+0.5*t_flight)
        else
            z[i] = func(tt-((step-1+0.5)*t_flight + step*t_stance))
        end
    end
    return z
end

function raibert_trajectories(h::Hopper,ẋ,t_step,ρ,tf,T;compress=0.875)
    t_flight = (1-ρ)*t_step
    t_stance = ρ*t_step

    Jeff = h.Jb*h.Jl/(h.Jb+h.Jl)
    kh = ((2π/t_step)^2)*Jeff
    ωh = sqrt(kh/Jeff)
    z_min = compress*h.r

    dθ0 = ẋ/z_min

    θ_max = dθ0/ωh

    t = range(0,stop=tf,length=T)
    r_param = [1 1; cos(ωh*(t_flight*0.5+t_stance*0.5)) 1]\[h.r;z_min]

    z_max = 0.125*h.g*((1-ρ)^2)*t_step^2 + h.r*cos(dθ0/ωh*sin(ωh*t_flight*0.5))

    żlo = 0.5*t_flight*h.g
    zlo = z_max - 0.5/h.g*żlo^2
    function z_traj(i)
        zlo + żlo*i - 0.5*h.g*i^2
    end

    x = range(0,stop=ẋ*tf,length=T)
    z = gen_z(z_traj,t,t_flight,t_stance)
    r_traj = r_param[1]*cos.(ωh.*t) .+ r_param[2]
    θ = θ_max.*sin.(ωh.*t)
    ϕ = -Jl/Jb.*θ

    M_traj = θ.*kh
    ṙ_traj = 100*(r_traj .- r)

    Q = [[x[t];z[t];r_traj[t];ϕ[t];θ[t]] for t=1:T]
    U = [[M_traj[t];ṙ_traj[t]] for t=1:T]

    return Q, U, x, z, r_traj, ϕ, θ, M_traj, ṙ_traj, t
end

ẋ = 3
t_step = 0.5
ρ = 0.125
tf = 1.5
T = 1000
Q,U,_x,_z,_r,_θ,_ϕ,_M,_ṙ,_t = raibert_trajectories(model,ẋ,t_step,ρ,tf,T,compress=0.89)

plot(_t,_z,color=:red,width=2.,label="z")
plot!(_t,_r,color=:orange,label="r",width=2.)
plot!(_t,_θ,color=:yellow,label="theta",width=2.)
plot!(_t,_ϕ,color=:cyan,label="phi",width=2.)

plot!(_t,0.01*_M)
plot!(_t,0.01*_ṙ)
