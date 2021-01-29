"""
    Double integrator
"""

struct DoubleIntegrator{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int
end

function fd(model::DoubleIntegrator{Discrete, FixedTime}, x⁺, x, u, w, h, t)
    x⁺ - [x[1] + x[2] + w[1]; x[2] + u[1] + w[2]]
end

function fd(model::DoubleIntegrator{Discrete, FixedTime}, x, u, w, h, t)
    [x[1] + x[2] + w[1]; x[2] + u[1] + w[2]]
end

function get_dynamics(model::DoubleIntegrator)
    A = @SMatrix [1.0 1.0; 0.0 1.0]
    B = @SMatrix [0.0; 1.0]

    return A, B
end

n, m, d = 2, 1, 2
model = DoubleIntegrator{Discrete, FixedTime}(n, m, d)

"""
    continuous-time double integrator
"""
struct DoubleIntegratorContinuous{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int
end

function f(model::DoubleIntegratorContinuous, x, u, w)
    [x[2] + w[1]; u[1] + w[2]]
end

state_output(model::DoubleIntegratorContinuous, z) = z

model_con = DoubleIntegratorContinuous{Midpoint, FixedTime}(n, m, d)

# visualization
function visualize!(vis, model::DoubleIntegrator, x;
        color = RGBA(0.0, 0.0, 0.0, 1.0),
        r = 0.1, Δt = 0.1)

    default_background!(vis)

    setobject!(vis["particle"], Sphere(Point3f0(0.0),
        convert(Float32, r)),
        MeshPhongMaterial(color = color))

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    T = length(x)
    for t = 1:T
        pos = [x[t][1]; 0.0; 0.0]

        MeshCat.atframe(anim,t) do
            settransform!(vis["particle"], Translation(pos))
        end
    end

    settransform!(vis["/Cameras/default"],
       compose(Translation(0.0 , 1.0 , -1.0), LinearMap(RotZ(-pi / 2.0))))

    MeshCat.setanimation!(vis, anim)
end
