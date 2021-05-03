"""
      Quadrotor

      Dynamics model from "Trajectory generation and control for precise
      aggressive maneuvers with quadrotors". Orientation represented with
      Modified Rodrigues Parameters.
"""

struct Quadrotor{I, T} <: Model{I, T}
      n::Int
      m::Int
      d::Int

      mass # mass
      J       # inertia matrix
      Jinv    # inertia matrix inverse
      g       # gravity
      L    # length
      kf   # coefficient
      km   # coefficient
end

function f(model::Quadrotor, z, u, w)
      # states
      x = view(z,1:3)
      r = view(z,4:6)
      v = view(z,7:9)
      ω = view(z,10:12)

      # controls
      w1 = u[1]
      w2 = u[2]
      w3 = u[3]
      w4 = u[4]

      # forces
      F1 = model.kf * w1
      F2 = model.kf * w2
      F3 = model.kf * w3
      F4 = model.kf * w4

      F = @SVector [0.0,
                    0.0,
                    F1 + F2 + F3 + F4] # total rotor force in body frame

      # moments
      M1 = model.km * w1
      M2 = model.km * w2
      M3 = model.km * w3
      M4 = model.km * w4

      τ = @SVector [model.L * (F2 - F4),
                    model.L * (F3 - F1),
                    (M1 - M2 + M3 - M4)] # total rotor torque in body frame

      SVector{12}([v;
                   0.25 * ((1.0 - r' * r) * ω - 2.0 * cross(ω, r) + 2.0*(ω' * r) * r);
                   model.g + (1.0 / model.mass) * MRP(r[1], r[2], r[3]) * F;
                   model.Jinv * (τ - cross(ω, model.J * ω))])
end

n, m, d = 12, 4, 12

mass = 0.5
J = Diagonal(@SVector[0.0023, 0.0023, 0.004])
Jinv = Diagonal(@SVector[1.0 / 0.0023, 1.0 / 0.0023, 1.0 / 0.004])
L = 0.175
kf = 1.0
km = 0.0245

model = Quadrotor{Midpoint, FixedTime}(n, m, d,
                  mass,
                  J,
                  Jinv,
                  @SVector[0.0, 0.0, -9.81],
                  L,
                  kf,
                  km)

function visualize!(vis, p::Quadrotor, q; Δt = 0.1)
    default_background!(vis)

    obj_path = joinpath(pwd(),
      "models/quadrotor/drone.obj")
    mtl_path = joinpath(pwd(),
      "models/quadrotor/drone.mtl")

    ctm = ModifiedMeshFileObject(obj_path, mtl_path, scale = 1.0)
    setobject!(vis["drone"], ctm)
    settransform!(vis["drone"], LinearMap(RotZ(pi) * RotX(pi / 2.0)))

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    for t = 1:length(q)
        MeshCat.atframe(anim, t) do
            settransform!(vis["drone"],
                  compose(Translation(q[t][1:3]),
                        LinearMap(MRP(q[t][4:6]...) * RotX(pi / 2.0))))
        end
    end
    # settransform!(vis["/Cameras/default"], compose(Translation(0.0, 0.0, 0.0),
    # LinearMap(RotZ(pi/2))))
    MeshCat.setanimation!(vis, anim)
end
