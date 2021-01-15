using LinearAlgebra, ForwardDiff, StaticArrays, Plots

# rod
struct Rod
      m # mass
      J # inertia
      l # length
      g # gravity

      nq # configuration dimension
      nλ # force-input dimension
end

# kinematics
function kinematics(model, q)
      y, z, θ = q
      l = model.l

      # (p1y, p1z, p2y, p2z)
      @SVector [y + 0.5 * l * sin(θ),
                z - 0.5 * l * cos(θ),
                y - 0.5 * l * sin(θ),
                z + 0.5 * l * cos(θ)]
end

# jacobian
function jacobian(model, q)
      y, z, θ = q
      l = model.l

      @SMatrix [1.0 0.0 0.5 * l * cos(θ);
                0.0 1.0 0.5 * l * sin(θ);
                1.0 0.0 -0.5 * l * cos(θ);
                0.0 1.0 -0.5 * l * sin(θ)]
end

# mass matrix
function mass_matrix(model)
      m = model.m
      J = model.J

      Diagonal(@SVector [m, m, J])
end

# gravity
function gravity(model, q)
      y, z, θ = q
      m = model.m
      l = model.l
      g = model.g

      @SVector [0.0, m * g, 0.0]
end

# dynamics
function dynamics(model, q1, q2, q3, λ, h)
      nq = model.nq
      SVector{nq}(mass_matrix(model) * (2.0 * q2 - q1 - q3) / h
            - h * gravity(model, q2)
            + h * jacobian(model, q3)' * λ)
end

# create model
m = 1.0                    # mass
l = 1.0                    # length
J = 1.0 / 12.0 * m * l^2.0 # inertia
g = 9.18                   # gravity
nq = 3                     # configuration dimension
nλ = 4                     # force input dimension
h = 0.01                    # time step

model = Rod(m, J, l, g, nq, nλ)
q = zeros(model.nq)
λ = zeros(model.nλ)

kinematics(model, q)
jacobian(model, q)
mass_matrix(model)
gravity(model, q)
dynamics(model, q, q, q, λ, h)

# 1-step simulate
include(joinpath(pwd(), "src/solvers/newton.jl"))
dynamics(x) = dynamics(model, q, q, x, λ, h)
dynamics(q)
q⁺ = newton(dynamics, q)
dynamics(x) = dynamics(model, q, q⁺, x, λ, h)
q⁺ = newton(dynamics, q⁺)

# pinned dynamics
function simulated_pinned_dynamics()
      # conditions
      q1 = @SVector [0.5 * model.l, 0.0, 0.5 * π]
      q2 = copy(q1)

      # initial force guess
      λ1 = zeros(2)

      # trajectories
      q_hist = [q1, q2]
      λ_hist = []

      for t = 1:100
            # pinned dynamics
            function pinned_dynamics(x)
                  nq = model.nq
                  nλ = model.nλ

                  q = view(x, 1:nq)
                  λ = [zeros(2); view(x, nq .+ (1:2))]

                  SVector{nq + 2}([dynamics(model, q1, q2, q, λ, h);
                                   kinematics(model, q)[3:4]])
            end

            # step
            x3 = newton(pinned_dynamics, [q2; λ1], tol_r = 1.0e-12)

            # cache results
            push!(q_hist, view(x3, 1:nq))
            push!(λ_hist, view(x3, nq .+ (1:2)))

            q2 = q_hist[end]
            q1 = q_hist[end-1]
            λ1 = λ_hist[end]
      end

      return q_hist, λ_hist
end

q_hist, λ_hist = simulated_pinned_dynamics()

plot(hcat(q_hist...)', labels = ["y" "z" "t"])
plot(hcat(λ_hist...)')

# pinned dynamics (double pendulum)
function simulated_pinned_dynamics()
      # conditions
      q1 = @SVector [0.5 * model.l, 1.5, 0.5 * π, model.l + 0.5 * model.l, 1.5, 0.5 * π]
      q2 = copy(q1)

      # initial force guess
      λ1 = zeros(4)

      # trajectories
      q_hist = [q1, q2]
      λ_hist = []

      for t = 1:500
            println("t = $t")
            # pinned dynamics
            function pinned_dynamics(x)
                  nq = model.nq
                  nλ = model.nλ

                  q = view(x, 1:2 * nq)
                  λ = view(x, 2 * nq .+ (1:4))

                  SVector{2 * nq + 4}([dynamics(model,
                                    view(q1, 1:nq),
                                    view(q2, 1:nq),
                                    view(q, 1:nq),
                                    [λ[3:4]; λ[1:2]],
                                    h);
                                   dynamics(model,
                                    view(q1, nq .+ (1:nq)),
                                    view(q2, nq .+ (1:nq)),
                                    view(q, nq .+ (1:nq)),
                                    [zeros(2); -1.0 * λ[3:4]],
                                    h);
                                   kinematics(model, view(q, 1:nq))[3:4] - [0.0; 1.5];
                                   kinematics(model, view(q, 1:nq))[1:2] -
                                    kinematics(model, view(q, nq .+ (1:nq)))[3:4]])
            end

            # step
            x3 = newton(pinned_dynamics, [q2; zeros(4)], tol_r = 1.0e-12)

            # cache results
            push!(q_hist, view(x3, 1:2 * nq))
            push!(λ_hist, view(x3, 2 * nq .+ (1:4)))

            q2 = q_hist[end]
            q1 = q_hist[end-1]
            λ1 = λ_hist[end]
      end

      return q_hist, λ_hist
end

q_hist, λ_hist = simulated_pinned_dynamics()

plot(hcat([q[1:nq] for q in q_hist]...)', labels = ["y1" "z1" "t1"])
plot(hcat([q[nq .+ (1:nq)] for q in q_hist]...)', labels = ["y2" "z2" "t2"])
plot(hcat(λ_hist...)')

# Visualize
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

function visualize!(vis, model, x;
        color=RGBA(0.0, 0.0, 0.0, 1.0),
        r = 0.1, Δt = 0.1)

    default_background!(vis)

    l1 = Cylinder(Point3f0(0.0, 0.0, 0.0), Point3f0(0.0, 0.0, model.l),
        convert(Float32, 0.025))
    setobject!(vis["l1"], l1, MeshPhongMaterial(color = color))
    l2 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l),
        convert(Float32, 0.025))
    setobject!(vis["l2"], l2, MeshPhongMaterial(color = color))

    setobject!(vis["elbow"], Sphere(Point3f0(0.0),
        convert(Float32, 0.05)),
        MeshPhongMaterial(color = RGBA(0.0, 1.0, 0.0, 1.0)))
    setobject!(vis["ee"], Sphere(Point3f0(0.0),
        convert(Float32, 0.05)),
        MeshPhongMaterial(color = RGBA(0.0, 1.0, 0.0, 1.0)))

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    T = length(x)
    for t = 1:T
        k1 = kinematics(model, view(x[t], 1:model.nq))
        k2 = kinematics(model, view(x[t], model.nq .+ (1:model.nq)))

        MeshCat.atframe(anim,t) do
            settransform!(vis["l1"], cable_transform([k1[3]; 0.0; k1[4]], [k1[1]; 0.0; k1[2]]))
            settransform!(vis["l2"], cable_transform([k2[3]; 0.0; k2[4]], [k2[1]; 0.0; k2[2]]))

            settransform!(vis["elbow"], Translation([k1[1]; 0.0; k1[2]]))
            settransform!(vis["ee"], Translation([k2[1]; 0.0; k2[2]]))
        end
    end

    settransform!(vis["/Cameras/default"],
       compose(Translation(0.0 , 1.0 , -1.0), LinearMap(RotZ(pi / 2.0))))

    MeshCat.setanimation!(vis, anim)
end

visualize!(vis, model, q_hist, Δt = h)

# pinned dynamics (double pendulum w/ impact)
"""
	Newton
"""
function newton_slack(res::Function, x;
        idx = (1:0),
		tol_r = 1.0e-8, tol_d = 1.0e-6)
	y = copy(x)
	Δy = copy(x)

    r = res(y)

    iter = 0

    while norm(r, 2) > tol_r && iter < 25
        ∇r = ForwardDiff.jacobian(res, y)

		try
        	Δy = -1.0 * ∇r \ r
		catch
			@warn "implicit-function failure"
			return y
		end

        α = 1.0

		iter_ls = 0

        while any((y + α * Δy)[idx] .< 0.0)
            α *= 0.5
            iter_ls += 1
            iter_ls == 50 && (@warn "line search (slack) failed")
        end

        iter_ls = 0

        while α > 1.0e-8 && iter_ls < 10
            ŷ = y + α * Δy
            r̂ = res(ŷ)

            if norm(r̂) < norm(r)
                y = ŷ
                r = r̂
                break
            else
                α *= 0.5
				iter_ls += 1
            end

			iter_ls == 10 && (@warn "line search failed")
        end

        iter += 1
    end

	iter == 10 && (@warn "Newton failure")

    return y
end

function simulated_pinned_impact_dynamics()
      # conditions
      q1 = @SVector [0.5 * model.l, 1.5, 0.5 * π, model.l + 0.5 * model.l, 1.5, 0.5 * π]
      q2 = copy(q1)

      function pinned_dynamics_nom(x)
            nq = model.nq
            nλ = model.nλ

            q = view(x, 1:2 * nq)
            λ = view(x, 2 * nq .+ (1:4))

            SVector{2 * nq + 4}([dynamics(model,
                              view(q1, 1:nq),
                              view(q2, 1:nq),
                              view(q, 1:nq),
                              [λ[3:4]; λ[1:2]],
                              h);
                             dynamics(model,
                              view(q1, nq .+ (1:nq)),
                              view(q2, nq .+ (1:nq)),
                              view(q, nq .+ (1:nq)),
                              [zeros(2); -1.0 * λ[3:4]],
                              h);
                             kinematics(model, view(q, 1:nq))[3:4] - [0.0; 1.5];
                             kinematics(model, view(q, 1:nq))[1:2] -
                              kinematics(model, view(q, nq .+ (1:nq)))[3:4]])
      end

      x1 = newton(pinned_dynamics_nom, [q2; zeros(4)],
          tol_r = 1.0e-8)

      # initial force guess
      λ1 = [x1[2nq .+ (1:4)]; 0.0]

      # trajectories
      q_hist = [q1, q2]
      λ_hist = []

      for t = 1:300
            println("t = $t")

            ρ = 1.0
            # pinned dynamics
            function pinned_dynamics(x)
                  nq = model.nq
                  nλ = model.nλ

                  q = view(x, 1:2 * nq)
                  λ = view(x, 2 * nq .+ (1:5))
                  s = x[2 * nq + 5 + 1]

                  SVector{2 * nq + 6}([dynamics(model,
                                    view(q1, 1:nq),
                                    view(q2, 1:nq),
                                    view(q, 1:nq),
                                    [λ[3:4]; λ[1:2]],
                                    h);
                                   dynamics(model,
                                    view(q1, nq .+ (1:nq)),
                                    view(q2, nq .+ (1:nq)),
                                    view(q, nq .+ (1:nq)),
                                    [0.0; λ[5]; -1.0 * λ[3:4]],
                                    h);
                                   kinematics(model, view(q, 1:nq))[3:4] - [0.0; 1.5];
                                   kinematics(model, view(q, 1:nq))[1:2] -
                                    kinematics(model, view(q, nq .+ (1:nq)))[3:4];
                                   s - kinematics(model, view(q, nq .+ (1:nq)))[2];
                                   1.0 / ρ + λ[5] * s])
            end

            # step
            x3 = newton_slack(pinned_dynamics, [q2; λ1[1:4]; 0.1; 0.1],
                idx = (2 * nq + 4 .+ (1:2)),
                tol_r = 1.0e-8)

            for i = 1:5
                println("   i = $i")

                ρ *= 10.0
                # pinned dynamics
                function _pinned_dynamics(x)
                      nq = model.nq
                      nλ = model.nλ

                      q = view(x, 1:2 * nq)
                      λ = view(x, 2 * nq .+ (1:5))
                      s = x[2 * nq + 6]

                      SVector{2 * nq + 6}([dynamics(model,
                                        view(q1, 1:nq),
                                        view(q2, 1:nq),
                                        view(q, 1:nq),
                                        [λ[3:4]; λ[1:2]],
                                        h);
                                       dynamics(model,
                                        view(q1, nq .+ (1:nq)),
                                        view(q2, nq .+ (1:nq)),
                                        view(q, nq .+ (1:nq)),
                                        [0.0; λ[5]; -1.0 * λ[3:4]],
                                        h);
                                       kinematics(model, view(q, 1:nq))[3:4] - [0.0; 1.5];
                                       kinematics(model, view(q, 1:nq))[1:2] -
                                        kinematics(model, view(q, nq .+ (1:nq)))[3:4];
                                       s - kinematics(model, view(q, nq .+ (1:nq)))[2];
                                       1.0 / ρ - λ[5] * s])
                end

                # step
                x3 = newton_slack(_pinned_dynamics, x3,
                    idx = (2 * nq + 4 .+ (1:2)),
                    tol_r = 1.0e-8)

                println("COMP: $(x3[2nq + 5] * x3[2nq + 6])")
                println("ρ: $ρ")
                println("error: $(norm(_pinned_dynamics(x3)))")
            end

            # cache results
            push!(q_hist, view(x3, 1:2 * nq))
            push!(λ_hist, view(x3, 2 * nq .+ (1:5)))

            q2 = q_hist[end]
            q1 = q_hist[end-1]
            λ1 = λ_hist[end]
      end

      return q_hist, λ_hist
end

q_hist, λ_hist = simulated_pinned_impact_dynamics()
vis = Visualizer()
render(vis)
visualize!(vis, model, q_hist, Δt = h)
