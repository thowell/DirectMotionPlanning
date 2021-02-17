abstract type Integration end

abstract type Model{I <: Integration, T <: Time} end

"""
	struct TemplateModel{I, T} <: Model{I, T}
		n::Int # state dimension
		m::Int # control dimension
		d::Int # disturbance dimension

		# additional model parameters ...
	end

	function f(model::TemplateModel, x, u, w)
		# continuous-time dynamics
		nothing
	end

	function k(model::TemplateModel, x)
		# kinematics
		nothing
	end

	model = TemplateModel{Integration, Time}(0, 0, 0)

	state_output(model, x) = x
	control_output(model, u) = u
"""

include_model(str::String) = include(joinpath(pwd(), "models", str * ".jl"))

"""
	propagate dynamics with implicit integrator
	- Levenberg-Marquardt
	- Newton
"""

function propagate_dynamics(model, x, u, w, h, t;
		solver = :levenberg_marquardt,
		tol_r = 1.0e-8, tol_d = 1.0e-6)

	res(z) = fd(model, z, x, u, w, h, t)

	return eval(solver)(res, copy(x), tol_r = tol_r, tol_d = tol_d)
end

function propagate_dynamics_jacobian(model, x, u, w, h, t;
		solver = :levenberg_marquardt,
		tol_r = 1.0e-8, tol_d = 1.0e-6)

	y = propagate_dynamics(model, x, u, w, h, t,
			solver = solver,
			tol_r = tol_r, tol_d = tol_d)

    dy(z) = fd(model, z, x, u, w, h, t)
	dx(z) = fd(model, y, z, u, w, h, t)
	du(z) = fd(model, y, x, z, w, h, t)

	Dy = ForwardDiff.jacobian(dy, y)
	A = -1.0 * Dy \ ForwardDiff.jacobian(dx, x)
	B = -1.0 * Dy \ ForwardDiff.jacobian(du, u)

	return y, A, B
end

"""
	get gravity compensating torques
"""
function initial_torque(model, q1, h;
	solver = :newton,
	tol_r = 1.0e-8, tol_d = 1.0e-6)

	x = [q1; q1]
    res(z) = fd(model, x, x, z, zeros(model.d), h, 0)

	return eval(solver)(res, 1.0e-5 * rand(model.m),
		tol_r = tol_r, tol_d = tol_d)
end

# Model conversions

"""
	free final time model
"""
function free_time_model(model::Model{I, FixedTime}) where I <: Integration
	model_ft = typeof(model).name.wrapper{I, FreeTime}([f == :m ? getfield(model,f) + 1 : getfield(model,f)
	 	for f in fieldnames(typeof(model))]...)
	return model_ft
end

function free_time_model(model::Model{I, FreeTime}) where I <: Integration
	@warn "Model is already free time"
	return model
end

"""
	no slip model
"""
function no_slip_model(model)
	# modify parameters
	m = model.nu + model.nc + model.nb + model.ns
	idx_ψ = (1:0)
	idx_η = (1:0)
	idx_s = model.nu + model.nc + model.nb .+ (1:model.ns)

	# assemble update parameters
	params = []
	for f in fieldnames(typeof(model))
		if f == :m
			push!(params, m)
		elseif f == :idx_ψ
			push!(params, idx_ψ)
		elseif f == :idx_η
			push!(params, idx_η)
		elseif f == :idx_s
			push!(params, idx_s)
		else
			push!(params, getfield(model,f))
		end
	end

	return typeof(model)(params...)
end

state_output(model, x) = x
