using MeshCat, MeshCatMechanisms, RigidBodyDynamics

urdf_path = joinpath(pwd(), "models/a1_quadruped/temp/a1.urdf")
a1 = MeshCatMechanisms.parse_urdf(urdf_path, remove_fixed_tree_joints = true)
a1_visuals = MeshCatMechanisms.URDFVisuals(urdf_path)

state = MechanismState(a1)
state_cache = StateCache(a1)
result = DynamicsResult(a1)
result_cache = DynamicsResultCache(a1)

vis = Visualizer()
mvis = MechanismVisualizer(a1, a1_visuals, vis[:base])
render(vis)
