from isaacgym import gymapi

import torch

def create_sim():
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()


    # set common parameters
    sim_params.dt = 1 / 60
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    # set PhysX-specific parameters
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0

    # Get the ordinal of the default CUDA device
    compute_device_id = 0#torch.cuda.current_device()
    # Get the ordinal of the default OpenGL device
    graphics_device_id = 0#torch.cuda.current_device() 

    sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)


    # configure the ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
    plane_params.distance = 0
    plane_params.static_friction = 1
    plane_params.dynamic_friction = 1
    plane_params.restitution = 0

    # create the ground plane
    gym.add_ground(sim, plane_params)

    return sim