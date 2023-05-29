from isaacgym import gymapi
from GymPoseCollector.simulation import create_sim
import torch

def main():
    gym = gymapi.acquire_gym()
    sim = create_sim()
    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)

    while True:
         # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)


if __name__ == '__main__':
    main()