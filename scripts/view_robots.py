# Created by skywoodsz on 2026/01/28.
import argparse
import itertools
from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="View ATEC Robots.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from dataclasses import replace
from atec_rl_lab.assets.robots import (
    UNITREE_B2_CFG,
    UNITREE_B2_PIPER_CFG,
    UNITREE_B2W_CFG,
    UNITREE_B2W_PIPER_CFG,
    TRON1A_WHEEL_CFG,
    TRON1A_PIPER_CFG,
    TRON2A_LEGGED_CFG,
    TRON2A_WHEEL_CFG,
    PIPER_CFG,
    UNITREE_G1_29DOF_DEX1_CFG
)

class ATECSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    b2 = replace(UNITREE_B2_CFG, prim_path="{ENV_REGEX_NS}/b2")

    b2w = replace(UNITREE_B2W_CFG, prim_path="{ENV_REGEX_NS}/b2w")
    b2w.init_state.pos = (b2w.init_state.pos[0], b2w.init_state.pos[1] + 1.0, b2w.init_state.pos[2] + 0.166)
    
    b2w_piper = replace(UNITREE_B2W_PIPER_CFG, prim_path="{ENV_REGEX_NS}/b2w_piper")
    b2w_piper.init_state.pos = (b2w_piper.init_state.pos[0], b2w_piper.init_state.pos[1] + 2.0, b2w_piper.init_state.pos[2] + 0.166)

    b2_piper = replace(UNITREE_B2_PIPER_CFG, prim_path="{ENV_REGEX_NS}/b2_piper")
    b2_piper.init_state.pos = (b2_piper.init_state.pos[0], b2_piper.init_state.pos[1] + 3.0, b2_piper.init_state.pos[2] + 0.166)

    tron1a = replace(TRON1A_WHEEL_CFG, prim_path="{ENV_REGEX_NS}/tron1a")
    tron1a.init_state.pos = (tron1a.init_state.pos[0], tron1a.init_state.pos[1] + 4.0, tron1a.init_state.pos[2] + 0.166)

    tron1a_piper = replace(TRON1A_PIPER_CFG, prim_path="{ENV_REGEX_NS}/tron1a_piper")
    tron1a_piper.init_state.pos = (tron1a_piper.init_state.pos[0], tron1a_piper.init_state.pos[1] + 5.0, tron1a_piper.init_state.pos[2] + 0.166)
    
    tron2a_legged = replace(TRON2A_LEGGED_CFG, prim_path="{ENV_REGEX_NS}/tron2a_legged")
    tron2a_legged.init_state.pos = (tron2a_legged.init_state.pos[0], tron2a_legged.init_state.pos[1] + 6.0, tron2a_legged.init_state.pos[2] + 0.166)

    tron2a_wheel = replace(TRON2A_WHEEL_CFG, prim_path="{ENV_REGEX_NS}/tron2a_wheel")
    tron2a_wheel.init_state.pos = (tron2a_wheel.init_state.pos[0], tron2a_wheel.init_state.pos[1] + 7.0, tron2a_wheel.init_state.pos[2] + 0.166)

    piper = replace(PIPER_CFG, prim_path="{ENV_REGEX_NS}/piper")
    piper.init_state.pos = (piper.init_state.pos[0], piper.init_state.pos[1] + 8.0, piper.init_state.pos[2])

    g1 = replace(UNITREE_G1_29DOF_DEX1_CFG, prim_path="{ENV_REGEX_NS}/g1")
    g1.init_state.pos = (g1.init_state.pos[0], g1.init_state.pos[1] + 9.0, g1.init_state.pos[2])

def main():
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    scene_cfg = ATECSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=4.0,
        replicate_physics=True
    )
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    scene.reset()

    for name, articulation in scene.articulations.items():
        print("-"*100)
        print("Robot name:", name)
        print("Bodies:", articulation.num_bodies, "->", articulation.body_names)
        print("Joints:", articulation.num_joints, "->", articulation.joint_names)
        articulation.set_joint_position_target(articulation.data.default_joint_pos)

    for i in itertools.count():
        if not simulation_app.is_running():
            break
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
