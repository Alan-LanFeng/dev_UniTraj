import argparse

from IPython.display import Image as IImage
import pygame
import numpy as np
from PIL import Image
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.envs.scenario_env import ScenarioEnv
import os
import os
from scenarionet.common_utils import read_dataset_summary, read_scenario
import matplotlib.pyplot as plt
import pickle
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm
import imageio
from numpy import array

camera_params = {'CAM_F0': {'distortion': array([-0.356123,  0.172545, -0.00213 ,  0.000464, -0.05231 ]), 'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
       [0.000e+00, 1.545e+03, 5.600e+02],
       [0.000e+00, 0.000e+00, 1.000e+00]]), 'sensor2lidar_rotation': array([[-0.00785972, -0.02271912,  0.99971099],
       [-0.99994262,  0.00745516, -0.00769211],
       [-0.00727825, -0.99971409, -0.02277642]]), 'sensor2lidar_translation': array([ 1.65506747, -0.01168732,  1.49112208])}, 'CAM_L0': {'distortion': array([-0.356123,  0.172545, -0.00213 ,  0.000464, -0.05231 ]), 'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
       [0.000e+00, 1.545e+03, 5.600e+02],
       [0.000e+00, 0.000e+00, 1.000e+00]]), 'sensor2lidar_rotation': array([[ 0.81776776, -0.0057693 ,  0.57551942],
       [-0.57553938, -0.01377628,  0.81765802],
       [ 0.0032112 , -0.99988846, -0.01458626]]), 'sensor2lidar_translation': array([1.63069485, 0.11956747, 1.48117884])}, 'CAM_L1': {'distortion': array([-0.356123,  0.172545, -0.00213 ,  0.000464, -0.05231 ]), 'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
       [0.000e+00, 1.545e+03, 5.600e+02],
       [0.000e+00, 0.000e+00, 1.000e+00]]), 'sensor2lidar_rotation': array([[ 0.93120104,  0.00261563, -0.36449662],
       [ 0.36447127, -0.02048653,  0.93098926],
       [-0.00503215, -0.99978671, -0.0200304 ]]), 'sensor2lidar_translation': array([1.29939471, 0.63819702, 1.36736822])}, 'CAM_L2': {'distortion': array([-0.356123,  0.172545, -0.00213 ,  0.000464, -0.05231 ]), 'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
       [0.000e+00, 1.545e+03, 5.600e+02],
       [0.000e+00, 0.000e+00, 1.000e+00]]), 'sensor2lidar_rotation': array([[ 0.63520782,  0.01497516, -0.77219607],
       [ 0.77232489, -0.00580669,  0.63520119],
       [ 0.00502834, -0.99987101, -0.01525415]]), 'sensor2lidar_translation': array([-0.49561003,  0.54750373,  1.3472672 ])}, 'CAM_R0': {'distortion': array([-0.356123,  0.172545, -0.00213 ,  0.000464, -0.05231 ]), 'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
       [0.000e+00, 1.545e+03, 5.600e+02],
       [0.000e+00, 0.000e+00, 1.000e+00]]), 'sensor2lidar_rotation': array([[-0.82454901,  0.01165722,  0.56567043],
       [-0.56528395,  0.02532491, -0.82450755],
       [-0.02393702, -0.9996113 , -0.01429199]]), 'sensor2lidar_translation': array([ 1.61828343, -0.15532203,  1.49007665])}, 'CAM_R1': {'distortion': array([-0.356123,  0.172545, -0.00213 ,  0.000464, -0.05231 ]), 'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
       [0.000e+00, 1.545e+03, 5.600e+02],
       [0.000e+00, 0.000e+00, 1.000e+00]]), 'sensor2lidar_rotation': array([[-0.92684778,  0.02177016, -0.37480562],
       [ 0.37497631,  0.00421964, -0.92702479],
       [-0.01859993, -0.9997541 , -0.01207426]]), 'sensor2lidar_translation': array([ 1.27299407, -0.60973112,  1.37217911])}, 'CAM_R2': {'distortion': array([-0.356123,  0.172545, -0.00213 ,  0.000464, -0.05231 ]), 'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
       [0.000e+00, 1.545e+03, 5.600e+02],
       [0.000e+00, 0.000e+00, 1.000e+00]]), 'sensor2lidar_rotation': array([[-0.62253245,  0.03706878, -0.78171558],
       [ 0.78163434, -0.02000083, -0.62341618],
       [-0.03874424, -0.99911254, -0.01652307]]), 'sensor2lidar_translation': array([-0.48771615, -0.493167  ,  1.35027683])}, 'CAM_B0': {'distortion': array([-0.356123,  0.172545, -0.00213 ,  0.000464, -0.05231 ]), 'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
       [0.000e+00, 1.545e+03, 5.600e+02],
       [0.000e+00, 0.000e+00, 1.000e+00]]), 'sensor2lidar_rotation': array([[ 0.00802542,  0.01047463, -0.99991293],
       [ 0.99989075, -0.01249671,  0.00789433],
       [-0.01241293, -0.99986705, -0.01057378]]), 'sensor2lidar_translation': array([-0.47463312,  0.02368552,  1.4341838 ])}}

def configure_all_ax(ax) :
    """
    Iterates through 2D ax list/array to apply configurations
    :param ax: 2D list/array of matplotlib ax object
    :return: configure axes
    """
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            configure_ax(ax[i][j])

    return ax

def configure_ax(ax):
    """
    Configure the ax object for general plotting
    :param ax: matplotlib ax object
    :return: ax object without a,y ticks
    """
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


LIDAR_CONFIG = {
    "color_element": "distance",  # ["none", "distance", "x", "y", "z", "intensity", "ring", "id"]
    "color_map": "viridis",
    "x_lim": [-32, 32],
    "y_lim": [-32, 32],
    "z_lim": [-4, 64],
    "alpha": 0.5,
    "size": 0.1,
    "zorder": 3,
}

def get_lidar_pc_color(lidar_pc):
    """
    Compute color map of lidar point cloud according to global configuration
    :param lidar_pc: numpy array of shape (6,n)
    :param as_hex: whether to return hex values, defaults to False
    :return: list of RGB or hex values
    """

    color_intensities = np.linalg.norm(lidar_pc[:, [0,1,2]], axis=-1)
    min, max = color_intensities.min(), color_intensities.max()
    norm_intensities = [(value - min) / (max - min) for value in color_intensities]
    colormap = plt.get_cmap("viridis")
    colors_rgb = np.array([colormap(value) for value in norm_intensities])
    colors_rgb = (colors_rgb[:, :3] * 255).astype(np.uint8)


    return [tuple(value) for value in colors_rgb]
def filter_lidar_pc(lidar_pc):
    """
    Filter lidar point cloud according to global configuration
    :param lidar_pc: numpy array of shape (6,n)
    :return: filtered point cloud
    """

    pc = lidar_pc
    mask = (
        np.ones((len(pc)), dtype=bool)
        & (pc[:, 0] > LIDAR_CONFIG["x_lim"][0])
        & (pc[:, 0] < LIDAR_CONFIG["x_lim"][1])
        & (pc[:, 1] > LIDAR_CONFIG["y_lim"][0])
        & (pc[:, 1] < LIDAR_CONFIG["y_lim"][1])
        & (pc[:, 2] > LIDAR_CONFIG["z_lim"][0])
        & (pc[:, 2] < LIDAR_CONFIG["z_lim"][1])
    )
    pc = pc[mask]
    return pc

def _transform_pcs_to_images(
    lidar_pc,
    sensor2lidar_rotation,
    sensor2lidar_translation,
    intrinsic,
    img_shape,
    eps: float = 1e-3,
):
    """
    Transforms points in camera frame to image pixel coordinates
    TODO: refactor
    :param lidar_pc: lidar point cloud
    :param sensor2lidar_rotation: camera rotation
    :param sensor2lidar_translation: camera translation
    :param intrinsic: camera intrinsics
    :param img_shape: image shape in pixels, defaults to None
    :param eps: threshold for lidar pc height, defaults to 1e-3
    :return: lidar pc in pixel coordinates, mask of values in frame
    """
    pc_xyz = lidar_pc

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    lidar2img_rt = viewpad @ lidar2cam_rt.T

    cur_pc_xyz = np.concatenate([pc_xyz, np.ones_like(pc_xyz)[:, :1]], -1)
    cur_pc_cam = lidar2img_rt @ cur_pc_xyz.T
    cur_pc_cam = cur_pc_cam.T
    cur_pc_in_fov = cur_pc_cam[:, 2] > eps
    cur_pc_cam = cur_pc_cam[..., 0:2] / np.maximum(cur_pc_cam[..., 2:3], np.ones_like(cur_pc_cam[..., 2:3]) * eps)

    if img_shape is not None:
        img_h, img_w = img_shape
        cur_pc_in_fov = (
            cur_pc_in_fov
            & (cur_pc_cam[:, 0] < (img_w - 1))
            & (cur_pc_cam[:, 0] > 0)
            & (cur_pc_cam[:, 1] < (img_h - 1))
            & (cur_pc_cam[:, 1] > 0)
        )
    return cur_pc_cam, cur_pc_in_fov
def add_lidar_to_camera_ax(ax, camera, lidar,camera_params):
    """
    Adds camera image with lidar point cloud on matplotlib ax object
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param lidar: navsim lidar dataclass
    :return: ax object with image
    """
    # import trimesh
    # # 转换为 Trimesh 点云
    # cloud = trimesh.points.PointCloud(pc)
    # # 创建坐标轴
    # axis = trimesh.creation.axis(origin_size=20)  # 坐标轴的原点大小
    # from trimesh.scene import Scene
    # # 创建场景并添加点云和坐标轴
    # scene = Scene()
    # scene.add_geometry(cloud)  # 添加点云
    # scene.add_geometry(axis)
    # scene.show()

    image, lidar_pc = camera.copy(),lidar.copy()
    image_height, image_width = image.shape[:2]

    lidar_pc = filter_lidar_pc(lidar_pc)
    lidar_pc_colors = np.array(get_lidar_pc_color(lidar_pc))

    pc_in_cam, pc_in_fov_mask = _transform_pcs_to_images(
        lidar_pc,
        camera_params['sensor2lidar_rotation'],
        camera_params['sensor2lidar_translation'],
        camera_params['intrinsics'],
        img_shape=(image_height, image_width),
    )

    for (x, y), color in zip(pc_in_cam[pc_in_fov_mask], lidar_pc_colors[pc_in_fov_mask]):
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.circle(image, (int(x), int(y)), 5, color, -1)

    ax.imshow(image)
    return ax

def plot_cameras_frame_with_lidar(rgb,lidar,bev):
    """
    Plots 8x cameras (including the lidar pc) and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    fig, ax = plt.subplots(3, 3, figsize=(12,7))

    add_lidar_to_camera_ax(ax[0, 0], rgb['CAM_L0'], lidar, camera_params['CAM_L0'])
    add_lidar_to_camera_ax(ax[0, 1], rgb['CAM_F0'], lidar, camera_params['CAM_F0'])
    add_lidar_to_camera_ax(ax[0, 2], rgb['CAM_R0'], lidar, camera_params['CAM_R0'])

    #add_lidar_to_camera_ax(ax[1, 0], rgb['CAM_L1'], lidar, camera_params['CAM_L1'])
    ax[1, 1].imshow(bev)
    # add_lidar_to_camera_ax(ax[1, 2], rgb['CAM_R1'], lidar, camera_params['CAM_R1'])
    #
    # add_lidar_to_camera_ax(ax[2, 0], rgb['CAM_L2'], lidar, camera_params['CAM_L2'])
    # add_lidar_to_camera_ax(ax[2, 1], rgb['CAM_B0'], lidar, camera_params['CAM_B0'])
    # add_lidar_to_camera_ax(ax[2, 2], rgb['CAM_R2'], lidar, camera_params['CAM_R2'])

    configure_all_ax(ax)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax

def make_GIF(frames, name="demo.gif"):
    print("Generate gif...")
    imgs = [frame for frame in frames]
    imgs = [Image.fromarray(img) for img in imgs]
    imgs[0].save(name, save_all=True, append_images=imgs[1:], duration=50, loop=0)

def concat_two(first,second):

    # Convert Matplotlib plots to images (PIL)
    first_img = figure_to_image(first)
    second_img = figure_to_image(second)
    # Concatenate images horizontally
    combined_img = Image.new("RGB", (first_img.width + second_img.width, first_img.height))
    combined_img.paste(first_img, (0, 0))
    combined_img.paste(second_img, (first_img.width, 0))
    return np.array(combined_img)


def figure_to_image(fig):
    """Convert a matplotlib figure to a PIL Image."""
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    w, h = canvas.get_width_height()
    image = Image.fromarray(buf.reshape(h, w, 3))
    return image

def load_lidar_from_path_list(lidar_path_list):
    """
    Load lidar point cloud from list of paths
    :param lidar_path_list: list of paths
    :return: lidar point cloud
    """
    import open3d as o3d
    lidar_list = []
    for lidar_path in lidar_path_list:
        lidar = o3d.io.read_point_cloud(lidar_path)
        lidar_list.append(np.asarray(lidar.points))
    return lidar_list
def load_camera_from_path_list(camera_path_list):
    """
    Load camera image from list of paths
    :param camera_path_list: list of paths
    :return: camera image
    """
    camera_list = []
    for camera_path in camera_path_list:
        camera_dict = {}
        for k,v in camera_path.items():
            camera_dict[k] = cv2.imread(v)
        camera_list.append(camera_dict)
    return camera_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/Users/fenglan/Dataset/traffic_dataset/processed", type=str)


    args = parser.parse_args()
    data_path = args.data_path
    summary_dict, summary_list, mapping = read_dataset_summary(data_path)
    num_scenarios = len(summary_list)
    num_scenarios = 2
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Hide the pygame window
    env = ScenarioEnv(
        {
            "manual_control": False,
            "reactive_traffic": False,
            "use_render": False,
            "agent_policy": ReplayEgoCarPolicy,
            "data_directory": data_path, # use nuscenes data
            "num_scenarios": num_scenarios, # load 10 scenarios
            "set_static": True,
        }
    )

    freq = 5
    for seed in tqdm(range(num_scenarios)): # only simulate the first 2 scenarios
        all_frames = []

        print("\nSimulate Scenario: {}".format(seed))
        scenario = read_scenario(data_path, mapping, summary_list[seed])
        all_synthetic_camera = scenario['synthetic_camera']
        all_synthetic_lidar = scenario['synthetic_lidar']
        all_real_camera = scenario['real_camera']
        all_real_lidar = scenario['real_lidar']
        sensor_root = scenario['sensor_root']
        simulated_sensor_root = scenario['sensor_root']

        all_synthetic_camera = load_camera_from_path_list(all_synthetic_camera)
        all_synthetic_lidar = load_lidar_from_path_list(all_synthetic_lidar)
        all_real_camera = load_camera_from_path_list(all_real_camera)
        all_real_lidar = load_lidar_from_path_list(all_real_lidar)

        o, _ = env.reset(seed=seed)
        bev = env.render(mode="top_down", film_size=(4000, 4000), screen_size=(500, 500),
                           target_agent_heading_up=True)
        synthetic_plot,ax = plot_cameras_frame_with_lidar(all_synthetic_camera[0], all_synthetic_lidar[0], bev)
        real_plot,ax = plot_cameras_frame_with_lidar(all_real_camera[0], all_real_lidar[0][:,:3], bev)
        all_frames.append(concat_two(synthetic_plot, real_plot))
        synthetic_frames.append(synthetic_plot)
        real_frames.append(real_plot)

        cnt=1
        scenario = env.engine.data_manager.current_scenario
        horizon = scenario['length']
        for i in tqdm(range(1,horizon)):
            o, r, tm, tc, info = env.step([1.0, 0.])
            if i % freq == 0:
                bev = env.render(mode="top_down", film_size=(4000, 4000), screen_size=(500, 500),
                                   target_agent_heading_up=True)
                synthetic_plot, ax = plot_cameras_frame_with_lidar(all_synthetic_camera[cnt], all_synthetic_lidar[cnt],
                                                                   bev)
                real_plot, ax = plot_cameras_frame_with_lidar(all_real_camera[cnt], all_real_lidar[cnt][:, :3], bev)

                all_frames.append(concat_two(synthetic_plot, real_plot))
                cnt+=1

        
        output_path = f"combined_plots_{seed}.gif"
        imageio.mimsave(output_path, all_frames, fps=2)
        print(f"GIF saved to {output_path}")
        # also save all the frames as jpg
        os.makedirs(f"frames_{seed}", exist_ok=True)
        for i, frame in enumerate(all_frames):
            plt.imsave(f"frames_{seed}/{i}.jpg", frame)

        #break
        env.close()
