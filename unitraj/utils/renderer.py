
import cv2
import numpy as np

from scenarionet.converter.nuplan.utils import camera_params, project_points_cam, COLOR_TABLE, world_to_camera_T, \
    yaw_to_rot


def save_as_video(img_list, save_path):
    # 确定视频的保存路径和帧率
    fps = 10  # 可以根据需要调整帧率

    # 获取图像尺寸
    first_frame = img_list[0]
    h, w, c = first_frame['CAM_F0'].shape
    # 拼接后宽度
    total_width = w * 3

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (total_width, h))

    for frame_dict in img_list:
        # 获取三张图像
        img_L = frame_dict['CAM_L0']
        img_F = frame_dict['CAM_F0']
        img_R = frame_dict['CAM_R0']

        # 确保图像格式是uint8
        img_L = img_L.astype(np.uint8)
        img_F = img_F.astype(np.uint8)
        img_R = img_R.astype(np.uint8)

        # 横向拼接
        concatenated_img = np.hstack((img_L, img_F, img_R))
        concatenated_img = concatenated_img[:, :, ::-1]  # BGR to RGB
        # 写入视频
        video_writer.write(concatenated_img)

    # 释放资源
    video_writer.release()
    print(f'视频已保存到 {save_path}')

def draw_polyline_depth(canvas, polyline3d, T_w2c, K, color,
                        radius=8, seg_interval=0.5,
                        near=1e-3, depth_max=80.):
    H, W = canvas.shape[:2]

    # ---------- 1. 一次性变换 & 投影 ---------- #
    pts_cam = (T_w2c[:3, :3] @ polyline3d.T + T_w2c[:3, 3:4]).T
    z = pts_cam[:, 2]
    cam_mask = z >= near  # 在近平面前方的点
    proj_uv = (K @ pts_cam.T)[:2].T  # shape (N, 2)
    proj_uv /= z[:, None]  # (x/z, y/z)

    u, v = proj_uv[:, 0], proj_uv[:, 1]

    # ---------- 2. per-segment 处理 ---------- #
    for i in range(len(pts_cam) - 1):
        p1c, p2c = pts_cam[i].copy(), pts_cam[i + 1].copy()
        z1, z2 = z[i], z[i + 1]

        # 2-a) z-裁剪到 NEAR
        if z1 < near and z2 < near:
            continue
        if z1 < near or z2 < near:
            t = (near - z1) / (z2 - z1) if z1 < near else (near - z2) / (z1 - z2)
            inter = p1c + t * (p2c - p1c) if z1 < near else p2c + t * (p1c - p2c)
            if z1 < near:
                p1c, z1 = inter, near
            else:
                p2c, z2 = inter, near

            # 只需要为**新增的交点**再算一次投影
            p = (K @ p1c) if z1 == near and (p1c is inter) else (K @ p2c)
            if z1 == near and (p1c is inter):
                proj_uv[i] = p[:2] / p[2]
            else:
                proj_uv[i + 1] = p[:2] / p[2]
            u, v = proj_uv[:, 0], proj_uv[:, 1]  # 更新引用

        # 2-b) 端点像素
        p1 = (int(round(u[i])), int(round(v[i])))
        p2 = (int(round(u[i + 1])), int(round(v[i + 1])))

        # 快速判定“整段在画面内” → 省一次 clipLine
        inside = (
                0 <= p1[0] < W and 0 <= p1[1] < H and
                0 <= p2[0] < W and 0 <= p2[1] < H
        )
        if inside:
            p1_img, p2_img = p1, p2
        else:
            ok, p1_img, p2_img = cv2.clipLine((0, 0, W - 1, H - 1), p1, p2)
            if not ok:
                continue

        # 2-c) 着色
        depth_mean = max(min((z1 + z2) * 0.5, depth_max), 0.)
        alpha = (depth_max - depth_mean) / depth_max
        col = (alpha * color).astype(np.uint8).tolist()

        cv2.line(canvas, p1_img, p2_img, col, radius, cv2.LINE_AA)


def _sutherland_hodgman(poly: np.ndarray, w: int, h: int) -> np.ndarray:
    """Clip a 2‑D polygon against an axis‑aligned screen rectangle using the
    Sutherland–Hodgman algorithm.

    Parameters
    ----------
    poly : (N, 2) array_like
        Polygon vertices (x, y) in image coordinates *in order*.
    w, h : int
        Image width and height.

    Returns
    -------
    np.ndarray, shape (M, 2)
        The clipped polygon (may be empty).
    """

    def clip_edge(pts: list[np.ndarray], inside_fn, intersect_fn):
        if not pts:
            return []
        output = []
        prev = pts[-1]
        prev_inside = inside_fn(prev)
        for curr in pts:
            curr_inside = inside_fn(curr)
            if curr_inside:
                if not prev_inside:  # entering – add intersection first
                    output.append(intersect_fn(prev, curr))
                output.append(curr)
            elif prev_inside:  # leaving – add intersection only
                output.append(intersect_fn(prev, curr))
            prev, prev_inside = curr, curr_inside
        return output

    # Work in float to avoid precision loss
    pts = [np.asarray(p, float) for p in poly.tolist()]

    # Left   (x >= 0)
    pts = clip_edge(
        pts,
        inside_fn=lambda p: p[0] >= 0,
        intersect_fn=lambda p, q: p + (q - p) * ((0 - p[0]) / (q[0] - p[0]))
    )
    if not pts:
        return np.empty((0, 2))

    # Right  (x <= w-1)
    pts = clip_edge(
        pts,
        inside_fn=lambda p: p[0] <= w - 1,
        intersect_fn=lambda p, q: p + (q - p) * ((w - 1 - p[0]) / (q[0] - p[0]))
    )
    if not pts:
        return np.empty((0, 2))

    # Top    (y >= 0)
    pts = clip_edge(
        pts,
        inside_fn=lambda p: p[1] >= 0,
        intersect_fn=lambda p, q: p + (q - p) * ((0 - p[1]) / (q[1] - p[1]))
    )
    if not pts:
        return np.empty((0, 2))

    # Bottom (y <= h-1)
    pts = clip_edge(
        pts,
        inside_fn=lambda p: p[1] <= h - 1,
        intersect_fn=lambda p, q: p + (q - p) * ((h - 1 - p[1]) / (q[1] - p[1]))
    )

    return np.asarray(pts, dtype=np.float32)


def draw_polygon_depth(canvas: np.ndarray,
                       hull3d: np.ndarray,
                       T_w2c: np.ndarray,
                       K: np.ndarray,
                       color: np.ndarray,
                       depth_max) -> None:
    """Project a convex 3‑D polygon and draw its visible part with depth shading.

    Compared with the original implementation, this version **clips** the
    projected polygon against the image boundary so that even if some of the
    polygon’s vertices are outside the frame (or behind the camera), the visible
    portion is still rendered.
    """

    # --- World → camera space ------------------------------------------------
    pts_cam = (T_w2c[:3, :3] @ hull3d.T + T_w2c[:3, 3:4]).T  # (N, 3)

    # Cull vertices that are *behind* the camera (negative z). We ignore them
    # for projection but keep their depth for α if any remain in front.
    in_front = pts_cam[:, 2] > 1e-6
    if not np.any(in_front):
        return  # whole polygon is behind camera

    pts_cam_front = pts_cam[in_front]

    # --- Perspective projection (no validity filtering yet) -----------------
    uv_h = (K @ pts_cam_front.T).T  # (M, 3) – homogeneous
    uv = uv_h[:, :2] / uv_h[:, 2:3]

    # --- Clip against the image rectangle -----------------------------------
    h, w = canvas.shape[:2]
    poly_clipped = _sutherland_hodgman(uv, w, h)
    if poly_clipped.shape[0] < 3:
        return  # Vanishes after clipping

    hull_uv = poly_clipped.astype(np.int32)

    # --- Depth‑based alpha ---------------------------------------------------
    depth_mean = float(np.clip(pts_cam_front[:, 2].mean(), 0.0, depth_max))
    alpha = (depth_max - depth_mean) / depth_max
    col = (alpha * np.asarray(color, dtype=float)).astype(np.uint8).tolist()

    # --- Rasterisation -------------------------------------------------------
    cv2.fillConvexPoly(canvas, hull_uv, col)


def vehicle_corners_local(L, W, H):
    """返回 (8,3) 车辆局部坐标顶点，Z 轴向上"""
    return np.array([
        [L / 2, W / 2, H / 2],  # 0 前左上
        [L / 2, -W / 2, H / 2],  # 1 前右上
        [-L / 2, -W / 2, H / 2],  # 2 后右上
        [-L / 2, W / 2, H / 2],  # 3 后左上
        [L / 2, W / 2, -H / 2],  # 4 前左下
        [L / 2, -W / 2, -H / 2],  # 5 前右下
        [-L / 2, -W / 2, -H / 2],  # 6 后右下
        [-L / 2, W / 2, -H / 2],  # 7 后左下
    ], dtype=np.float32)


def draw_cuboid_depth(canvas,
                      corners_world,  # (8,3) world
                      T_w2c,  # 4×4 world→camera
                      K,  # (3,3) 内参
                      color_rgb=[200, 0, 0],
                      radius=8, depth_max=120.0):
    """
    依次绘制立方体 12 条边；颜色随深度线性衰减
    """
    H, W = canvas.shape[:2]
    color_bgr = tuple(int(c) for c in color_rgb)

    # ---- 世界 → 相机 → 像素 ----
    pts_cam = (T_w2c[:3, :3] @ corners_world.T + T_w2c[:3, 3:4]).T
    uv, valid = project_points_cam(pts_cam, K, (H, W))

    # 若 2 个点都不可见则整车跳过
    if valid.sum() < 2:
        return

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 顶面
        (4, 5), (5, 6), (6, 7), (7, 4),  # 底面
        (0, 4), (1, 5), (2, 6), (3, 7)  # 垂直
    ]

    for i, j in edges:
        if not (valid[i] or valid[j]):
            continue

        z_mean = np.clip((pts_cam[i, 2] + pts_cam[j, 2]) * 0.5, 0, depth_max)
        alpha = (depth_max - z_mean) / depth_max  # 0~1
        col_edge = tuple(int(alpha * c) for c in color_bgr)

        p1, p2 = tuple(uv[i]), tuple(uv[j])
        ok, p1c, p2c = cv2.clipLine((0, 0, W, H), p1, p2)
        if ok:
            cv2.line(canvas, p1c, p2c, col_edge, radius, cv2.LINE_AA)


def draw_heading_arrow(canvas,
                       pos_world,  # (3,)  物体在世界坐标中的质心
                       yaw,  # 标量 (弧度)
                       T_w2c,  # (4,4) 世界→相机
                       K,  # (3,3) 内参
                       color_rgb=(255, 255, 0),
                       arrow_len=3.0,  # 以米为单位，在图上可调
                       thickness=6):
    H, W = canvas.shape[:2]
    color_bgr = tuple(int(c) for c in color_rgb)

    # ---- 1. 计算箭头两端的世界坐标 ------------------------------------------
    # “车头”方向向量（世界系）
    dir_world = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    p_tail_w = pos_world
    p_head_w = pos_world + dir_world * arrow_len

    # ---- 2. 世界 → 相机 -------------------------------------------------------
    pw_tail_c = T_w2c[:3, :3] @ p_tail_w + T_w2c[:3, 3]
    pw_head_c = T_w2c[:3, :3] @ p_head_w + T_w2c[:3, 3]

    # 过滤：若尾点就在相机后面（z<=0），直接跳过
    if pw_tail_c[2] <= 0 or pw_head_c[2] <= 0:
        return

    # ---- 3. 投影到像素坐标 ----------------------------------------------------
    pts_cam = np.vstack([pw_tail_c, pw_head_c])  # shape (2,3)
    uv, valid = project_points_cam(pts_cam, K, (H, W))  # uv: (2,2)

    if not valid.all():
        return

    p_tail_px, p_head_px = map(tuple, uv.astype(int))

    # ---- 4. 画箭头 ------------------------------------------------------------
    cv2.arrowedLine(canvas,
                    p_tail_px,
                    p_head_px,
                    color_bgr,
                    thickness,
                    tipLength=0.25)  # tipLength 相对箭头长度的比例


class ScenarioRenderer:
    def __init__(self, camera_channel_list=['CAM_F0', 'CAM_L0', 'CAM_R0'], width=1920, height=1120, depth_max=120.0):
        self.width = width
        self.height = height
        self.depth_max = depth_max
        self.camera_models = {}
        for k, v in camera_params.items():
            if not k in camera_channel_list: continue
            self.camera_models[k] = v

    def observe(self, scenario, timestamp_idx):
        sdc_track = scenario['tracks'][scenario['metadata']['sdc_id']]
        lidar_pos = sdc_track["state"]["position"][timestamp_idx]
        lidar_yaw = sdc_track["state"]["heading"][timestamp_idx]  # float
        ret_dict = {}
        for cam_id, cam_model in self.camera_models.items():
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            sdc_id = scenario["metadata"]["sdc_id"]
            cam_t = cam_model["sensor2lidar_translation"].copy()  # (3,)
            cam_t[2] += 0.8
            cam_t[0] -= 2
            cam_R = cam_model["sensor2lidar_rotation"]  # (3,3)
            K = cam_model["intrinsics"]  # (3,3)
            T_w2c = world_to_camera_T(lidar_pos, lidar_yaw, cam_t, cam_R)  # 4×4

            for feat in scenario['dynamic_map_states'].values():
                state = feat['state']['object_state'][timestamp_idx]
                pos = feat['stop_point'].copy()
                pos.append(5.0)
                corners_loc = vehicle_corners_local(0.1, 0.1, 0.3)
                pos = np.array(pos, dtype=np.float32)  # (3,)
                corners_world = corners_loc + pos
                if 'RED' in state:
                    color = COLOR_TABLE['traffic_light_red']
                elif 'YELLOW' in state:
                    color = COLOR_TABLE['traffic_light_yellow']
                elif 'GREEN' in state:
                    color = COLOR_TABLE['traffic_light_green']
                else:
                    color = COLOR_TABLE['traffic_light_unknown']
                draw_cuboid_depth(canvas,
                                  corners_world,
                                  T_w2c, K, color_rgb=color, radius=10,depth_max=self.depth_max)

            for feat in scenario['map_features'].values():
                ftype = feat['type']
                if 'LANE' in ftype:
                    poly2d = feat['polygon'].astype(np.float32)
                    pts3d = np.hstack([poly2d, np.zeros((poly2d.shape[0], 1), np.float32)])
                    pts_dist = np.linalg.norm(poly2d - lidar_pos[np.newaxis, :2], axis=1)
                    if np.min(pts_dist) > self.depth_max:
                        continue
                    draw_polyline_depth(canvas, pts3d, T_w2c, K, COLOR_TABLE['lanelines'], radius=2,depth_max=self.depth_max)

                elif 'CROSSWALK' in ftype or 'SPEED_BUMP' in ftype:
                    poly2d = feat['polygon'].astype(np.float32)
                    pts3d = np.hstack([poly2d, np.zeros((poly2d.shape[0], 1), np.float32)])
                    draw_polygon_depth(canvas, pts3d, T_w2c, K, COLOR_TABLE['crosswalks'],self.depth_max)
                    draw_polyline_depth(canvas, pts3d, T_w2c, K, COLOR_TABLE['lanelines'],depth_max=self.depth_max)

                elif 'BOUNDARY' in ftype or 'SOLID' in ftype:
                    poly2d = feat['polyline'].astype(np.float32)
                    pts3d = np.hstack([poly2d, np.zeros((poly2d.shape[0], 1), np.float32)])
                    draw_polyline_depth(canvas, pts3d, T_w2c, K,
                                        COLOR_TABLE['road_boundaries'], radius=10,depth_max=self.depth_max)

            for obj_id, track in scenario["tracks"].items():
                if obj_id == sdc_id:
                    continue
                pos = track["state"]["position"][timestamp_idx]
                yaw = track["state"]["heading"][timestamp_idx]
                L = track["state"]["length"][timestamp_idx][0]
                Wd = track["state"]["width"][timestamp_idx][0]
                H_box = track["state"]["height"][timestamp_idx][0]

                if track["type"] == "BICYCLE":
                    color = COLOR_TABLE['bicycle']
                elif track["type"] == "PEDESTRIAN":
                    color = COLOR_TABLE['pedestrian']
                elif track["type"] == "VEHICLE":
                    color = COLOR_TABLE['vehicle']
                    Wd, H_box = H_box, Wd
                else:
                    color = COLOR_TABLE['vehicle']

                corners_loc = vehicle_corners_local(L, Wd, H_box)
                R_yaw = yaw_to_rot(yaw)
                corners_world = (R_yaw @ corners_loc.T).T + pos

                draw_cuboid_depth(canvas,
                                  corners_world,
                                  T_w2c, K, color)

                # ---- 新增：画朝向箭头 -------------------------------
                draw_heading_arrow(canvas,
                                   pos,  # 物体世界中心
                                   yaw,  # 航向角
                                   T_w2c,
                                   K,
                                   color_rgb=color,  # 与立方体同色
                                   arrow_len=L * 0.9)  # 让箭头大约占车长 60%
            ret_dict[cam_id] = canvas

        return ret_dict
