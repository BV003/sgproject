import os
import cv2
import time
import torch
import viser
from copy import deepcopy
from viser.transforms import SE3, SO3
import numpy as np
from omegaconf import OmegaConf
from model import GaussianModel, render, render_chn
from model.openseg_predictor import OpenSeg
from model.lseg_predictor import LSeg
from model.render_utils import render_palette
from scene import Scene
from utils.system_utils import searchForMaxIteration, set_seed
from utils.camera_utils import get_camera_viser
from utils.sh_utils import RGB2SH, SH2RGB
from data.scannet.scannet_constants import COLORMAP

#输入是一个颜色向量（RGB），输出对应的16进制颜色字符串
def to_hex(color):
    return "{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

#输入是一个字符串列表,用于生成文本提示词集合
def get_text(vocabulary, prefix_prompt=""):
    texts = [prefix_prompt + x.lower().replace("_", " ") for x in vocabulary]
    return texts

def load_models_and_data(config):
    """加载3D高斯模型、语义嵌入及初始属性"""
    # 加载3D高斯场模型
    scene_config = deepcopy(config)
    if config.model.dynamic:
        scene_config.scene.scene_path = os.path.join(config.scene.scene_path, "0")
    scene = Scene(scene_config.scene)
    gaussians = GaussianModel(config.model.sh_degree)

    if config.model.model_dir:
        if config.model.dynamic:
            gaussians.load_dynamic_npz(os.path.join(config.model.model_dir, "params.npz"), 0)
        else:
            if config.model.load_iteration == -1:
                loaded_iter = searchForMaxIteration(os.path.join(config.model.model_dir, "point_cloud"))
            else:
                loaded_iter = config.model.load_iteration
            print(f"Loading trained model at iteration {loaded_iter}")
            gaussians.load_ply(
                os.path.join(
                    config.model.model_dir,
                    "point_cloud",
                    f"iteration_{loaded_iter}",
                    "point_cloud.ply",
                )
            )
    else:
        raise ValueError("模型路径未配置，请设置 model.model_dir")

    # 加载语义嵌入
    fusion = torch.load(config.render.fusion_dir)
    features, mask = fusion["feat"].float().cuda(), fusion["mask_full"].cuda()

    # 保存高斯模型初始属性（用于恢复）
    original_opacity = gaussians._opacity.detach().clone()
    original_scale = gaussians._scaling.detach().clone()
    original_color = gaussians._features_dc.detach().clone()
    original_coord = gaussians._xyz.detach().clone()

    return scene, gaussians, features, mask, (original_opacity, original_scale, original_color, original_coord)

def initialize_core_components(config, scene, features, mask):
    """初始化文本编码模型、颜色映射、相机、可视化服务器等"""
    # 创建文本编码模型
    if config.render.model_2d == "lseg":
        text_model = LSeg(None)  # 512dim CLIP
    else:
        text_model = OpenSeg(None, "ViT-L/14@336px")  # 768dim CLIP

    # 初始化特征存储
    features_save = torch.zeros((mask.shape[0], text_model.embedding_dim)).float().cuda()
    features_save[mask] = features

    # 初始化颜色映射
    colormap = COLORMAP
    colormap_hex = [to_hex(e) for e in colormap]
    colormap_cuda = torch.tensor(colormap).cuda()

    # 初始化相机和背景
    bg_color = [1, 1, 1] if config.scene.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    scene_camera = scene.getTrainCameras()[0]
    width, height = scene_camera.image_width, scene_camera.image_height
    w2c = scene_camera.world_view_transform.cpu().numpy().transpose()

    # 初始化可视化服务器
    server = viser.ViserServer()
    server.world_axes.visible = False

    return (
        text_model, features_save,
        colormap, colormap_hex, colormap_cuda,
        background, scene_camera, width, height, w2c,
        server
    )
    
def create_gui_controls(server):
    """创建所有GUI控件并定义各渲染模式的可见性"""
    # 渲染模式相关控件
    gui_text_rendermode = server.add_gui_markdown("Render Mode")
    gui_render_mode = server.add_gui_dropdown(
        label="Render mode",
        options=["RGB", "Depth", "Semantic", "Relevancy"],
        initial_value="RGB",
        hint="Select rendering mode (RGB/Depth/Semantic/Relevancy)"
    )

    # 渲染参数相关控件
    server.add_gui_markdown(" ")
    gui_text_renderpara = server.add_gui_markdown("Render Parameters")
    gui_near_slider = server.add_gui_slider("Depth near", min=0, max=3, step=0.2, initial_value=1.5)
    gui_far_slider = server.add_gui_slider("Depth far", min=6, max=20, step=0.5, initial_value=6)
    gui_scale_slider = server.add_gui_slider("Gaussian scale", min=0.01, max=1, step=0.01, initial_value=1)

    # 编辑相关控件
    server.add_gui_markdown(" ")
    gui_text_edit = server.add_gui_markdown("Editing")
    gui_edit_mode = server.add_gui_dropdown(
        label="Edit mode",
        options=["Remove", "Color", "Size", "Move"],
        initial_value="Remove",
        hint="选择编辑操作类型：Remove（移除物体）、Color（修改颜色）、Size（调整大小）、Move（移动位置）"
    )
    gui_edit_input = server.add_gui_text("Edit prompt ", "", hint="输入需要编辑的物体名称（用逗号分隔）")
    gui_preserve_input = server.add_gui_text("Preserve prompt ", "", hint="输入需要保留的物体名称（用逗号分隔）")
    gui_editing_button = server.add_gui_button("Apply editing prompt")

    # 文本提示相关控件
    server.add_gui_markdown(" ")
    gui_text_prompt = server.add_gui_markdown("Text Prompt")
    gui_prompt_input = server.add_gui_text(
        "Text prompt",
        "wall,floor,chair,table,door,window,bed,sofa",
        hint="输入场景中需要识别的物体名称（用逗号分隔）"
    )
    gui_prompt_button = server.add_gui_button("Apply text prompt")

    # 颜色映射相关控件
    server.add_gui_markdown(" ")
    gui_text_color = server.add_gui_markdown("Colormap")
    gui_markdown = server.add_gui_markdown("")

    # 整理所有控件
    all_gui_elements = [
        gui_scale_slider, gui_near_slider, gui_far_slider,
        gui_prompt_input, gui_text_prompt, gui_prompt_button,
        gui_edit_mode, gui_edit_input, gui_preserve_input, gui_editing_button,
        gui_markdown, gui_text_color
    ]

    # 定义各渲染模式下的可见控件
    render_mode_visibility = {
        "RGB": [
            gui_scale_slider, gui_edit_mode, gui_edit_input, gui_preserve_input, gui_editing_button
        ],
        "Depth": [
            gui_near_slider, gui_far_slider, gui_scale_slider,
            gui_edit_mode, gui_edit_input, gui_preserve_input, gui_editing_button
        ],
        "Semantic": [
            gui_scale_slider, gui_prompt_input, gui_text_prompt, gui_prompt_button,
            gui_edit_mode, gui_edit_input, gui_preserve_input, gui_editing_button,
            gui_markdown, gui_text_color
        ],
        "Relevancy": [
            gui_scale_slider, gui_prompt_input, gui_text_prompt, gui_prompt_button,
            gui_edit_mode, gui_edit_input, gui_preserve_input, gui_editing_button,
            gui_markdown, gui_text_color
        ]
    }

    return (
        gui_render_mode, gui_near_slider, gui_far_slider, gui_scale_slider,
        gui_edit_mode, gui_edit_input, gui_preserve_input, gui_editing_button,
        gui_prompt_input, gui_prompt_button, gui_markdown,
        all_gui_elements, render_mode_visibility
    )
    
    
def bind_gui_events(
    server, state,
    gui_render_mode, gui_edit_mode, gui_prompt_button, gui_editing_button, gui_scale_slider,
    all_gui_elements, render_mode_visibility
):
    """绑定GUI控件的事件处理逻辑"""
    # 更新GUI控件可见性的工具函数
    def update_gui_visibility(current_mode):
        for elem in all_gui_elements:
            elem.visible = False
        if current_mode in render_mode_visibility:
            for elem in render_mode_visibility[current_mode]:
                elem.visible = True

    # 初始化可见性
    update_gui_visibility(state.render_mode)

    # 渲染模式下拉框事件
    @gui_render_mode.on_update
    def _(event: viser.GuiEvent) -> None:
        state.render_mode = event.target.value
        update_gui_visibility(state.render_mode)
        print(f"切换渲染模式为：{state.render_mode}")

    # 编辑模式下拉框事件
    @gui_edit_mode.on_update
    def _(event: viser.GuiEvent) -> None:
        state.edit_mode = event.target.value
        state.need_color_compute = True

    # 文本提示按钮事件
    @gui_prompt_button.on_click
    def _(_) -> None:
        state.need_color_compute = True

    # 编辑按钮事件
    @gui_editing_button.on_click
    def _(_) -> None:
        state.need_color_compute = True

    # 高斯缩放滑块事件
    @gui_scale_slider.on_update
    def _(_) -> None:
        state.need_color_compute = True

    # 客户端连接事件（相机初始化）
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        print("新客户端连接！")
        c2w_transform = SE3.from_matrix(state.w2c).inverse()
        client.camera.wxyz = c2w_transform.wxyz_xyz[:4]
        client.camera.position = c2w_transform.wxyz_xyz[4:]

        # 相机更新事件
        @client.camera.on_update
        def _(_: viser.CameraHandle) -> None:
            state.need_update = True
    


def main_render_loop(
    config, state,
    gaussians, features, mask, original_props,
    text_model, features_save,
    colormap, colormap_hex, colormap_cuda,
    background, scene_camera,
    server,
    gui_near_slider, gui_far_slider, gui_scale_slider,
    gui_edit_input, gui_preserve_input, gui_prompt_input, gui_markdown
):
    """主渲染循环，处理渲染逻辑和客户端更新"""
    original_opacity, original_scale, original_color, original_coord = original_props
    start_time = time.time() if config.model.dynamic else None
    num_timesteps = config.model.num_timesteps if config.model.dynamic else 0

    while True:
        # 动态模型的时间步更新
        if config.model.dynamic:
            passed_time = time.time() - start_time
            passed_frames = passed_time * config.render.dynamic_fps
            t = int(passed_frames % num_timesteps)

        # 计算颜色和语义映射（当需要时）
        if state.need_color_compute:
            # 1. 处理文本提示和语义特征
            labelset = ["other"] + get_text(gui_prompt_input.value.split(","))
            text_features = text_model.extract_text_feature(labelset).float()
            sim = torch.einsum("cq,dq->dc", text_features, features)
            label_soft = sim.softmax(dim=1)
            label_hard = torch.nn.functional.one_hot(sim.argmax(dim=1), num_classes=label_soft.shape[1]).float()
            soft_save = torch.zeros((mask.shape[0], label_soft.shape[1])).float().cuda()
            soft_save[mask] = label_hard

            # 2. 更新颜色映射UI
            label = sim.argmax(dim=1)
            colors = colormap_cuda[label] / 255
            color_save = torch.zeros((mask.shape[0], 3)).float().cuda()
            color_save[mask] = colors

            # 3. 生成颜色映射表格
            color_mapping = list(zip(labelset, colormap_hex))
            content_head = "| | |\n|:-:|:-|"
            content_body = "".join([
                f"\n|![color](https://dummyimage.com/5x5/{color}/ffffff?text=+)|{label_name}||"
                for label_name, color in color_mapping
            ])
            gui_markdown.content = content_head + content_body

            # 4. 处理场景编辑
            scale_factor = 0.5  # 固定缩放因子
            height = int(scale_factor * scene_camera.image_height)
            width = int(scale_factor * scene_camera.image_width)

            if gui_edit_input.value != "":
                # 编辑逻辑：根据编辑模式修改高斯属性
                len_edit = len(gui_edit_input.value.split(","))
                edit_features = text_model.extract_text_feature(
                    ["other"] + gui_edit_input.value.split(",") + gui_preserve_input.value.split(",")
                ).float()
                sim_edit = torch.einsum("cq,dq->dc", edit_features, features)
                sim_edit[sim_edit < 0] = -2
                label_edit = sim_edit.argmax(dim=1)
                # 恢复初始属性
                gaussians._opacity[:] = original_opacity
                gaussians._features_dc[:] = original_color
                gaussians._scaling[:] = original_scale
                gaussians._xyz[:] = original_coord

                edit_mask = (label_edit > 0) * (label_edit <= len_edit)
                # 根据编辑模式执行操作
                if state.edit_mode == "Remove":
                    tmp = gaussians._opacity[mask]
                    tmp[edit_mask] = -9999
                    gaussians._opacity[mask] = tmp
                elif state.edit_mode == "Color":
                    tmp = gaussians._features_dc[mask]
                    tmp_rgb = SH2RGB(tmp[edit_mask])
                    tmp_rgb = 1 - tmp_rgb
                    tmp_rgb = torch.clamp(tmp_rgb, 0, 1)
                    tmp[edit_mask] = RGB2SH(tmp_rgb)
                    gaussians._features_dc[mask] = tmp
                elif state.edit_mode == "Size":
                    tmp = gaussians._scaling[mask]
                    tmp[edit_mask] *= 2
                    gaussians._scaling[mask] = tmp
                    tmp = gaussians._xyz[mask]
                    tmp[edit_mask] *= 2
                    gaussians._xyz[mask] = tmp
                elif state.edit_mode == "Move":
                    tmp = gaussians._xyz[mask]
                    tmp[edit_mask] += 1
                    gaussians._xyz[mask] = tmp
            else:
                # 无编辑时恢复初始属性
                gaussians._opacity[:] = original_opacity
                gaussians._features_dc[:] = original_color
                gaussians._scaling[:] = original_scale
                gaussians._xyz[:] = original_coord

            state.need_color_compute = False

        # 为每个客户端渲染画面
        for client in server.get_clients().values():
            client_info = client.camera
            # 计算相机矩阵
            w2c_matrix = (
                SE3.from_rotation_and_translation(SO3(client_info.wxyz), client_info.position).inverse().as_matrix()
            )
            client.camera.up_direction = SO3(client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])
            new_camera = get_camera_viser(
                scene_camera,
                w2c_matrix[:3, :3].transpose(),
                w2c_matrix[:3, 3],
                client_info.fov,
                client_info.aspect,
            )
            new_camera.cuda()

            # 动态模型加载当前时间步参数
            if config.model.dynamic:
                gaussians.load_dynamic_npz(os.path.join(config.model.model_dir, "params.npz"), t)

            # 根据渲染模式执行渲染
            if state.render_mode == "RGB":
                output = render(
                    new_camera,
                    gaussians,
                    config.pipeline,
                    background,
                    scaling_modifier=gui_scale_slider.value,
                    override_shape=(width, height),
                )
                rendering = output["render"].cpu().numpy().transpose(1, 2, 0)
            elif state.render_mode == "Depth":
                output = render(
                    new_camera,
                    gaussians,
                    config.pipeline,
                    background,
                    scaling_modifier=gui_scale_slider.value,
                    override_shape=(width, height),
                )
                rendering = output["depth"].cpu().numpy().transpose(1, 2, 0)
                rendering = np.clip(
                    (rendering - gui_near_slider.value) * 255 / (gui_far_slider.value - gui_near_slider.value),
                    0, 255
                )
                rendering = cv2.cvtColor(rendering.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            elif state.render_mode == "Semantic":
                output = render_chn(
                    new_camera,
                    gaussians,
                    config.pipeline,
                    background,
                    scaling_modifier=gui_scale_slider.value,
                    num_channels=soft_save.shape[1],
                    override_color=soft_save,
                    override_shape=(width//2, height//2),
                )
                sim_render = output["render"]
                label_render = sim_render.argmax(dim=0).cpu()
                sem = render_palette(label_render, colormap_cuda.reshape(-1))
                rendering = sem.cpu().numpy().transpose(1, 2, 0)
            else:  # Relevancy
                output = render(
                    new_camera,
                    gaussians,
                    config.pipeline,
                    background,
                    scaling_modifier=gui_scale_slider.value,
                    override_color=color_save,
                    override_shape=(width//2, height//2),
                )
                rendering = output["render"].cpu().numpy().transpose(1, 2, 0)

            # 发送渲染结果到客户端
            client.set_background_image(rendering)


def main(config):
    with torch.no_grad():
        # 定义状态变量（用简单类管理）
        class RenderState:
            def __init__(self):
                self.need_update = False
                self.need_color_compute = True
                self.render_mode = "RGB"
                self.edit_mode = "Remove"
                self.w2c = None  # 后续会赋值

        state = RenderState()

        # 1. 加载模型和数据
        scene, gaussians, features, mask, original_props = load_models_and_data(config)

        # 2. 初始化核心组件
        (
            text_model, features_save,
            colormap, colormap_hex, colormap_cuda,
            background, scene_camera, width, height, w2c,
            server
        ) = initialize_core_components(config, scene, features, mask)
        state.w2c = w2c  # 将相机矩阵存入状态

        # 3. 创建GUI控件
        (
            gui_render_mode, gui_near_slider, gui_far_slider, gui_scale_slider,
            gui_edit_mode, gui_edit_input, gui_preserve_input, gui_editing_button,
            gui_prompt_input, gui_prompt_button, gui_markdown,
            all_gui_elements, render_mode_visibility
        ) = create_gui_controls(server)

        # 4. 绑定GUI事件
        bind_gui_events(
            server, state,
            gui_render_mode, gui_edit_mode, gui_prompt_button, gui_editing_button, gui_scale_slider,
            all_gui_elements, render_mode_visibility
        )

        # 5. 启动主渲染循环
        main_render_loop(
            config, state,
            gaussians, features, mask, original_props,
            text_model, features_save,
            colormap, colormap_hex, colormap_cuda,
            background, scene_camera,
            server,
            gui_near_slider, gui_far_slider, gui_scale_slider,
            gui_edit_input, gui_preserve_input, gui_prompt_input, gui_markdown
        )
        
if __name__ == "__main__":
    config = OmegaConf.load("./config/view.yaml")
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)
    print(OmegaConf.to_yaml(config))

    set_seed(config.pipeline.seed)
    main(config)
        
        
        
        
    
