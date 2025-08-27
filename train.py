import os
import uuid
import torch
from tqdm import tqdm
from functools import partial
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import GaussianModel
from model.renderer import render
from scene import Scene
from utils.system_utils import set_seed
from utils.loss_utils import l1_loss, ssim, psnr


def init_dir(config):
    """
    初始化输出目录并保存配置文件
    
    Args:
        config: 配置参数对象，包含训练和模型相关设置
        
    Returns:
        SummaryWriter: TensorBoard日志记录器
    """
    if not config.train.exp_name:
        unique_str = str(uuid.uuid4())
        config.model.model_dir = os.path.join("./output", unique_str[0:10])
    else:
        config.model.model_dir = f"./output/{config.train.exp_name}"

    print("Output folder: {}".format(config.model.model_dir))
    os.makedirs(config.model.model_dir, exist_ok=True)
    with open(os.path.join(config.model.model_dir, "config.yaml"), "w") as fp:
        OmegaConf.save(config, fp)

    os.makedirs(os.path.join(config.model.model_dir, "tb_logs"), exist_ok=True)
    writer = SummaryWriter(os.path.join(config.model.model_dir, "tb_logs"))
    return writer


def eval(args, iteration, scene: Scene, render_partial, writer):
    """
    在测试集和训练集上评估模型性能
    
    Args:
        args: 配置参数
        iteration: 当前迭代次数
        scene: 场景对象，包含相机和点云数据
        render_partial: 部分应用的渲染函数
        writer: TensorBoard日志记录器
    """
    torch.cuda.empty_cache()
    eval_configs = (
        {"name": "test", "cameras": scene.getTestCameras()},
        {
            "name": "train",
            "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)],
        },
    )

    for config in eval_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            for idx, viewpoint in enumerate(config["cameras"]):
                viewpoint.cuda()
                image = torch.clamp(render_partial(viewpoint)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                if writer and (idx < 5):
                    writer.add_images(
                        "view_{}/render".format(viewpoint.image_name),
                        image[None],
                        global_step=iteration,
                    )
                    if iteration == args.train.test_iterations[0]:
                        writer.add_images(
                            "view_{}/ground_truth".format(viewpoint.image_name),
                            gt_image[None],
                            global_step=iteration,
                        )
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
            psnr_test /= len(config["cameras"])
            l1_test /= len(config["cameras"])
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test))
            writer.add_scalar(f"{config['name']}/l1_loss", l1_test, iteration)
            writer.add_scalar(f"{config['name']}/psnr", psnr_test, iteration)

    torch.cuda.empty_cache()




def train(config):
    # 加载场景数据
    scene = Scene(config.scene)
    # 初始化高斯模型
    gaussians = GaussianModel(config.model.sh_degree)
    first_iter = 0

    
    # 从点云数据创建高斯分布
    gaussians.create_from_pcd(scene.pcd, scene.cameras_extent, config.model.random_init)
    
    # 初始化输出目录并获取日志记录器
    writer = init_dir(config)

    # 设置高斯模型的训练参数（优化器等）
    gaussians.training_setup(config.train)
    # 设置背景颜色
    bg_color = [1, 1, 1] if config.scene.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # 用于测量迭代时间的CUDA事件
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    # 用于平滑显示损失的指数移动平均
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, config.train.iterations), desc="Training progress")
    # 创建训练相机的数据加载器
    loader = DataLoader(
        scene.getTrainCameras(),
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: x,
        num_workers=config.train.num_workers,
    )
    data_iter = iter(loader)
    first_iter += 1

    # 主训练循环
    for iteration in range(first_iter, config.train.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        try:
            viewpoint_cam = next(data_iter)[0]
        except StopIteration:
            data_iter = iter(loader)
            viewpoint_cam = next(data_iter)[0]
        viewpoint_cam.cuda()

        # 渲染当前视角
        # 根据配置决定是否使用随机背景
        bg = torch.rand((3), device="cuda") if config.train.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, config.pipeline, bg)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # 计算损失
        gt_image = viewpoint_cam.original_image #.cuda()
        if config.train.cut_edge:
            h, w = image.shape[1:3]
            ch, cw = h // 100, w // 100
            Ll1 = l1_loss(image[:, ch:-ch, cw:-cw], gt_image[:, ch:-ch, cw:-cw])
            loss = (1.0 - config.train.lambda_dssim) * Ll1 + config.train.lambda_dssim * (
                1.0 - ssim(image[:, ch:-ch, cw:-cw], gt_image[:, ch:-ch, cw:-cw])
            )
        else:
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - config.train.lambda_dssim) * Ll1 + config.train.lambda_dssim * (1.0 - ssim(image, gt_image))
        # 反向传播计算梯度
        loss.backward()

        iter_end.record()

        # 不计算梯度的操作
        with torch.no_grad():
            # Densification
            if iteration < config.train.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > config.train.densify_from_iter and iteration % config.train.densification_interval == 0:
                    size_threshold = 20 if iteration > config.train.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        config.train.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % config.train.opacity_reset_interval == 0 or (
                    config.scene.white_background and iteration == config.train.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < config.train.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad()

            # Logging 记录训练日志到TensorBoard
            writer.add_scalar("train/l1_loss", Ll1.item(), iteration)
            writer.add_scalar("train/total_loss", loss.item(), iteration)
            writer.add_scalar("train/iter_time", iter_start.elapsed_time(iter_end), iteration)
            writer.add_scalar("train/total_points", gaussians.get_xyz.shape[0], iteration)
            writer.add_histogram("train/opacity_histogram", gaussians.get_opacity, iteration)

            # Evaluation在指定迭代次数进行评估
            if iteration in config.train.test_iterations:
                eval(
                    config,
                    iteration,
                    scene,
                    partial(render, pc=gaussians, pipe=config.pipeline, bg_color=bg),
                    writer,
                )

            # Saving gaussians在指定迭代次数保存高斯模型
            if iteration in config.train.save_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                point_cloud_path = os.path.join(config.model.model_dir, "point_cloud/iteration_{}".format(iteration))
                gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        
            # Progress bar更新进度条
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 20 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(20)
            if iteration == config.train.iterations:
                progress_bar.close()


if __name__ == "__main__":
    config = OmegaConf.load("./config/train.yaml")
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)
    print(OmegaConf.to_yaml(config))
    set_seed(config.pipeline.seed)
    train(config)
