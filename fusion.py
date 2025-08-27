import os
import torch
import imageio
import warnings
import torchvision
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from omegaconf import OmegaConf
import skimage.transform as sktf
from torch.utils.data import DataLoader
from model.samclip_predictor import SAMCLIP
from model import GaussianModel, render, render_chn
from scene import Scene
from utils.system_utils import searchForMaxIteration, set_seed
from model.render_utils import get_text_features, render_palette
from data.fusion_utils import PointCloudToImageMapper
from data.scannet.scannet_constants import SCANNET20_CLASS_LABELS

def load_3d_model(config):
    """加载3D高斯模型及其属性"""
    scene_config = deepcopy(config)
    if config.model.dynamic:
        scene_config.scene.scene_path = os.path.join(config.scene.scene_path, "0")
    scene = Scene(scene_config.scene)
    gaussians = GaussianModel(config.model.sh_degree)

    if config.model.model_dir:
        if config.model.dynamic:
            gaussians.load_dynamic_npz(os.path.join(config.model.model_dir, "params.npz"), config.model.dynamic_t)
        else:
            loaded_iter = config.model.load_iteration
            if loaded_iter == -1:
                loaded_iter = searchForMaxIteration(os.path.join(config.model.model_dir, "point_cloud"))
            print(f"Loading iteration {loaded_iter}...")
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
    
    return scene, gaussians

def get_depth_data(view, gaussians, config, background):
    """根据配置获取深度数据"""
    depth_source = config.fusion.depth
    if depth_source == "image":
        depth_path = os.path.join(config.scene.scene_path, "depth", view.image_name + ".png")
        return imageio.v2.imread(depth_path) / config.fusion.depth_scale
    elif depth_source == "render":
        output = render(
            view,
            gaussians,
            config.pipeline,
            background,
            override_shape=config.fusion.img_dim,
        )
        return output["depth"].cpu().numpy()[0]
    elif depth_source == "surface":
        return "surface"
    else:
        return None

def process_views(gaussians, views, model_2d, config, background):
    """处理每个视图，进行特征提取和融合"""
    loader = DataLoader(
        views,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=config.fusion.num_workers,
    )
    
    vis_id = torch.zeros((gaussians._xyz.shape[0], len(views)), dtype=int)
    with torch.no_grad():
        for idx, view in enumerate(tqdm(loader)):
            if idx % 5 != 0:
                continue
            view = view[0]
            view.cuda()

            # 提取2D特征
            gt_path = view.image_path
            features = model_2d.extract_image_feature(
                gt_path,
                [config.fusion.img_dim[1], config.fusion.img_dim[0]],
            )
            
            # 获取深度数据
            depth = get_depth_data(view, gaussians, config, background)
            
            # 建立3D到2D的映射
            mapper = PointCloudToImageMapper(
                config.fusion.img_dim,
                config.fusion.visibility_threshold,
                config.fusion.cut_boundary,
                views.camera_info[idx].intrinsics,
            )
            mapping = np.ones([gaussians._xyz.shape[0], 4], dtype=int)
            mapping[:, 1:4], weight = mapper.compute_mapping(
                view.world_view_transform.cpu().numpy(),
                gaussians._xyz.cpu().numpy(),
                depth,
            )

            if mapping[:, 3].sum() == 0:
                continue

            # 融合特征
            mapping = torch.from_numpy(mapping)
            mask = mapping[:, 3]
            vis_id[:, idx] = mask
            features_mapping = features[:, mapping[:, 1], mapping[:, 2]].permute(1, 0).cuda()

            mask_k = mask != 0
            gaussians._times[mask_k] += 1
            gaussians._features_semantic[mask_k] += features_mapping[mask_k]
    
    gaussians._times[gaussians._times == 0] = 1e-5
    gaussians._features_semantic /= gaussians._times
    
    return vis_id

def save_fused_features(gaussians, vis_id, config):
    """保存融合后的特征"""
    point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])
    
    # 创建保存目录
    if config.model.dynamic:
        out_dir = os.path.join(config.fusion.out_dir, str(config.model.dynamic_t))
    else:
        out_dir = config.fusion.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # 保存文件
    for n in range(config.fusion.num_rand_file_per_scene):
        if config.model.dynamic:
            save_path = os.path.join(out_dir, f"{n}.pt")
        else:
            save_path = os.path.join(out_dir, f"{n}.pt")
            
        if gaussians._xyz.shape[0] < config.fusion.n_split_points:
            torch.save(
                {
                    "feat": gaussians._features_semantic.cpu().half(),
                    "mask_full": torch.ones(gaussians._xyz.shape[0], dtype=torch.bool),
                },
                save_path
            )
        else:
            n_points_cur = config.fusion.n_split_points
            rand_ind = np.random.choice(range(gaussians._xyz.shape[0]), n_points_cur, replace=False)

            mask_entire = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool)
            mask_entire[rand_ind] = True
            mask = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool)
            mask[point_ids] = True
            mask_entire = mask_entire & mask

            torch.save(
                {
                    "feat": gaussians._features_semantic[mask_entire].cpu().half(),
                    "mask_full": mask_entire,
                },
                save_path
            )

def fuse_one_scene(config, model_2d):
    """主函数：协调加载、处理和保存"""
    scene, gaussians = load_3d_model(config)
    gaussians.create_semantic(model_2d.embedding_dim)

    bg_color = [1] * model_2d.embedding_dim if config.scene.white_background else [0] * model_2d.embedding_dim
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    views = scene.getTrainCameras()
    
    gaussians._features_semantic = torch.zeros_like(gaussians._features_semantic)
    gaussians._times = torch.zeros_like(gaussians._times)

    vis_id = process_views(gaussians, views, model_2d, config, background)
    save_fused_features(gaussians, vis_id, config)


if __name__ == "__main__":
    config = OmegaConf.load("./config/fusion.yaml")
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)
    print(OmegaConf.to_yaml(config))

    set_seed(config.pipeline.seed)

    model_2d_name = config.fusion.model_2d.lower().replace("_", "")
    model_2d = SAMCLIP("./weights/sam_vit_h_4b8939.pth", "ViT-L/14@336px")

    scenes = os.listdir(config.model.model_dir)
    scenes.sort()
    model_2d.set_predefined_cls(SCANNET20_CLASS_LABELS)

    fuse_one_scene(config, model_2d)
