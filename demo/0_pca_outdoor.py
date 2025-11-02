# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import concerto
import torch
import open3d as o3d
import argparse
import numpy as np

from concerto.transform import Compose

try:
    import flash_attn
except ImportError:
    flash_attn = None
device = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM_CONFIG = [
    dict(type="RandomScale", scale=[0.2, 0.2]),
    dict(
        type="GridSample",
        grid_size=0.01,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        return_inverse=True,
    ),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "color", "inverse"),
        feat_keys=("coord", "color", "normal"),
    ),
]

def get_pca_color(feat, brightness=1.25, center=True):
    u, s, v = torch.pca_lowrank(feat, center=center, q=12, niter=5)
    projection = feat @ v
    # projection = projection[:, 9:12]
    projection = projection[:, :3] * 0.2 + projection[:, 3:6] * 0.2 + projection[:, 6:9] * 0.1 + projection[:, 9:12] * 0.5
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    color = color.clamp(0.0, 1.0)
    return color


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wo_color',
        dest='wo_color',
        action='store_true',
        help="disable the color."
    )
    parser.add_argument(
        '--wo_normal',
        dest='wo_normal', 
        action='store_true', 
        help="disable the normal."
    )
    args = parser.parse_args()
    # set random seed
    # (random seed affect pca color, yet change random seed need manual adjustment kmeans)
    # (the pca prevent in paper is with another version of cuda and pytorch environment)
    concerto.utils.set_seed(6783)
    # Load model
    if flash_attn is not None:
        model = concerto.load("concerto_large_outdoor", repo_id="Pointcept/Concerto").to(device)
    else:
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],  # reduce patch size if necessary
            enable_flash=False,
        )
        model = concerto.load(
            "concerto_large_outdoor", repo_id="Pointcept/Concerto", custom_config=custom_config
        ).to(device)
    # Load default data transform pipeline
    transform = Compose(TRANSFORM_CONFIG)
    # Load data
    point = concerto.data.load("sample2_outdoor")

    if args.wo_color:
        point["color"] = np.zeros_like(point["coord"])
    if args.wo_normal:
        point["normal"] = np.zeros_like(point["coord"])

    original_coord = point["coord"].copy()
    original_color = point["color"].copy()
    point = transform(point)

    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor) and device == "cuda":
                point[key] = point[key].cuda(non_blocking=True)
        # model forward:
        point = model(point)
        # upcast point feature
        # Point is a structure contains all the information during forward
        for _ in range(2):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent

        # here point is down-sampled by GridSampling in default transform pipeline
        # feature of point cloud in original scale can be acquired by:
        _ = point.feat[point.inverse]

        # PCA
        pca_color = get_pca_color(point.feat, brightness=1, center=True)

    # # inverse back to original scale before grid sampling
    # # point.inverse is acquired from the GirdSampling transform
    # original_pca_color = pca_color[point.inverse]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(original_coord)
    # pcd.colors = o3d.utility.Vector3dVector(original_pca_color.cpu().detach().numpy())
    # o3d.visualization.draw_geometries([pcd])
    # # or
    # # o3d.visualization.draw_plotly([pcd])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point.coord.cpu().detach().numpy())
    pcd.colors = o3d.utility.Vector3dVector(pca_color.cpu().detach().numpy())
    o3d.io.write_point_cloud("pca.ply", pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point.coord.cpu().detach().numpy())
    pcd.colors = o3d.utility.Vector3dVector(point.color.cpu().detach().numpy())
    o3d.io.write_point_cloud("pcd.ply", pcd)
