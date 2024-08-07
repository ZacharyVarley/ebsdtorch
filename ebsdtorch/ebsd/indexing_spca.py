# from torch import Tensor
# import torch
# from ebsdtorch.ebsd.master_pattern import MasterPattern
# from ebsdtorch.ebsd.experiment_pats import ExperimentPatterns
# from ebsdtorch.ebsd.geometry import EBSDGeometry
# from ebsdtorch.s2_and_so3.laue_fz_ori import sample_ori_fz_laue_angle
# from ebsdtorch.s2_and_so3.laue_fz_s2 import s2_in_fz_laue, s2_to_fz_laue
# from ebsdtorch.s2_and_so3.quaternions import qu_apply
# from ebsdtorch.s2_and_so3.sphere import inv_rosca_lambert, xyz_to_theta_phi
# from ebsdtorch.preprocessing.radial_mask import get_radial_mask
# from ebsdtorch.utils.knn import ChunkedKNN
# from ebsdtorch.utils.progressbar import progressbar
# from ebsdtorch.utils.pca import OnlineCovMatrix
# import kikuchipy as kp
# import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# geom = EBSDGeometry(
#     detector_shape=(60, 60),
#     proj_center=(0.4221, 0.2179, 0.4954),
# ).to(device)

# # create the master pattern
# kp_mp = kp.data.nickel_ebsd_master_pattern_small(
#     projection="lambert", hemisphere="both"
# )
# mp_nh = torch.from_numpy(kp_mp.data[0, :, :].astype(np.float32)).to(torch.float32)
# mp_sh = torch.from_numpy(kp_mp.data[1, :, :].astype(np.float32)).to(torch.float32)
# master_pattern = torch.concat((mp_nh, mp_sh), dim=-1)
# mp = MasterPattern(
#     master_pattern,
#     laue_group=11,
# ).to(device)
# mp.normalize(norm_type="minmax")
# # mp.apply_clahe()

# # create the experiment patterns object
# experiment_patterns = (
#     torch.tensor(kp.data.ni_gain(number=1, allow_download=True).data)
#     .to(device)
#     .to(torch.float32)
# )

# experiment_patterns = ExperimentPatterns(
#     experiment_patterns,
# )

# # subtract background and do clahe
# experiment_patterns.standard_clean()
# # exp_pats.do_nlpar()


# # index the patterns
# # create a dictionary object for each master pattern
# # we could save some compute by only computing the cosine vectors once
# # per unique Laue group ID and reuse those coordinates for all master patterns
# # with the same Laue group ID. But more than a few phases in a sample is rare.

# mp = mp.to(device)
# pca_resolution_pixels = 500
# dictionary_resolution_degrees = 0.5
# dictionary_chunk_size = 4096
# virtual_binning = 1
# top_k_matches = 1
# distance_metric = "angular"
# match_dtype = torch.float16
# quantized_via_ao = False


# # get an orientation dictionary
# ori_tensor = sample_ori_fz_laue_angle(
#     laue_id=mp.laue_group,
#     angular_resolution_deg=dictionary_resolution_degrees,
#     device=mp.master_pattern.device,
# )

# # make an object to do the pattern comparisons
# knn = ChunkedKNN(
#     data_size=len(ori_tensor),
#     query_size=experiment_patterns.n_patterns,
#     topk=top_k_matches,
#     match_device=mp.master_pattern.device,
#     distance_metric=distance_metric,
#     match_dtype=match_dtype,
#     quantized_via_ao=quantized_via_ao,
# )

# # get a helper function to sync devices
# if experiment_patterns.patterns.device.type == "cuda":
#     sync = torch.cuda.synchronize
# elif experiment_patterns.patterns.device.type == "mps":
#     sync = torch.mps.synchronize
# elif (
#     experiment_patterns.patterns.device.type == "xpu"
#     or experiment_patterns.patterns.device.type == "xla"
# ):
#     sync = torch.xpu.synchronize
# else:
#     sync = lambda: None

# # get planar coordinates that are within the fundamental sector
# planar_coords = torch.stack(
#     torch.meshgrid(
#         torch.linspace(
#             -1.0, 1.0, pca_resolution_pixels, device=mp.master_pattern.device
#         ),
#         torch.linspace(
#             -1.0, 1.0, pca_resolution_pixels, device=mp.master_pattern.device
#         ),
#         indexing="ij",
#     ),
#     dim=-1,
# ).view(-1, 2)

# xyz_NH = inv_rosca_lambert(planar_coords)
# xyz_SH = xyz_NH.clone()
# xyz_SH[..., 2] = -xyz_SH[..., 2]
# xyz = torch.cat((xyz_NH, xyz_SH), dim=0)

# in_fz = s2_in_fz_laue(xyz, mp.laue_group)
# xyz_fz = xyz[in_fz]
# xyz_nfz = xyz[~in_fz]

# print(f"proportion of points in the fundamental zone: {in_fz.float().mean()}")

# # make an object to do the PCA
# pcacovmat = OnlineCovMatrix(
#     n_features=len(xyz_fz),
# ).to(mp.master_pattern.device)

# pb = progressbar(
#     list(torch.split(ori_tensor, dictionary_chunk_size)), "Indexing Orientations"
# )

# print("Starting PCA indexing...")

# # iterate over the dictionary in chunks
# for ori_batch in pb:

#     # use orientations to rotate the rays to spherical pixel locations x, y, z
#     rotated_rays = qu_apply(ori_batch[:, None, :], xyz_fz)

#     # interpolate the master pattern at the rotated coordinates
#     simulated_patterns = mp.interpolate(
#         rotated_rays,
#         mode="bilinear",
#         align_corners=True,
#         normalize_coords=True,  # not already normalized above
#         virtual_binning=virtual_binning,  # coarser grid requires blur of MP
#     ).squeeze()

#     # update the covariance matrix
#     pcacovmat(simulated_patterns)

#     # flatten simulated patterns to (n_ori, H*W)
#     simulated_patterns = simulated_patterns.view(len(ori_batch), -1)

#     # # must remove mean from each simulated pattern
#     # simulated_patterns = simulated_patterns - torch.mean(
#     #     simulated_patterns,
#     #     dim=-1,
#     #     keepdim=True,
#     # )

#     # sync to make sure progress bar is updated
#     sync()

# # get the eigenvectors and visualize them
# eigenvectors = pcacovmat.get_eigenvectors()

# # trim the eigenvectors to the number of components
# # they are returned in ascending order of eigenvalue
# pca_matrix = eigenvectors[:, -500:]

# # each xyz_nfz gets the value of the nearest coordinate in the fundamental zone
# xyz_nfz_into_fz = s2_to_fz_laue(xyz_nfz, mp.laue_group)

# # # can use Euclidean distance to find the nearest neighbor
# # distance_matrix = torch.cdist(xyz_nfz_into_fz, xyz_fz)
# # # get the indices of the nearest neighbors
# # _, nearest_indices = torch.min(distance_matrix, dim=-1)
# # # get the PCA values for the nearest neighbors
# # pca_values_nfz = pca_matrix[nearest_indices]
# # pca_values = torch.empty(
# #     (len(xyz), pca_matrix.shape[-1]), device=mp.master_pattern.device
# # )
# # pca_values[in_fz] = pca_matrix
# # pca_values[~in_fz] = pca_values_nfz

# # After computing pca_matrix, use KDTree for efficient nearest neighbor search
# from scipy.spatial import cKDTree

# tree = cKDTree(xyz_fz.cpu().numpy())
# distances, indices = tree.query(xyz_nfz_into_fz.cpu().numpy(), k=1)
# pca_values_nfz = pca_matrix[indices]
# pca_values = torch.empty(
#     (len(xyz), pca_matrix.shape[-1]), device=mp.master_pattern.device
# )
# pca_values[in_fz] = pca_matrix
# pca_values[~in_fz] = pca_values_nfz.to(mp.master_pattern.device)


# # # multiply by -1 if the mean cube value is negative
# # mean_cube_lt_0 = (pca_values**3).mean(dim=0) < 0
# # pca_values[:, mean_cube_lt_0] *= -1

# # now make a gif showing the first 10 eigenvectors
# import matplotlib.animation as animation

# # fig, ax = plt.subplots()
# # ims = []
# # for i in range(64):
# #     im = ax.imshow(
# #         pca_values[: (pca_resolution_pixels * pca_resolution_pixels), -(i + 1)]
# #         .view(pca_resolution_pixels, pca_resolution_pixels)
# #         .cpu()
# #         .numpy(),
# #     )
# #     ims.append([im])

# # ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=100)

# # # save gif
# # ani.save("pca.gif")

# # # make a 10x10 grid of the first 100 eigenvectors
# # fig, axs = plt.subplots(8, 8, figsize=(20, 20))
# # for i in range(64):
# #     ax = axs[i // 8, i % 8]
# #     ax.imshow(
# #         pca_values[: (pca_resolution_pixels * pca_resolution_pixels), -(i + 1)]
# #         .view(pca_resolution_pixels, pca_resolution_pixels)
# #         .cpu()
# #         .numpy(),
# #     )
# #     ax.axis("off")

# # plt.tight_layout()

# # # save image
# # plt.savefig("pca.png")

# # # do another plot with eigenvectors 1, 2, 4, 8, 16, 32, 64, 128
# # fig, axs = plt.subplots(3, 3, figsize=(10, 10))
# # for i, ax in enumerate(axs.flat):
# #     ax.imshow(
# #         pca_values[: (pca_resolution_pixels * pca_resolution_pixels), -(2**i)]
# #         .view(pca_resolution_pixels, pca_resolution_pixels)
# #         .cpu()
# #         .numpy(),
# #     )
# #     ax.axis("off")
# #     ax.set_title(f"PC {2**i}")

# # plt.tight_layout()
# # plt.savefig("pca_2.png")

# # make a gif cycling through 1st, 2nd, 4th, 8th, 16th, 32nd, 64th, 128th, 256th eigenvectors
# # but plot it on the surface of a sphere instead of the rosca-lambert projection to the square


# import plotly.graph_objects as go
# import numpy as np
# import moviepy.editor as mpy
# from PIL import Image
# import io


# def create_frame(i):
#     colors = pca_values[: (pca_resolution_pixels * pca_resolution_pixels), -(2**i)]
#     colors = colors - colors.min()
#     colors = colors / colors.max()

#     x = (
#         xyz_NH[..., 0]
#         .cpu()
#         .numpy()
#         .reshape(pca_resolution_pixels, pca_resolution_pixels)
#     )
#     y = (
#         xyz_NH[..., 1]
#         .cpu()
#         .numpy()
#         .reshape(pca_resolution_pixels, pca_resolution_pixels)
#     )
#     z = (
#         xyz_NH[..., 2]
#         .cpu()
#         .numpy()
#         .reshape(pca_resolution_pixels, pca_resolution_pixels)
#     )

#     # colorscale = [[0, "rgb(0,0,255)"], [1, "rgb(255,0,0)"]]

#     fig = go.Figure(
#         data=[
#             go.Surface(
#                 x=x,
#                 y=y,
#                 z=z,
#                 surfacecolor=colors.cpu()
#                 .numpy()
#                 .reshape(pca_resolution_pixels, pca_resolution_pixels),
#                 colorscale="Greys",
#             )
#         ]
#     )
#     fig.update_layout(
#         title=f"PC {2**i}",
#         scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
#         width=1000,
#         height=1000,
#         scene_camera=dict(eye=dict(x=0, y=0, z=2), up=dict(x=0, y=1, z=0)),
#     )
#     return fig


# # Create frames
# frames = [create_frame(i) for i in range(8)]


# # Create animation using MoviePy
# def make_frame(t):
#     frame_idx = int(t)  # 2 fps, so multiply by 2
#     img_bytes = frames[frame_idx].to_image(format="png")
#     img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#     return np.array(img)


# animation = mpy.VideoClip(make_frame, duration=8)  # 4 seconds total
# # animation.write_videofile(
# #     "sphere_pca_plotly_hq.mp4",
# #     fps=1,
# #     codec="libx264",
# #     bitrate="8000k",
# #     preset="veryslow",
# #     ffmpeg_params=["-crf", "17"],
# # )

# # # Write WebM
# # animation.write_videofile(
# #     "sphere_pca_plotly_hq.webm",
# #     fps=1,
# #     codec="libvpx-vp9",
# #     bitrate="2000k",
# #     ffmpeg_params=["-crf", "30", "-b:v", "0"],
# # )

# animation.write_gif("sphere_pca_plotly.gif", fps=1)

# # If you want an HTML version (interactive)
# frames[0].write_html(
#     "sphere_pca_plotly.html",
#     auto_play=True,
#     animation_opts=dict(
#         frame=dict(duration=500, redraw=True), transition=dict(duration=0)
#     ),
# )


# # # repeat the entire process with a master pattern consisting of Gaussian noise (mean 0, std 1)
# # mp = MasterPattern(
# #     torch.randn_like(mp.master_pattern),
# #     laue_group=11,
# # ).to(device)
# # mp.normalize(norm_type="minmax")

# # # make an object to do the PCA
# # pcacovmat = OnlineCovMatrix(
# #     n_features=len(xyz_fz),
# # ).to(mp.master_pattern.device)

# # pb = progressbar(
# #     list(torch.split(ori_tensor, dictionary_chunk_size)), "Indexing Orientations"
# # )

# # print("Starting PCA indexing...")

# # # iterate over the dictionary in chunks

# # for ori_batch in pb:
# #     # use orientations to rotate the rays to spherical pixel locations x, y, z
# #     rotated_rays = qu_apply(ori_batch[:, None, :], xyz_fz)

# #     # interpolate the master pattern at the rotated coordinates
# #     simulated_patterns = mp.interpolate(
# #         rotated_rays,
# #         mode="bilinear",
# #         align_corners=True,
# #         normalize_coords=True,  # not already normalized above
# #         virtual_binning=virtual_binning,  # coarser grid requires blur of MP
# #     ).squeeze()

# #     # update the covariance matrix
# #     pcacovmat(simulated_patterns)

# #     # flatten simulated patterns to (n_ori, H*W)
# #     simulated_patterns = simulated_patterns.view(len(ori_batch), -1)

# #     # # must remove mean from each simulated pattern
# #     # simulated_patterns = simulated_patterns - torch.mean(
# #     #     simulated_patterns,
# #     #     dim=-1,
# #     # )

# #     # sync to make sure progress bar is updated
# #     sync()

# # # get the eigenvectors and visualize them
# # eigenvectors = pcacovmat.get_eigenvectors()

# # # trim the eigenvectors to the number of components
# # # they are returned in ascending order of eigenvalue
# # pca_matrix = eigenvectors

# # # each xyz_nfz gets the value of the nearest coordinate in the fundamental zone
# # xyz_nfz_into_fz = s2_to_fz_laue(xyz_nfz, mp.laue_group)

# # # can use Euclidean distance to find the nearest neighbor
# # distance_matrix = torch.cdist(xyz_nfz_into_fz, xyz_fz)

# # # get the indices of the nearest neighbors
# # _, nearest_indices = torch.min(distance_matrix, dim=-1)

# # # get the PCA values for the nearest neighbors
# # pca_values_nfz = pca_matrix[nearest_indices]

# # pca_values = torch.empty(
# #     (len(xyz), pca_matrix.shape[-1]), device=mp.master_pattern.device
# # )

# # pca_values[in_fz] = pca_matrix
# # pca_values[~in_fz] = pca_values_nfz

# # # multiply by -1 if the mean cube value is negative
# # mean_cube_lt_0 = (pca_values**3).mean(dim=0) < 0
# # pca_values[:, mean_cube_lt_0] *= -1

# # # now make a gif showing the first 10 eigenvectors
# # fig, ax = plt.subplots()
# # ims = []
# # for i in range(64):
# #     im = ax.imshow(
# #         pca_values[: (pca_resolution_pixels * pca_resolution_pixels), -(i + 1)]
# #         .view(pca_resolution_pixels, pca_resolution_pixels)
# #         .cpu()
# #         .numpy(),
# #     )
# #     ims.append([im])

# # ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=100)

# # # save gif
# # ani.save("pca_noise.gif")

# # # make a 10x10 grid of the first 100 eigenvectors
# # fig, axs = plt.subplots(8, 8, figsize=(20, 20))

# # for i in range(64):
# #     ax = axs[i // 8, i % 8]
# #     ax.imshow(
# #         pca_values[: (pca_resolution_pixels * pca_resolution_pixels), -(i + 1)]
# #         .view(pca_resolution_pixels, pca_resolution_pixels)
# #         .cpu()
# #         .numpy(),
# #     )
# #     ax.axis("off")

# # plt.tight_layout()

# # # save image
# # plt.savefig("pca_noise.png")


# # # do another plot with eigenvectors 1, 2, 4, 8, 16, 32, 64, 128
# # fig, axs = plt.subplots(3, 3, figsize=(10, 10))

# # for i, ax in enumerate(axs.flat):
# #     ax.imshow(
# #         pca_values[: (pca_resolution_pixels * pca_resolution_pixels), -(2**i)]
# #         .view(pca_resolution_pixels, pca_resolution_pixels)
# #         .cpu()
# #         .numpy(),
# #     )
# #     ax.axis("off")
# #     ax.set_title(f"PC {2**i}")

# # plt.tight_layout()
# # plt.savefig("pca_noise_2.png")
