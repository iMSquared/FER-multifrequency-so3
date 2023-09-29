import numpy as np
import open3d as o3d
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import Optional, List



def visualize_point_cloud(pcds:list, 
                          lower_lim=-0.25, 
                          upper_lim=0.25, 
                          save:bool=False, 
                          save_path:Optional[str]=None):
    '''
    Visualize the numpy point cloud
    '''

    if save:
        plt.switch_backend('Agg') # tkinter keeps crashing... :(

    colors = ["Red", "Blue", "Green", "tab:orange", "magenta", "tab:blue", "tab:purple", "tab:olive"]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([lower_lim, upper_lim])
    ax.set_ylim([lower_lim, upper_lim])
    ax.set_zlim([lower_lim, upper_lim])

    # Plot points
    for i, pcd in enumerate(pcds):
        ax.scatter(pcd[:,0], pcd[:,1], pcd[:,2], s=0.2, c=colors[i % len(colors)])

    if not save:
        plt.show()
    else:
        fig.savefig(save_path)
        plt.close(fig)


def visualize_registration(pcd1_in, pcd2_in, pcd2_reg, lower_lim=-1., upper_lim=1., block=True):

    color1 = "Red"
    color2 = "Blue"
    
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([lower_lim, upper_lim])
    ax1.set_ylim([lower_lim, upper_lim])
    ax1.set_zlim([lower_lim, upper_lim])
    ax1.scatter(pcd1_in[:,0], pcd1_in[:,1], pcd1_in[:,2], s=1.0, c=color1)
    ax1.scatter(pcd2_in[:,0], pcd2_in[:,1], pcd2_in[:,2], s=1.0, c=color2)
    

    ax2 = fig.add_subplot(1,2,2,projection='3d')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim([lower_lim, upper_lim])
    ax2.set_ylim([lower_lim, upper_lim])
    ax2.set_zlim([lower_lim, upper_lim])
    ax2.scatter(pcd1_in[:,0], pcd1_in[:,1], pcd1_in[:,2], s=1.0, c=color1)
    ax2.scatter(pcd2_reg[:,0], pcd2_reg[:,1], pcd2_reg[:,2], s=1.0, c=color2)
    plt.show(block=block)


def visualize_point_cloud_render(pcds: List, ball_radius: float = 0.035):

    # Nice looking pastel colors
    colors = {
        "Red": (1, 0.3118, 0.3118), 
        "Blue": (0.2784, 0.4196, 0.6275), 
        "Green": (0.5961, 0.9843, 0.5961), 
        "tab:orange": (1, 0.7059, 0.5098), 
        "magenta": (0.9569, 0.6039, 0.7608), 
        "tab:blue": (0.5294, 0.8078, 0.9804), 
        "tab:purple": (0.6863, 0.6039, 0.8471), 
        "tab:olive":  (0.6784, 0.7529, 0.4902)}

    def point_to_mesh(pcd):
        mesh_pcd = o3d.geometry.TriangleMesh()
        for point in pcd:
            vector = np.reshape(point, (3, 1))
            ball = o3d.geometry.TriangleMesh.create_sphere(ball_radius)
            ball.translate(vector)
            mesh_pcd += ball
        return mesh_pcd
    
    meshes = []
    for i, pcd in enumerate(pcds):
        mesh = point_to_mesh(pcd)
        c = list(colors.keys())[i % len(colors)]
        mesh.paint_uniform_color(colors[c])
        mesh.compute_vertex_normals()
        meshes.append(mesh)
    
    # for i, m in enumerate(meshes):
    #     o3d.io.write_triangle_mesh(f"testmesh_{i}.ply", m)
    
    # # Show (y-up)
    # o3d.visualization.draw(
    #     geometry      = meshes,
    #     width         = 1000,
    #     height        = 1000,
    #     lookat        = (0,0,0),
    #     eye           = (0,2,0),
    #     up            = (0,0,-1),
    #     show_skybox   = False,
    #     bg_color      = (1.5, 1.5, 1.5, 1.5)) # over the limit for the true white
    o3d.visualization.draw(
        geometry      = meshes,
        width         = 1000,
        height        = 1000,
        lookat        = (0,0,0),
        eye           = (0,0,2),
        up            = (0,1,0),
        show_skybox   = False,
        bg_color      = (1.5, 1.5, 1.5, 1.5)) # over the limit for the true white
