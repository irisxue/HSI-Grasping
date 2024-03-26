import smplx
import pickle
import torch
import numpy as np
import trimesh
import os
from GRAB.tools.objectmodel import ObjectModel
from GRAB.tools.utils import parse_npz, params2torch, to_cpu
import os
import os.path as osp
import json
import open3d as o3d
import mano
from mano.utils import Mesh
from pytorch3d import transforms
from smplx.lbs import batch_rodrigues

##smplx_loc2glob and grab_loc2glob compute the global orientations from relative orientations

smplx_parents = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
                16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
                35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52,
                53]
grab_parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]

def smplx_loc2glob(local_pose):
    bs = local_pose.shape[0]
    local_pose = local_pose.view(bs, -1, 3, 3)
    global_pose = local_pose.clone()
    for i in range(1,len(smplx_parents)):
        global_pose[:,i] = torch.matmul(global_pose[:, smplx_parents[i]], global_pose[:, i].clone())
    return global_pose.reshape(bs,-1,3,3)


def grab_loc2glob(local_pose):
    bs = local_pose.shape[0]
    local_pose = local_pose.view(bs, -1, 3, 3)
    global_pose = local_pose.clone()
    for i in range(1,len(grab_parents)):
        global_pose[:,i] = torch.matmul(global_pose[:, grab_parents[i]], global_pose[:, i].clone())
    return global_pose.reshape(bs,-1,3,3)

#returns object vertices relative to the thumb
def new_verts(obj_verts, thumb_joint_location, rhand_fullpose):
    """
    rhand_global_orient_rotation_matrix = transforms.axis_angle_to_matrix(rhand_global_orient)[0]
    rhand_reshaped = rhand_fullpose.reshape(1,16,3)
    thumb_joint_orient = torch.tensor([rhand_reshaped[0][-1].numpy()])          #thumb3
    thumb_joint_matrix = transforms.axis_angle_to_matrix(thumb_joint_orient)[0]
    thumb_orient_global = np.matmul(thumb_joint_matrix, rhand_global_orient_rotation_matrix)
    """
    rhand_reshaped = rhand_fullpose.reshape(1,16,3) #local orient axis angle
    rhand_matrices = torch.tensor([batch_rodrigues(rhand_reshaped.detach().squeeze()).numpy()]) #local orient rotation matrices
    glob_rhand = grab_loc2glob(rhand_matrices) #global orient rotation matrices
    thumb_orient_global = glob_rhand[0][-1] #thumb3 global orient

    new_obj_verts = np.matmul(obj_verts - thumb_joint_location, thumb_orient_global.T) #object vertices relative to thumb3  #changed the order....
    return new_obj_verts

#moves relative object to location of the thumb of the new pose
def move_obj(relative_obj_verts, thumb_joint_new, body_full_pose):
    """
    rhand_reshaped = body_full_pose[:, 120:].reshape(1,15,3)
    thumb_joint_orient = torch.tensor([rhand_reshaped[0][-1].numpy()])[0]       #thumb3
    thumb_joint_matrix = transforms.axis_angle_to_matrix(thumb_joint_orient)
    thumb_orient_global = np.matmul(thumb_joint_matrix, new_wrist_global_orient)
    moved_vertices = np.matmul(relative_obj_verts, thumb_orient_global)  + thumb_joint_new
    """
    body_full_pose_rotmats = torch.tensor([batch_rodrigues(body_full_pose.reshape(1,55,3).detach().squeeze()).numpy()]) #convert local orient axis angle to local orient rotation matrices
    glob_fullpose = smplx_loc2glob(body_full_pose_rotmats) #global orient rotation matrices
    new_thumb_global_orient = glob_fullpose[0][-1] #thumb3 global orient

    #I changed new_thumb_global_orient to new_thumb_global_orient.T to make 
    # the orientation of the bottle look correct, but I think it should be 
    # new_thumb_global_orient since we are going from local to global. 
    # not sure why new_thumb_global_orient doesn't work...
    moved_vertices = np.matmul(relative_obj_verts, new_thumb_global_orient.T)  + thumb_joint_new #move relative object vertices to new thumb3
    return moved_vertices

#prox setup
base_dir = '/proj/vondrick3/datasets/PROX/qualitative_prox_dataset'
model_folder = '/proj/vondrick/sx2335/smplx/support_data/downloads/models'
cam2world_dir = osp.join(base_dir, 'cam2world')
scene_dir = osp.join(base_dir, 'scenes')

def prox_setup(fitting_dir):
    recording_name = os.path.abspath(fitting_dir).split("/")[-1]
    fitting_dir = osp.join(fitting_dir, 'results')
    scene_name = recording_name.split("_")[0]
    female_subjects_ids = [162, 3452, 159, 3403]
    subject_id = int(recording_name.split('_')[1])
    if subject_id in female_subjects_ids:
        gender = 'female'
    else:
        gender = 'male'

    scene = o3d.io.read_triangle_mesh(osp.join(scene_dir, scene_name + '.ply'))
    with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
        trans = np.array(json.load(f))

    return scene_name, fitting_dir, gender, scene, trans

#grab setup
mano_path = '/proj/vondrick/sx2335/models/mano'

def grab_setup(grab_example, grab_frame):
    seq_data = parse_npz(grab_example)
    rhand_fullpose = torch.tensor([seq_data.rhand.params.fullpose[grab_frame]])
    
    rh_model = mano.load(model_type = 'mano',
                     model_path=mano_path,
                     is_rhand= True,
                     num_pca_comps=45,
                     batch_size=1,
                     flat_hand_mean=True)

    rhand_transl = torch.tensor([seq_data.rhand.params.transl[grab_frame]])
    rhand_global_orient = torch.tensor([seq_data.rhand.params.global_orient[grab_frame]]) #1 by 3

    output = rh_model(global_orient = rhand_global_orient,  #direction of wrist of the hand
                  hand_pose = rhand_fullpose,      #relative pose
                  transl = rhand_transl,
                  return_verts = True,
                  return_full_pose = True)    
    
    joints = output.joints # (should be 16 by 3) 0 is wrist, 123 pinky, ... last is for tip of thumb
    rhand_fullpose = output.full_pose 
    thumb_joint = joints[0][-1] #thumb3 

    
    #get vertices of the object
    T = seq_data.n_frames
    filename = os.path.join('/proj/vondrick3/datasets/GRAB', seq_data.object.object_mesh)
    mesh = trimesh.load(filename, process = False)
    vertices = mesh.vertices
    faces = mesh.faces
    obj_vtemp = np.array(vertices)
    obj_m = ObjectModel(v_template=obj_vtemp, batch_size=T)
    obj_parms = params2torch(seq_data.object.params)
    verts_obj = to_cpu(obj_m(**obj_parms).vertices)[grab_frame]   

    sbj_mesh = os.path.join('/proj/vondrick3/datasets/GRAB/grab', '..', seq_data.body.vtemp)
    sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)  

    thumb_orient_global = smplx_loc2glob(torch.tensor([batch_rodrigues(torch.tensor([seq_data.body.params.fullpose[grab_frame].reshape(55,3)][0])).numpy()]))[0][-1]

    return seq_data, verts_obj, thumb_joint, rhand_global_orient, rhand_fullpose, faces, sbj_vtemp, thumb_orient_global

#replace_hand
def prox_replace_hand(gender, sbj_vtemp, fitting_dir, seq_data, prox_frame, grab_frame, trans):
    model = smplx.create(model_folder, model_type='smplx',
                         gender=gender, ext='npz',
                         num_pca_comps=45,
                         create_global_orient=True,
                         create_body_pose=True,
                         create_betas=True,
                         create_left_hand_pose=True,
                         create_right_hand_pose=True,
                         create_expression=True,
                         create_jaw_pose=True,
                         create_leye_pose=True,
                         create_reye_pose=True,
                         create_transl=True,
                         #v_template = sbj_vtemp
                         )

    img_name = sorted(os.listdir(fitting_dir))[prox_frame] 
    print('viz frame {}'.format(img_name))

    with open(osp.join(fitting_dir, img_name, '000.pkl'), 'rb') as f:
        param = pickle.load(f)
    torch_param = {}
    for key in param.keys():
        torch_param[key] = torch.tensor(param[key])

    #prox
    global_orient = torch.tensor(param['global_orient'])
    body_pose = torch.tensor(param['body_pose'])
    transl = torch.tensor(param['transl'])
    camera_rotation = torch.tensor(param['camera_rotation'])
    camera_translation = torch.tensor(param['camera_translation'])
    jaw_pose = torch.tensor(param['jaw_pose'])
    leye_pose = torch.tensor(param['leye_pose'])
    reye_pose = torch.tensor(param['reye_pose'])
    expression = torch.tensor(param['expression'])
    pose_embedding = torch.tensor(param['pose_embedding'])


    #modifications from grab (replaced both lhand and rhand, but focus on rhand grasp)
    lhand_pose = torch.tensor([seq_data.lhand.params.fullpose[grab_frame]])
    rhand_pose = torch.tensor([seq_data.rhand.params.fullpose[grab_frame]])
    
    output_body = model(global_orient=global_orient,
               body_pose=body_pose,
               transl=transl,
               camera_rotation=camera_rotation,
               camera_translation = camera_translation,
               jaw_pose = jaw_pose,
               leye_pose = leye_pose,
               reye_pose= reye_pose,
               expression = expression,
               pose_embedding = pose_embedding,
               left_hand_pose = lhand_pose,
               right_hand_pose = rhand_pose,
               flat_hand_mean = True,
               return_verts=True,
               return_full_pose = True)
    
    vertices = output_body.vertices.detach().cpu().numpy().squeeze()
    body_full_pose = output_body.full_pose

    body = o3d.geometry.TriangleMesh()
    body.vertices = o3d.utility.Vector3dVector(vertices)
    body.triangles = o3d.utility.Vector3iVector(model.faces)
    body.vertex_normals = o3d.utility.Vector3dVector([])
    body.triangle_normals = o3d.utility.Vector3dVector([])
    body.compute_vertex_normals()
    body.transform(trans)

    prox_thumb_joint = output_body.joints[0][54].detach().numpy()  #thumb3 
    prox_global_orient = output_body.global_orient

    
    relative_orient = torch.tensor([batch_rodrigues(body_full_pose.reshape(1,55,3)[:,0:24].detach().squeeze()).numpy()])
    
    return body, prox_thumb_joint, prox_global_orient, relative_orient, body_full_pose

def new_object_location(verts_obj, thumb_joint, rhand_global_orient, rhand_fullpose, prox_global_orient, relative_orient, prox_thumb_joint, body_full_pose, trans, obj_faces):
    relative_obj_verts = new_verts(verts_obj, thumb_joint.detach().numpy(), rhand_fullpose)

    new_obj_loc = move_obj(relative_obj_verts, prox_thumb_joint, body_full_pose)

    obj = o3d.geometry.TriangleMesh()
    obj.vertices = o3d.utility.Vector3dVector(new_obj_loc)

    obj.triangles = o3d.utility.Vector3iVector(obj_faces)
    obj.vertex_normals = o3d.utility.Vector3dVector([])
    obj.triangle_normals = o3d.utility.Vector3dVector([])
    obj.transform(trans)

    return new_obj_loc, obj

#visualize ground truth from grab dataset
def vis_grab(seq_data, gender, verts_obj, obj_faces, grab_frame):
    n_comps = seq_data['n_comps']
    sbj_mesh = os.path.join('/proj/vondrick3/datasets/GRAB/grab', '..', seq_data.body.vtemp)
    sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)

    sbj_m = smplx.create(model_path=model_folder,
                        model_type = 'smplx',
                        gender = gender,
                        num_pca_comps = n_comps,
                        v_template = sbj_vtemp)

    sbj_parms = params2torch(seq_data.body.params)
    verts_sbj = to_cpu(sbj_m(**sbj_parms).vertices)[grab_frame]

    obj = o3d.geometry.TriangleMesh()
    obj.vertices = o3d.utility.Vector3dVector(verts_obj)
    obj.triangles = o3d.utility.Vector3iVector(obj_faces)
    obj.vertex_normals = o3d.utility.Vector3dVector([])
    obj.triangle_normals = o3d.utility.Vector3dVector([])

    body = o3d.geometry.TriangleMesh()
    body.vertices = o3d.utility.Vector3dVector(verts_sbj)
    body.triangles = o3d.utility.Vector3iVector(sbj_m.faces)
    body.vertex_normals = o3d.utility.Vector3dVector([])
    body.triangle_normals = o3d.utility.Vector3dVector([])

    o3d.visualization.draw_plotly([body,obj])

###examples

#ground truth from grab
grab_frame = 1000
scene_name, fitting_dir, gender, scene, trans = prox_setup(
    fitting_dir = '/proj/vondrick3/datasets/PROX/qualitative_prox_dataset/PROXD/BasementSittingBooth_00142_01')

seq_data, verts_obj, thumb_joint, rhand_global_orient, rhand_fullpose, obj_faces, sbj_vtemp, thumb_orient_global = grab_setup(
    '/proj/vondrick3/datasets/GRAB/grab/s1/waterbottle_pour_1.npz', grab_frame)
vis_grab(seq_data, gender, verts_obj, obj_faces, grab_frame)


#visualize prox with copy pasted hand with object
body, prox_thumb_joint, prox_global_orient, relative_orient, body_full_pose = prox_replace_hand(gender, sbj_vtemp, 
                                                                                                fitting_dir, seq_data, 1500, 1000, trans)

new_obj_loc, obj = new_object_location(verts_obj, thumb_joint, rhand_global_orient, rhand_fullpose, 
                                       prox_global_orient, relative_orient, prox_thumb_joint, body_full_pose, trans, obj_faces)

o3d.visualization.draw_plotly([body, obj])