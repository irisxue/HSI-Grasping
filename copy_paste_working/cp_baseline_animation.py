import torch
import numpy as np
import smplx
import torch.optim as optim
import os
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from body_visualizer.tools.vis_tools import render_smpl_params
from body_visualizer.tools.vis_tools import show_image
from human_body_prior.body_model.body_model import BodyModel
import trimesh
import json
from smplx.lbs import batch_rodrigues
import mano
from mano.utils import Mesh
import os
from copy import copy
import pickle
from body_visualizer.tools.vis_tools import meshes_as_png
import cv2
import matplotlib.pyplot as plt


os.environ["PYOPENGL_PLATFORM"] = "osmesa" 
from OpenGL import osmesa
print(os.environ['PYOPENGL_PLATFORM'])

from os import path as osp
import sys
import numpy as np

import torch
import torch.nn as nn
from smplx.lbs import batch_rodrigues
from collections import namedtuple
from body_visualizer.mesh.mesh_viewer import MeshViewer

model_output = namedtuple('output', ['vertices', 'global_orient', 'transl'])


"""
support_dir = '/proj/vondrick/sx2335/smplx/support_data/downloads'
expr_dir = osp.join(support_dir,'vposer_v2_05') #'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
bm_fname =  osp.join(support_dir,'models/smplx/neutral/model.npz')#'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads
sample_amass_fname = osp.join(support_dir, 'amass_sample.npz')# a sample npz file from AMASS

vp, ps = load_model(expr_dir, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True)
vp = vp.to('cpu')
"""

####
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

def new_verts(obj_verts, thumb_joint_location, rhand_fullpose):
    rhand_reshaped = rhand_fullpose.reshape(1,16,3) #local orient axis angle
    rhand_matrices = torch.tensor([batch_rodrigues(rhand_reshaped.detach().squeeze()).numpy()]) #local orient rotation matrices
    glob_rhand = grab_loc2glob(rhand_matrices) #global orient rotation matrices
    thumb_orient_global = glob_rhand[0][-1] #thumb3 global orient

    new_obj_verts = np.matmul(obj_verts - thumb_joint_location, thumb_orient_global) #object vertices relative to thumb3  #changed the order....
    return new_obj_verts

#moves relative object to location of the thumb of the new pose
def move_obj(relative_obj_verts, thumb_joint_new, body_full_pose):
    body_full_pose_rotmats = torch.tensor([batch_rodrigues(body_full_pose.reshape(1,55,3).detach().squeeze()).numpy()]) #convert local orient axis angle to local orient rotation matrices
    glob_fullpose = smplx_loc2glob(body_full_pose_rotmats) #global orient rotation matrices
    new_thumb_global_orient = glob_fullpose[0][-1] #thumb3 global orient
    
    moved_vertices = np.matmul(new_thumb_global_orient, relative_obj_verts.t()).t()  + torch.tensor(thumb_joint_new)
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

    #scene = o3d.io.read_triangle_mesh(osp.join(scene_dir, scene_name + '.ply'))

    scene = trimesh.load_mesh(osp.join(scene_dir, scene_name + '.ply'))
    with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
        trans = np.array(json.load(f))

    return scene_name, fitting_dir, gender, scene, trans

#grab setup
mano_path = '/proj/vondrick/sx2335/models/mano'

def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return DotDict(npz)

def DotDict(in_dict):

    out_dict = copy(in_dict)
    for k,v in out_dict.items():
       if isinstance(v,dict):
           out_dict[k] = DotDict(v)
    return dotdict(out_dict)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def params2torch(params, dtype = torch.float32):
    return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}

to_cpu = lambda tensor: tensor.detach().cpu().numpy()

class ObjectModel(nn.Module):

    def __init__(self,
                 v_template,
                 batch_size=1,
                 dtype=torch.float32):

        super(ObjectModel, self).__init__()
        self.dtype = dtype
        # Mean template vertices
        v_template = np.repeat(v_template[np.newaxis], batch_size, axis=0)
        self.register_buffer('v_template', torch.tensor(v_template, dtype=dtype))

        transl = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('transl', nn.Parameter(transl, requires_grad=True))

        global_orient = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('global_orient', nn.Parameter(global_orient, requires_grad=True))

        self.batch_size = batch_size


    def forward(self, global_orient=None, transl=None, v_template=None, **kwargs):
        if global_orient is None:
            global_orient = self.global_orient
        if transl is None:
            transl = self.transl
        if v_template is None:
            v_template = self.v_template

        rot_mats = batch_rodrigues(global_orient.view(-1, 3)).view([self.batch_size, 3, 3])

        vertices = torch.matmul(v_template, rot_mats) + transl.unsqueeze(dim=1)

        output = model_output(vertices=vertices,
                              global_orient=global_orient,
                              transl=transl)

        return output

###

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



########

def prox_replace_hand(gender, sbj_vtemp, fitting_dir, seq_data, prox_frame, grab_frame, trans):
    model = smplx.create(model_folder, model_type='smplx',
                         gender=gender, 
                         #ext='npz',
                         num_pca_comps=45,
                         #create_global_orient=True,
                         #create_body_pose=True,
                         #create_betas=True,
                         #create_left_hand_pose=True,
                         #create_right_hand_pose=True,
                         #create_expression=True,
                         #create_jaw_pose=True,
                         #create_leye_pose=True,
                         #create_reye_pose=True,
                         #create_transl=True,
                         use_pca = False,
                         flat_hand_mean = True,
                         v_template = sbj_vtemp
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

    body = trimesh.Trimesh(vertices, model.faces)

    body = body.apply_transform(trans)

    prox_thumb_joint = output_body.joints[0][54].detach().numpy()  #thumb3 
    prox_global_orient = output_body.global_orient
    relative_orient = torch.tensor([batch_rodrigues(body_full_pose.reshape(1,55,3)[:,0:24].detach().squeeze()).numpy()])
    return body, prox_thumb_joint, prox_global_orient, relative_orient, body_full_pose


def new_object_location(verts_obj, thumb_joint, rhand_global_orient, rhand_fullpose, prox_global_orient, relative_orient, prox_thumb_joint, body_full_pose, trans, obj_faces):
    relative_obj_verts = new_verts(verts_obj, thumb_joint.detach().numpy(), rhand_fullpose)
    new_obj_loc = move_obj(relative_obj_verts, prox_thumb_joint, body_full_pose)

    obj = trimesh.Trimesh (new_obj_loc, obj_faces)
    obj = obj.apply_transform(trans)
    return new_obj_loc, obj

def meshes_as_png(meshes, outpath=None, view_angles=[0]):

    imw = 5000
    imh = 5000
    mv = MeshViewer(imh, imw)
    #mv.set_cam_trans([0, -.5, 1.75])
    #mv.set_cam_trans([1, 1, 1.75])
    #mv.set_cam_trans([1, 1, 3])
    mv.set_cam_trans([0.5, 0, 3])

    images = np.zeros([len(meshes), len(view_angles), 1, imw, imh, 3])
    for mIdx, mesh in enumerate(meshes):
        for rId, angle in enumerate(view_angles):
            if angle != 0: mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(angle), (0, 1, 0)))
            mv.set_meshes([mesh], group_name='static')
            images[mIdx, rId, 0] = cv2.cvtColor(mv.render(render_wireframe=False), cv2.COLOR_BGR2RGB)
            if angle != 0: mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-angle), (0, 1, 0)))

    if outpath is not None: imagearray2file(images, outpath)
    return images


def imagearray2file(img_array, outpath=None, fps=30):
    if outpath is not None:
        outdir = os.path.dirname(outpath)
        if not os.path.exists(outdir): os.makedirs(outdir)

    if not isinstance(img_array, np.ndarray) or img_array.ndim < 6:
        raise ValueError('img_array should be a numpy array of shape RxCxTxwidthxheightx3')

    R, C, T, img_h, img_w, img_c = img_array.shape

    out_images = []
    for tIdx in range(T):
        row_images = []
        for rIdx in range(R):
            col_images = []
            for cIdx in range(C):   
                col_images.append(img_array[rIdx, cIdx, tIdx])
            row_images.append(np.hstack(col_images))
        t_image = np.vstack(row_images)
        out_images.append(t_image)

    if outpath is not None:
        ext = outpath.split('.')[-1]
        if ext in ['png', 'jpeg', 'jpg']:
            for tIdx in range(T):
                if T > 1:
                    cur_outpath = outpath.replace('.%s'%ext, '_%03d.%s'%(tIdx, ext))
                else:
                    cur_outpath = outpath
                #out_images[tIdx] = cv2.normalize(out_images[tIdx], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
                #img = cv2.cvtColor(out_images[tIdx], cv2.COLOR_BGR2RGB)
                img = out_images[tIdx]
                cv2.imwrite(cur_outpath, img)
                while not os.path.exists(cur_outpath): continue  # wait until the snapshot is written to the disk
        elif ext == 'gif':
            import imageio
            with imageio.get_writer(outpath, mode='I', duration = fps) as writer:
                for tIdx in range(T):
                    img = out_images[tIdx].astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    writer.append_data(img)
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video = cv2.VideoWriter(outpath, fourcc, fps, (img_w, img_h), True)
            for tIdx in range(T):
                img = out_images[tIdx].astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video.write(img)

            video.release()
            cv2.destroyAllWindows()
        elif ext == 'mp4':
            #
            # from moviepy.editor import ImageSequenceClip
            # animation = ImageSequenceClip(out_images, fps=fps)
            # animation.write_videofile(outpath, verbose=False)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(outpath, fourcc, fps, (img_w, img_h), True)
            for tIdx in range(T):
                img = out_images[tIdx].astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video.write(img)

            video.release()
            try:
                cv2.destroyAllWindows()
            except:
                pass

    return out_images

grab_frame = 1000
scene_name, fitting_dir, gender, scene, trans = prox_setup(
    fitting_dir = '/proj/vondrick3/datasets/PROX/qualitative_prox_dataset/PROXD/BasementSittingBooth_00142_01')
seq_data, verts_obj, thumb_joint, rhand_global_orient, rhand_fullpose, obj_faces, sbj_vtemp, thumb_orient_global = grab_setup(
    '/proj/vondrick3/datasets/GRAB/grab/s1/waterbottle_pour_1.npz', grab_frame)


all_meshes = []
for prox_frame in range(275, 375):
    body, prox_thumb_joint, prox_global_orient, relative_orient, body_full_pose = prox_replace_hand(gender, sbj_vtemp, 
                                                                                                    fitting_dir, seq_data, prox_frame, grab_frame, trans)
    new_obj_loc, obj = new_object_location(verts_obj, thumb_joint, rhand_global_orient, rhand_fullpose, 
                                        prox_global_orient, relative_orient, prox_thumb_joint, body_full_pose, trans, obj_faces)
    combined = trimesh.util.concatenate([obj, scene, body])
    all_meshes.append(combined)
print("loop done")
imgs = meshes_as_png(all_meshes)
print(imgs.shape)
imgs = imgs.reshape(1,1,imgs.shape[0],5000,5000,3)

gif2 = imagearray2file(imgs, '/home/sx2335/puppet/human_body_prior/tutorials/gif2.gif', fps=30)
print('gif created.')

