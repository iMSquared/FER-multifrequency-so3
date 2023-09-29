# %%
from __future__ import annotations
import numpy as np
import numpy.typing as npt
import scipy
from scipy.spatial.transform import Rotation as sciR
import torch
import einops


def commutator(A: npt.NDArray|torch.Tensor, B: npt.NDArray|torch.Tensor) -> npt.NDArray|torch.Tensor:
    """Simple commutator operator"""
    return A@B-B@A


def random_skew(n: int, rng: np.random.Generator, b: int = 1) -> npt.NDArray:
    """???

    Args:
        n (int): Size of square matrix
        rng (np.random.Generator): Numpy random generator
        b (int, optional): Batch size. Defaults to 1.

    Returns:
        randQ (npt.NDArray): Similiarity transformation matrix
    """
    randQ = rng.uniform(-10,10, size=(b,n,n))
    randQ = randQ - np.transpose(randQ, (0,2,1))
    if b==1:
        randQ = np.squeeze(randQ, 0)
    return randQ


def random_aligned_skew(l: int, rng: np.random.Generator, eigen_limit: int = None) -> npt.NDArray:
    """???

    Args:
        l (int): Type
        rng (np.random.Generator): Numpy random generator.
        eigen_limit (int, optional): ???. Defaults to None.

    Returns:
        rec2 (npt.NDArray): ???
    """
    rskew = random_skew(n=int(2*l+1), rng=rng)
    s, ev = np.linalg.eig(rskew)
    ev = ev[:,np.argsort(s.imag)]
    s = s[np.argsort(s.imag)]

    # reduced order test
    diag_vec = np.zeros((2*l+1,))
    if eigen_limit is None:
        eigen_limit = l
    diag_vec[l-eigen_limit:l+eigen_limit+1] = np.linspace(-eigen_limit, eigen_limit, int(2*eigen_limit+1))
    rec2 = (ev@np.diag(diag_vec*1j)@np.conj(ev.T)).real

    return rec2


def get_basis_nullspace(A: npt.NDArray):
    """???"""
    V,S,VT = np.linalg.svd(A)
    basis = VT[np.where(np.abs(S) < 1e-7)]
    return basis


def random_generators(l: int, rng: npt.NDArray) -> npt.NDArray:
    """???

    Args:
        l (int): Type
        rng (npt.NDArray): Numpy random generator.

    Returns:
        npt.NDArray: J shape=[2*l+1, 2*l+1]
    """
    if l==0:
        return np.array([[[0]],[[0]],[[0]]])
    # rng = np.random.default_rng(seed+42)
    Jz = random_aligned_skew(l, rng).astype(np.float64)
    n = Jz.shape[-1]
    testA = (np.kron(np.eye(n), Jz)-np.kron(Jz.T, np.eye(n))) @ (np.kron(Jz.T,np.eye(n))-np.kron(np.eye(n), Jz)) - np.eye(n*n)
    Jyvec_basis = get_basis_nullspace(testA)

    assert Jyvec_basis.shape[0] >= 1

    Jxvec_basis = (np.kron(Jz.T,np.eye(n))-np.kron(np.eye(n), Jz))@Jyvec_basis[...,None]
    Jxvec_basis = np.squeeze(Jxvec_basis, axis=-1)
    
    # diagonal condition
    Jx_basis = Jxvec_basis.reshape(Jxvec_basis.shape[0], n, n)
    Jy_basis = Jyvec_basis.reshape(Jyvec_basis.shape[0], n, n)

    # further reduce
    Jxdiag = []
    Jydiag = []
    for i in range(Jy_basis.shape[0]):
        Jxdiag.append(np.diag(Jx_basis[i]))
        Jydiag.append(np.diag(Jy_basis[i]))
    Jxdiag = np.stack(Jxdiag, -1)
    Jydiag = np.stack(Jydiag, -1)
    conmat = np.concatenate([Jxdiag, Jydiag], 0)
    alpha_basis = get_basis_nullspace(conmat)

    # optimization reduced space
    def loss_func(beta):
        alpha = np.sum(beta[...,None] * alpha_basis, -2)
        Jx = np.sum(Jx_basis*alpha[...,None,None], -3)
        Jy = np.sum(Jy_basis*alpha[...,None,None], -3)
        return np.sum((Jx@Jy-Jy@Jx-Jz)**2, (-1,-2))

    # CEM iterations
    total_n = 2000
    nd = alpha_basis.shape[-2]
    beta = rng.uniform(-2,2,size=(total_n, nd))
    itr_no = 200
    cem_ratio=0.15
    for i in range(itr_no):
        loss_ = loss_func(beta)
        topkargs = np.argsort(loss_)[:int(total_n*cem_ratio)]
        mean = np.mean(beta[topkargs], 0)
        std = np.std(beta[topkargs], 0).clip(1e-7)
        beta = mean + rng.normal(size=(total_n,nd)) * std
    loss_ = loss_func(beta)
    beta = beta[np.argmin(loss_)]
    alpha = np.sum(beta[...,None] * alpha_basis, -2)
    Jx = np.sum(Jx_basis*alpha[...,None,None], -3)
    Jy = np.sum(Jy_basis*alpha[...,None,None], -3)

    print(f'entire loss: {loss_func(beta)}, {np.abs(commutator(Jx,Jy) - Jz).max()}, {np.abs(commutator(Jy,Jz) - Jx).max()}, {np.abs(commutator(Jz,Jx) - Jy).max()}')

    if np.abs(commutator(Jx,Jy) - Jz).max() > 4e-8:
        print(f'rerun order: {l}')
        return random_generators(l, rng)

    ring_permutation_rand = rng.integers(0,3)
    if ring_permutation_rand==0:
        return np.stack([Jx, Jy, Jz], 0)
    elif ring_permutation_rand==1:
        return np.stack([Jz, Jx, Jy], 0)
    elif ring_permutation_rand==2:
        return np.stack([Jy, Jz, Jx], 0)


def Rm2q(Rm: torch.Tensor) -> torch.Tensor:
    """Rotation matrix to quaternion"""
    Rm = einops.rearrange(Rm, '... i j -> ... j i')
    con1 = (Rm[...,2,2] < 0) & (Rm[...,0,0] > Rm[...,1,1])
    con2 = (Rm[...,2,2] < 0) & (Rm[...,0,0] <= Rm[...,1,1])
    con3 = (Rm[...,2,2] >= 0) & (Rm[...,0,0] < -Rm[...,1,1])
    con4 = (Rm[...,2,2] >= 0) & (Rm[...,0,0] >= -Rm[...,1,1]) 

    t1 = 1 + Rm[...,0,0] - Rm[...,1,1] - Rm[...,2,2]
    t2 = 1 - Rm[...,0,0] + Rm[...,1,1] - Rm[...,2,2]
    t3 = 1 - Rm[...,0,0] - Rm[...,1,1] + Rm[...,2,2]
    t4 = 1 + Rm[...,0,0] + Rm[...,1,1] + Rm[...,2,2]

    q1 = torch.stack([t1, Rm[...,0,1]+Rm[...,1,0], Rm[...,2,0]+Rm[...,0,2], Rm[...,1,2]-Rm[...,2,1]], dim=-1) / torch.sqrt(t1.clip(1e-7))[...,None]
    q2 = torch.stack([Rm[...,0,1]+Rm[...,1,0], t2, Rm[...,1,2]+Rm[...,2,1], Rm[...,2,0]-Rm[...,0,2]], dim=-1) / torch.sqrt(t2.clip(1e-7))[...,None]
    q3 = torch.stack([Rm[...,2,0]+Rm[...,0,2], Rm[...,1,2]+Rm[...,2,1], t3, Rm[...,0,1]-Rm[...,1,0]], dim=-1) / torch.sqrt(t3.clip(1e-7))[...,None]
    q4 = torch.stack([Rm[...,1,2]-Rm[...,2,1], Rm[...,2,0]-Rm[...,0,2], Rm[...,0,1]-Rm[...,1,0], t4], dim=-1) / torch.sqrt(t4.clip(1e-7))[...,None]
 
    q = torch.zeros(Rm.shape[:-2]+(4,)).to(Rm.device)
    q = torch.where(con1[...,None], q1, q)
    q = torch.where(con2[...,None], q2, q)
    q = torch.where(con3[...,None], q3, q)
    q = torch.where(con4[...,None], q4, q)
    q *= 0.5

    return q


def qlog(q: torch.Tensor) -> torch.Tensor:
    """???"""
    alpha = torch.arccos(q[...,3:])
    sinalpha = torch.sin(alpha)
    abssinalpha = torch.maximum(torch.abs(sinalpha), torch.tensor(1e-6, device=q.device))
    n = q[...,:3]/(abssinalpha*torch.sign(sinalpha))
    return torch.where(torch.abs(q[...,3:])<1-1e-6, n*alpha, torch.zeros_like(n))


def safe_norm(x: torch.Tensor, axis: int, keepdims: bool = False, eps: float = 0.0) -> torch.Tensor:
    """Get norm of the tensor

    Args:
        x (torch.Tensor): Tensor to get the norm
        axis (int): Dimension to get the norm along
        keepdims (bool, optional): Defaults to False.
        eps (float, optional): Defaults to 0.0.

    Returns:
        torch.Tensor: Norm
    """
    is_zero = torch.all(torch.isclose(x,torch.tensor(0., device=x.device)), dim=axis, keepdim=True)
    # temporarily swap x with ones if is_zero, then swap back
    x = torch.where(is_zero, torch.ones_like(x), x)
    n = torch.norm(x, dim=axis, keepdim=keepdims)
    n = torch.where(is_zero if keepdims else torch.squeeze(is_zero, dim=-1), 0., n)
    return n.clip(eps)


def normalize(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize the last dimension of a tensor"""
    # return vec/(safe_norm(vec, axis=-1, keepdims=True, eps=eps) + 1e-8)
    return vec/safe_norm(vec, axis=-1, keepdims=True, eps=eps)


def qrand(outer_shape: torch.Tensor) -> torch.Tensor:
    """???"""
    return normalize(torch.normal(0., 1., outer_shape + (4,)))


def q2aa(q: torch.Tensor) -> torch.Tensor:
    """Quaternion to axis-angle"""
    return 2*qlog(q)


def aa2q(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle to quaternion
    
    Args:
        aa (npt.NDArray|torch.Tensor): Axis-angle, shape=[..., 3]
    
    Returns:
        torch.Tensor: Quaternion
    """
    return qexp(aa*0.5)


def q2R(q):
    """Quaternion to rotation matrix"""
    i,j,k,r = torch.split(q, 1, dim=-1)
    R1 = torch.concat([1-2*(j**2+k**2), 2*(i*j-k*r), 2*(i*k+j*r)], dim=-1)
    R2 = torch.concat([2*(i*j+k*r), 1-2*(i**2+k**2), 2*(j*k-i*r)], dim=-1)
    R3 = torch.concat([2*(i*k-j*r), 2*(j*k+i*r), 1-2*(i**2+j**2)], dim=-1)
    return torch.stack([R1,R2,R3], dim=-2)


def qexp(logq):
    """???"""
    if isinstance(logq, np.ndarray):
        alpha = np.linalg.norm(logq[...,:3], axis=-1, keepdims=True)
        alpha = np.maximum(alpha, 1e-6)
        return np.concatenate([logq[...,:3]/alpha*np.sin(alpha), np.cos(alpha)], axis=-1)
    else:
        alpha = safe_norm(logq[...,:3], axis=-1, keepdims=True)
        alpha = torch.maximum(alpha, torch.tensor(1e-6, device=alpha.device))
        return torch.concat([logq[...,:3]/alpha*torch.sin(alpha), torch.cos(alpha)], dim=-1)


def qmulti(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions
    
    `q1` on the right of `q2`.
    """
    b,c,d,a = torch.split(q1, 1, dim=-1)
    f,g,h,e = torch.split(q2, 1, dim=-1)
    w,x,y,z = a*e-b*f-c*g-d*h, a*f+b*e+c*h-d*g, a*g-b*h+c*e+d*f, a*h+b*g-c*f+d*e
    return torch.concat([x,y,z,w], dim=-1)


def qinv(q: torch.Tensor):
    """Get inverse quaternion"""
    x,y,z,w = torch.split(q, 1, dim=-1)
    return torch.concat([-x,-y,-z,w], dim=-1)


def qaction(quat, pos):
    """???"""
    return qmulti(qmulti(quat, torch.concat([pos, torch.zeros_like(pos[...,:1])], dim=-1)), qinv(quat))[...,:3]


def custom_rotm(l, R, Js):
    """???"""
    if l==1:
        return R
    w = q2aa(Rm2q(R))
    # w = np.where(w[...,0]>0,w, np.array([2*np.pi,0,0])+w)
    return torch.matrix_exp(torch.sum(w[...,None,None] * Js[l], axis=-3))


def custom_rotmV2(l, Rm, D_basis):
    if l==0:
        return torch.tensor([[1.]]).to(Rm.device)
    if l==1:
        return Rm
    Rm_flat = einops.rearrange(Rm, '... i j -> ... (i j)')
    Rm_concat = einops.rearrange((Rm_flat[...,None]*Rm_flat[...,None,:]), '... i j -> ... (i j)')
    if l==3:
        Rm_concat = einops.rearrange((Rm_concat[...,None]*Rm_flat[...,None,:]), '... i j -> ... (i j)')
    
    return einops.rearrange(torch.einsum('...i,...ik', Rm_concat, D_basis[l]), '... (r i)-> ... r i', r=2*l+1)


def rand_matrix(n):
    """???"""
    quat = np.random.normal(size=(n,4)).astype(dtype=np.float64)
    quat = quat/np.linalg.norm(quat, axis=-1, keepdims=True)
    return sciR.from_quat(quat).as_matrix()


def make_Rp(xin, base_axis='z', normalize=True):
    """???"""
    if normalize:
        mag = torch.norm(xin, din=-1, keepdim=True)
    else:
        mag = 1

    if base_axis=='z':
        z = xin/mag
        y = torch.normal(0,1,size=xin.shape).to(xin.device)
        # y = np.random.normal(size=xin.shape)
        x = torch.cross(y, z)
        x = x/torch.norm(x, dim=-1, keepdim=True)
        y= torch.cross(z, x)
        y = y/torch.norm(y, dim=-1, keepdim=True)
        Rp = torch.stack([x,y,z], axis=-1)
    elif base_axis=='x':
        x = xin/mag
        z = np.random.default_rng(0).normal(size=xin.shape)
        y = np.cross(x, z)
        y = y/np.linalg.norm(y, axis=-1, keepdims=True)
        z= np.cross(x, y)
        z = z/np.linalg.norm(z, axis=-1, keepdims=True)
        Rp = np.stack([x,y,z], axis=-1)
    elif base_axis=='y':
        y = xin/mag
        z = np.random.default_rng(0).normal(size=xin.shape)
        x = np.cross(y, z)
        x = x/np.linalg.norm(x, axis=-1, keepdims=True)
        z= np.cross(x, y)
        z = z/np.linalg.norm(z, axis=-1, keepdims=True)
        Rp = np.stack([x,y,z], axis=-1)
    return Rp


def Y_func(l, xin, rot_configs):
    """???"""
    if l==1:
        return xin
    mag = torch.norm(xin, dim=-1, keepdim=True)
    if l==0:
        return mag
    xnm = xin / mag.clip(1e-6)
    Rps = make_Rp(xnm, base_axis='z', normalize=False)
    Dm = custom_rotm(l, Rps, rot_configs['Js'])
    return mag * torch.einsum('...ij,...j', Dm, rot_configs['Y_basis'][l])


def Y_func_V2(l, x, rot_configs):
    """???"""
    if l==0:
        return torch.ones_like(x[...,:1])
    if l==1:
        return x
    mag = torch.norm(x, dim=-1, keepdim=True) + 1e-6
    xhat = x/mag
    xxhat = (xhat[...,None]*xhat[...,None,:]).reshape(*xhat.shape[:-1],-1)
    xhat_blinear = torch.concat([xhat*xhat, xxhat], dim=-1) # for order 2
    if l==2:
        if rot_configs['constant_scale']:
            return mag*xhat_blinear@rot_configs['Y_linear_coef'][2]
        else:
            return mag**2*xhat_blinear@rot_configs['Y_linear_coef'][2]
    if l==3:
        xhat_blinear_sq = (xhat_blinear[...,None]*xhat[...,None,:]).reshape(*xhat.shape[:-1],-1)  # for order 3
        if rot_configs['constant_scale']:
            return mag*xhat_blinear_sq@rot_configs['Y_linear_coef'][3]
        else:
            return mag**3*xhat_blinear_sq@rot_configs['Y_linear_coef'][3]
    if l==4:
        xhat_blinear_sqsq = (xhat_blinear[...,None]*xhat_blinear[...,None,:]).reshape(*xhat.shape[:-1],-1)
        if rot_configs['constant_scale']:
            return mag*xhat_blinear_sqsq@rot_configs['Y_linear_coef'][4]
        else:
            return mag**4*xhat_blinear_sqsq@rot_configs['Y_linear_coef'][4]
    else:
        return Y_func(l, x, rot_configs)


def Y_func_V2_np(l, x, rot_configs):
    """???"""
    if l==0:
        return np.ones_like(x[...,:1])
    if l==1:
        return x
    mag = np.linalg.norm(x, axis=-1, keepdims=True)
    xhat = x/mag
    xxhat = (xhat[...,None]*xhat[...,None,:]).reshape(*xhat.shape[:-1],-1)
    xhat_blinear = np.concatenate([xhat*xhat, xxhat], axis=-1) # for order 2
    if l==2:
        return mag*xhat_blinear@rot_configs['Y_linear_coef'][2]
    if l==3:
        xhat_blinear_sq = (xhat_blinear[...,None]*xhat[...,None,:]).reshape(*xhat.shape[:-1],-1)  # for order 3
        return mag*xhat_blinear_sq@rot_configs['Y_linear_coef'][3]
    if l==4:
        xhat_blinear_sqsq = (xhat_blinear[...,None]*xhat_blinear[...,None,:]).reshape(*xhat.shape[:-1],-1)
        return mag*xhat_blinear_sqsq@rot_configs['Y_linear_coef'][4]
    else:
        return Y_func(l, x, rot_configs)


def create_Y_basis(random_generators_list_):
    """???"""
    eig_vecs_list_ = {}
    for k_ in random_generators_list_:
        eig_res_ = np.linalg.eig(random_generators_list_[k_][2])
        eig_vec_for_zero = np.squeeze(eig_res_[1][:,np.where(np.abs(eig_res_[0])<1e-9)[0]], axis=-1)
        assert np.all(np.abs(eig_vec_for_zero.imag)<1e-8)
        eig_vecs_list_[k_] = eig_vec_for_zero.real
    return eig_vecs_list_


def custom_rotm_np(l, R, Js):
    """???"""
    w = sciR.from_matrix(R).as_rotvec()
    return scipy.linalg.expm(np.sum(w[...,None,None] * Js[l], axis=-3))


def Y_func_np(l, xin, rot_configs):
    """???"""
    mag = np.linalg.norm(xin, axis=-1, keepdims=True)
    xnm = xin / mag.clip(1e-6)
    
    # make Rps
    z = xnm
    y = np.random.default_rng(0).normal(size=xnm.shape)
    x = np.cross(y, z)
    x = x/np.linalg.norm(x, axis=-1, keepdims=True)
    y= np.cross(z, x)
    y = y/np.linalg.norm(y, axis=-1, keepdims=True)
    Rps = np.stack([x,y,z], axis=-1)

    Dm = custom_rotm_np(l, Rps, rot_configs['Js'])
    return mag * np.einsum('...ij,...j', Dm, rot_configs['Y_basis'][l])


def calculate_Y_func_linear_coef_np(rot_configs, rng, device):
    """???"""
    ns_ = 100000
    x = rng.normal(size=(ns_,3)).astype(np.float64)

    mag = np.linalg.norm(x, axis=-1, keepdims=True)
    xhat = x/mag
    xxhat = (xhat[...,None]*xhat[...,None,:]).reshape(xhat.shape[0],-1)
    xhat_blinear = np.concatenate([xhat*xhat, xxhat], axis=-1) # for order 2
    xhat_blinear_sq = (xhat_blinear[...,None]*xhat[...,None,:]).reshape(xhat_blinear.shape[0],-1)  # for order 3
    xhat_blinear_sqsq = (xhat_blinear[...,None]*xhat_blinear[...,None,:]).reshape(xhat_blinear.shape[0],-1)  # for order 3
    coef_order2 = np.linalg.pinv(mag*xhat_blinear)@Y_func_np(2, x, rot_configs).astype(np.float64)
    coef_order3 = np.linalg.pinv(mag*xhat_blinear_sq)@Y_func_np(3, x, rot_configs).astype(np.float64)
    # coef_order4 = np.linalg.pinv(mag*xhat_blinear_sqsq)@Y_func_np(4, x, rot_configs).astype(np.float64)
    
    linear_coef = {2:coef_order2, 
                   3:coef_order3,
                    # 4:coef_order4
                    }
    rot_configs = {**rot_configs, 'Y_linear_coef':linear_coef}
    print(np.abs(Y_func_V2_np(2, x, rot_configs)-Y_func_np(2, x, rot_configs)).max())
    print(np.abs(Y_func_V2_np(3, x, rot_configs)-Y_func_np(3, x, rot_configs)).max())
    # print(np.abs(Y_func_V2_np(4, x, rot_configs)-Y_func_np(4, x, rot_configs)).max())

    return {2:torch.tensor(coef_order2, dtype=torch.float32, device=device), 
                   3:torch.tensor(coef_order3, dtype=torch.float32, device=device),
                #    4:torch.tensor(coef_order4, dtype=torch.float32, device=device)
                   }


def calculate_Y_func_linear_coef(Y_func_, device):
    """???"""
    ns_ = 100000
    x = np.random.normal(size=(ns_,3)).astype(np.float64)
    xtc = torch.tensor(x, dtype=torch.float32, device=device)

    mag = np.linalg.norm(x, axis=-1, keepdims=True)
    xhat = x/mag
    xxhat = (xhat[...,None]*xhat[...,None,:]).reshape(xhat.shape[0],-1)
    xhat_blinear = np.concatenate([xhat*xhat, xxhat], axis=-1) # for order 2
    xhat_blinear_sq = (xhat_blinear[...,None]*xhat[...,None,:]).reshape(xhat_blinear.shape[0],-1)  # for order 3
    # xhat_blinear_sqsq = (xhat_blinear[...,None]*xhat_blinear[...,None,:]).reshape(xhat_blinear.shape[0],-1)  # for order 3
    coef_order2 = np.linalg.pinv(mag*xhat_blinear)@Y_func_(2, xtc).to('cpu').numpy().astype(np.float64)
    coef_order3 = np.linalg.pinv(mag*xhat_blinear_sq)@Y_func_(3, xtc).to('cpu').numpy().astype(np.float64)
    # coef_order4 = np.linalg.pinv(mag*xhat_blinear_sqsq)@Y_func_(4, xtc).to('cpu').numpy().astype(np.float64)
    
    linear_coef = {2:torch.tensor(coef_order2, dtype=torch.float32, device=device), 
                   3:torch.tensor(coef_order3, dtype=torch.float32, device=device),
                #    4:torch.tensor(coef_order4, dtype=torch.float32, device=device)
                   }
    rot_configs = {'Y_linear_coef':linear_coef}
    
    print(torch.abs(Y_func_V2(2, xtc, rot_configs)-Y_func_(2, xtc)).max())
    print(torch.abs(Y_func_V2(3, xtc, rot_configs)-Y_func_(3, xtc)).max())
    # print(torch.abs(Y_func_V2(4, xtc, rot_configs)-Y_func_(4, xtc)).max())

    return linear_coef


def init_rot_config(seed=0, dim_list=[0,1,2,3], device=torch.device("cpu")):
    """???"""
    rng = np.random.default_rng(seed+42)
    random_generators_list = {
        2:random_generators(2, rng),
        3:random_generators(3, rng),
        # 4:random_generators(4, rng),
    }
    Y_basis_list = create_Y_basis(random_generators_list)
    rot_configs = {
        "type":'custom',
        "Js":random_generators_list,
        "Y_basis":Y_basis_list,
        "dim_list":dim_list,
    }
    Y_linear_coef = calculate_Y_func_linear_coef_np(rot_configs, rng, device)
    rotm_linear_coef = calculate_rotm_basis(rot_configs)

    for ybk in Y_basis_list:
        random_generators_list[ybk] = torch.tensor(random_generators_list[ybk], dtype=torch.float32, device=device)
        Y_basis_list[ybk] = torch.tensor(Y_basis_list[ybk], dtype=torch.float32, device=device)
        rotm_linear_coef[ybk] = torch.tensor(rotm_linear_coef[ybk], dtype=torch.float32, device=device)

    rot_configs = {
        "type":'custom',
        "Js":random_generators_list,
        "Y_basis":Y_basis_list,
        "D_basis":rotm_linear_coef,
        "dim_list":dim_list,
        "Y_linear_coef":Y_linear_coef,
    }

    print(rot_configs)

    return rot_configs


def get_feat_np(x, l, rot_configs, feature_axis=-2):
    idx = rot_configs['dim_list'].index(l)
    sidx = np.sum(np.array(rot_configs['dim_list'][:idx]) * 2 + 1).astype(int)
    eidx = sidx + 2*l+1
    return x.transpose(x, feature_axis, -1)[..., sidx:eidx].transpose(x, feature_axis, -1)

def get_feat(x, l, rot_configs, feature_axis=-2):
    idx = rot_configs['dim_list'].index(l)
    sidx = torch.sum(torch.Tensor(rot_configs['dim_list'][:idx]) * 2 + 1, dtype=int)
    eidx = sidx + 2*l+1
    return x.transpose(feature_axis, -1)[..., sidx:eidx].transpose(feature_axis, -1)


def apply_rot(x, R, rot_configs, feature_axis=-2):
    x = torch.transpose(x, feature_axis, -1)
    sidx = 0
    xr = []
    for l in rot_configs['dim_list']:
        cn = 2*l+1
        Dr = custom_rotm(l, R, rot_configs['Js'])
        xr_ = torch.einsum('...ij,...j', Dr, x[...,sidx:sidx+cn])
        xr.append(xr_)
        sidx += cn
    return torch.concat(xr, dim=-1).transpose(feature_axis, -1)


def calculate_rotm_basis(rot_configs):
    WDCOEF = [None,None,None,None]
    ns_ = 10000
    Rmin = q2R(qrand((ns_,)))
    Rmin = np.array(Rmin).astype(np.float64)
    Rmin_flat = Rmin.reshape(-1,9)

    # order 2
    y_ = np.array(custom_rotm_np(2,Rmin, rot_configs['Js']).reshape((ns_,-1))).astype(np.float64)
    Rmin_concat = (Rmin_flat[...,None]*Rmin_flat[...,None,:]).reshape((ns_,-1))
    WDCOEF[2] = np.linalg.pinv(Rmin_concat)@y_
    WDCOEF[2] = np.where(np.abs(WDCOEF[2])<1e-5, 0, WDCOEF[2])
    print(np.max(np.abs(Rmin_concat@WDCOEF[2]-y_)))

    #order 3
    Rmin_concat = (Rmin_concat[...,None]*Rmin_flat[...,None,:]).reshape((ns_,-1)).astype(np.float64)
    y_ = np.array(custom_rotm_np(3,Rmin, rot_configs['Js']).reshape((ns_,-1))).astype(np.float64)
    WDCOEF[3] = np.linalg.pinv(Rmin_concat)@y_
    WDCOEF[3] = np.where(np.abs(WDCOEF[3])<1e-5, 0, WDCOEF[3])

    print(np.max(np.abs(Rmin_concat@WDCOEF[3]-y_)))
    del Rmin, y_, Rmin_concat

    return {2:WDCOEF[2], 
            3:WDCOEF[3]}





if __name__ == '__main__':
    #%%
    # generate basis
    seed = 0
    random_generators_list = {
        0:random_generators(0, seed),
        1:random_generators(1, seed),
        2:random_generators(2, seed),
        3:random_generators(3, seed),
        4:random_generators(4, seed),
    }
    Y_basis_list = create_Y_basis(random_generators_list)


    # res = scipy.linalg.expm(np.random.uniform(-100,100, size=(100,1,1))*random_generators_list[2][2])@Y_basis_list[2][...,None]

    # print(1)


    # %%
    ## check properties
    test_n = 200
    for ii in range(1,5):
        i = ii
        print(f'test order {i}')
        x = np.random.normal(size=(test_n,3))
        rand_R1 = rand_matrix(test_n)
        rand_R2 = rand_matrix(test_n)

        # %%
        # equivariance D(R1)Y(x)=Y(R1x)
        y1 = Y_func(i,np.einsum('...ij,...j', rand_R1, x))
        y2 = np.einsum('...ij,...j', custom_rotm(i, rand_R1), Y_func(i,x))
        print(np.abs(y1-y2).max())


        # %%
        # D(I) = I
        resD = custom_rotm(i, np.eye(3))
        residual = resD - np.eye(resD.shape[-1])
        print(np.count_nonzero(np.abs(residual)>5e-4))

        # %%
        # D(R1)D(R2) = D(R1R2)
        resD1 = custom_rotm(i, rand_R1)
        resD2 = custom_rotm(i, rand_R2)
        resD3 = custom_rotm(i, rand_R1@rand_R2)
        residual = resD1@resD2 - resD3
        print(np.count_nonzero(np.any(np.abs(residual)>5e-4, (-1,-2))))

        # %%
        # det
        detres = np.linalg.det(resD1)
        residual = detres-1
        print(np.count_nonzero(np.abs(residual)>5e-4))

        # %%
        # orthogonality
        D1D1T = resD1 @ np.transpose(resD1, axes=(0,2,1))
        residual = D1D1T - np.eye(D1D1T.shape[-1])
        print(np.count_nonzero(np.abs(residual)>5e-4))
