import cv2
import torch
import torch.nn.functional as F
import kornia
from skimage import morphology
import scipy
import scipy.cluster
from utils_app.utils import *
import scipy.ndimage as ndimage

def weighted_mean(tensor, weight, dim, keepdim=False):
    """
        weighted sum
    """
    s = (tensor * weight).sum(dim=dim, keepdim=keepdim) / torch.clamp(weight.sum(dim=dim, keepdim=keepdim), min=1e-10)
    return s

def top_half(im):
    return im[:,:,:im.shape[2]//2]

def bottom_half(im):
    return im[:,:,im.shape[2]//2:]

def add_top_half(bottom, top=None):
    if top is None:
        top = 0 * bottom
    im = torch.cat((top, bottom), 2)
    return im

def remove_small_island(img: torch.Tensor, threshold=0, min_size=500, use_multi_scale=False):
    if img.shape[1] == 1:
        mask = img
    else:
        if use_multi_scale:
            mask = (img > threshold).float()
            mask = mask.sum(1, keepdims=True) > 2.5

        else:
            mask = kornia.rgb_to_grayscale(img) > threshold

    mask = mask[0, 0].numpy()
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_size, connectivity=1).astype(float)[None, None]
    mask = torch.Tensor(mask)
    img = img * mask

    return img, mask


def binary_fill_holes(mask: torch.Tensor):
    dtype = mask.dtype
    mask = mask[0, 0].numpy()
    mask = ndimage.binary_fill_holes(mask > 0).astype(np.float)[None, None]
    mask = torch.Tensor(mask).type(dtype)

    return mask



def get_dominant_color(img, mask, NUM_CLUSTERS=5, return_img=False, scale=255):
    mask_idx = np.where(mask>0)

    ar = img[mask_idx[0], mask_idx[1]] * scale
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    print('cluster centres:\n', codes)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max] / scale

    if return_img:
        img_r = img.copy()
        img_r = img_r * (1-mask) + peak[None, None] * mask
        return peak, img_r

    return peak

def bilateralFilter(img, d=200, sigmaColor=200, sigmaSpace=75):
    max_val = img.max()

    tmp = img / max_val

    tmp = np.uint8(torch2np_im(tmp) * 255)


    tmp = cv2.bilateralFilter(tmp, d, sigmaColor, sigmaSpace) / 255
    tmp = np2torch_im(tmp)
    img = tmp * max_val


    return img


def color_matching(src_img: torch.Tensor, src_mask: torch.Tensor, ref_img: torch.Tensor, ref_mask: torch.Tensor, method='hm-mvgd-hm'):
    if src_mask.sum() < 10 or ref_mask.sum() < 10:
        return src_img
    src_img = torch2np_im(src_img)
    ref_img = torch2np_im(ref_img)
    src_mask = torch2np_im(src_mask)
    ref_mask = torch2np_im(ref_mask)

    src_mask_idx = np.where(src_mask > 0)
    ref_mask_idx = np.where(ref_mask > 0)

    src_rgb = src_img[src_mask_idx[0], src_mask_idx[1]]
    ref_rgb = ref_img[ref_mask_idx[0], ref_mask_idx[1]]

    matcher = ColorMatcher()
    img_res = matcher.transfer(src=src_rgb, ref=ref_rgb, method=method)

    img_dst = src_img.copy()
    img_dst[src_mask_idx[0], src_mask_idx[1]] = img_res
    img_dst = np2torch_im(img_dst)

    return img_dst

def pano_padding(pano, pad):
    if type(pad) is int:
        pad = (pad, pad, pad, pad)
    pano = F.pad(pano, (pad[0], pad[1], 0, 0), mode='circular')
    pano = F.pad(pano, (0, 0, pad[2], pad[3]), mode='replicate')
    return pano

def im_grad(im):
    gradx = im[:,:,:,1:] - im[:,:,:,:-1]
    grady = im[:,:,1:,:] - im[:,:,:-1,:]
    return gradx, grady

def im_grad_pano(pano):
    pano = pano_padding(pano, 1)
    gradx, grady = im_grad(pano)
    gradx = gradx[:,:,1:-1,:-1]
    grady = grady[:,:,:-1,1:-1]
    return gradx, grady

def spatial_gradient_pano(pano):
    pano = pano_padding(pano, 1)
    grad = kornia.spatial_gradient(pano)
    grad = grad[:,:,:,1:-1,1:-1]
    return grad

def total_variation_pano(pano):
    pano = pano_padding(pano, 1)
    gradx, grady = im_grad(pano)
    tv = gradx.abs().mean() + grady.abs().mean()
    return tv

def morpho_dilate_pano(pano, k=3):
    assert(k % 2 == 1)
    pano = pano_padding(pano, k // 2)
    pano = F.max_pool2d(pano, k, 1)
    return pano

def morpho_erode_pano(pano, k=3):
    assert(k % 2 == 1)
    pano = pano_padding(pano, k // 2)
    pano = -F.max_pool2d(-pano, k, 1)
    return pano

def morpho_close_pano(pano, k=3):
    pano = morpho_dilate_pano(pano, k)
    pano = morpho_erode_pano(pano, k)
    return pano

def morpho_open_pano(pano, k=3):
    pano = morpho_erode_pano(pano, k)
    pano = morpho_dilate_pano(pano, k)
    return pano

def morpho_grad_pano(pano, k=3):
    erode = morpho_erode_pano(pano, k)
    dilate = morpho_dilate_pano(pano, k)
    return dilate - erode

def color_match(input, target, mask=None, scale_only=False):
    B, C1, H, W = input.shape
    assert(B == 1)
    _, C2, _, _ = target.shape
    if mask is None:
        input_colors = input.reshape(C1,-1).transpose(0,1)
        target_colors = target.reshape(C2,-1).transpose(0,1)
    else:
        mask = (mask > 0.5).flatten()
        input_colors = input.reshape(C1,-1)[:,mask].transpose(0,1)
        target_colors = target.reshape(C2,-1)[:,mask].transpose(0,1)
    if len(input_colors) == 0:
        assert(C1 == C2)
        return input, None
    if scale_only:
        assert(C1 == C2)
        M = torch.diag((target_colors / input_colors).mean(0))
    else:
        M, _ = torch.lstsq(target_colors, input_colors)
        M = M[:C1]
    corrected = color_correct(input, M)
    return corrected, M

def color_correct(input, M):
    B, C, H, W = input.shape
    corrected = torch.matmul(input.reshape(M.shape[0],-1).transpose(0,1), M).transpose(0,1).reshape(1, M.shape[1], H, W)
    return corrected

def masked_l1(input, target, mask):
    return ((input - target).abs() * mask.detach()).mean()

def masked_l2(input, target, mask):
    return ((input - target)**2 * mask.detach()).mean()

def masked_smooth_l1(input, target, mask, beta=1.0):
    return (F.smooth_l1_loss(input, target, reduction='none', beta=beta) * mask.detach()).mean()

def masked_bce_logits(input, target, mask):
    return (F.binary_cross_entropy_with_logits(input, target, reduction='none') * mask.detach()).mean()

def masked_hinge(input, target, mask):
    return (F.relu(1 - target * input) * mask.detach()).mean()

def masked_excl(input, target, mask=None, level=1):
    assert(input.shape == target.shape)
    if mask is None:
        mask = torch.ones_like(input[:,:1])
    mask = mask.detach()
    loss = 0.0
    for l in range(level):
        maskx = torch.min(mask[:,:,:,1:], mask[:,:,:,:-1])
        masky = torch.min(mask[:,:,1:,:], mask[:,:,:-1,:])
        ix, iy = im_grad(input)
        tx, ty = im_grad(target)
        ix, iy = torch.tanh(ix) * maskx, torch.tanh(iy) * masky
        tx, ty = torch.tanh(tx) * maskx, torch.tanh(ty) * masky
        lossx = (torch.mean((ix[:,:,None] ** 2) * (tx[:,None,:] ** 2)) + 1e-8) ** 0.25
        lossy = (torch.mean((iy[:,:,None] ** 2) * (ty[:,None,:] ** 2)) + 1e-8) ** 0.25
        loss = loss + lossx + lossy
        if l < level - 1:
            input = F.avg_pool2d(input, 2)
            target = F.avg_pool2d(target, 2)
            mask = -F.max_pool2d(-mask, 2)
    loss = loss / (2 * level)
    return loss

def masked_excl_pano(input, target, mask=None, level=1):
    assert(input.shape == target.shape)
    if mask is None:
        mask = torch.ones_like(input[:,:1])
    mask = mask.detach()
    loss = 0.0
    for l in range(level):
        mask_pad = torch.cat((mask, mask[:, :, :, 0:1]), 3)
        input_pad = torch.cat((input, input[:, :, :, 0:1]), 3)
        target_pad = torch.cat((target, target[:, :, :, 0:1]), 3)
        maskx = torch.min(mask_pad[:,:,:,1:], mask_pad[:,:,:,:-1])
        masky = torch.min(mask_pad[:,:,1:,:], mask_pad[:,:,:-1,:])
        ix, iy = im_grad(input_pad)
        tx, ty = im_grad(target_pad)
        ix, iy = torch.tanh(ix) * maskx, torch.tanh(iy) * masky
        tx, ty = torch.tanh(tx) * maskx, torch.tanh(ty) * masky
        lossx = (torch.mean((ix[:,:,None] ** 2) * (tx[:,None,:] ** 2)) + 1e-8) ** 0.25
        lossy = (torch.mean((iy[:,:,None] ** 2) * (ty[:,None,:] ** 2)) + 1e-8) ** 0.25
        loss = loss + lossx + lossy
        if l < level - 1:
            input = F.avg_pool2d(input, 2)
            target = F.avg_pool2d(target, 2)
            mask = -F.max_pool2d(-mask, 2)
    loss = loss / (2 * level)
    return loss

def perceptual_loss(input_feats, target_feats):
    assert(len(input_feats) == len(target_feats))
    loss = 0.0
    for i in range(len(input_feats)):
        loss = loss + F.l1_loss(input_feats[i], target_feats[i])
    return loss

def style_loss(input_feats, target_feats):
    assert(len(input_feats) == len(target_feats))
    loss = 0.0
    for i in range(len(input_feats)):
        input_feat = input_feats[i]
        target_feat = target_feats[i]
        B, C, H, W = input_feat.size()
        input_feat = input_feat.reshape(B, C, H * W)
        target_feat = target_feat.reshape(B, C, H * W)
        A_style = torch.matmul(input_feat, input_feat.transpose(2, 1))
        B_style = torch.matmul(target_feat, target_feat.transpose(2, 1))
        loss += torch.mean(torch.abs(A_style - B_style)/(C * H * W))
    return loss

