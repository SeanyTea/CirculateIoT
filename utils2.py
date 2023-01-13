from PIL import Image
import numpy as np

def letterbox_image(image, size):
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)

    boxes[:,0].clip(0, img_shape[1], out=np.array(boxes[:,0]))  # x1
    boxes[:,1].clip(0, img_shape[0], out=np.array(boxes[:,1]))  # y1
    boxes[:,2].clip(0, img_shape[1], out=np.array(boxes[:,2]))  # x2
    boxes[:,3].clip(0, img_shape[0], out=np.array(boxes[:,3]))  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    if len(coords == 1):
        coords = coords.reshape(1,4)
    #print(coords.shape)
    #coords[0] -= pad[0]
    #coords[2] -= pad[0]
    #coords[1] -= pad[1]
    #coords[3] -= pad[1]
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:4] /= gain
    clip_coords(np.array(coords), img0_shape)
    return coords
