import os
import numpy as np
import cv2
from torch.cuda.amp import autocast, GradScaler
import torch
from torch.autograd import Variable
import torch.nn.functional as f
from imgproc_funcs import fill, bwareaopen

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def predict_output(model, window_frames):
    seq_length = len(window_frames)
    rgbFrames = np.zeros((3, seq_length, 512, 512))

    for count, img_rgb in enumerate(window_frames):
        
        
        img_rgb = img_rgb/255.0
        img_rgb = np.rollaxis(img_rgb, -1, 0)
        rgbFrames[:, count, :, :] = (img_rgb)

    rgbFrames = np.expand_dims(rgbFrames, 0)
    rgbFrames = torch.from_numpy(rgbFrames)

    rgb = Variable(rgbFrames).type(torch.FloatTensor).to(device)

    with autocast():
        out = model(rgb)

    return out

# def overlap_prediction(final_output, video_frame):

def draw_boundaries(rgb_image, thresh_image, color=(0, 0, 255)):
    ret, thresh_image = cv2.threshold(thresh_image,127,255,cv2.THRESH_BINARY)
    # thresh_image = cv2.cvtColor(thresh_image, cv2.COLOR_BGR2GRAY)
    # print(thresh_image.shape, np.unique(thresh_image))
    thresh_image = np.uint8(thresh_image)
    rgb_image = rgb_image.copy()
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(rgb_image, [contour], 0, color, 2)
    return rgb_image

def majority_label(seq_mask):
        seq_length = seq_mask.shape[1]
        threshold = int(seq_length//2)
        
        summ = np.sum(seq_mask, axis=1)
        summ[summ >= threshold] = 255.0
        summ[summ < threshold] = 0
        
        return summ

def IR2Label(ir_image, thresh=180):
    ir_image = ir_image.copy()
    frame_gray = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)
    ret, frame_thresholded = cv2.threshold(frame_gray, thresh, 255, cv2.THRESH_BINARY)

    # dilate
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(frame_thresholded, kernel, iterations=2)
    # fill
    image_filled = fill(img_dilation)
    # erode
    img_erosion = cv2.erode(image_filled, kernel, iterations=1)
    return img_erosion

def IR2LabelSmooth(ir_image):
    kernel = np.ones((5,5),np.float32)/25
    frame_ir_smooth = cv2.filter2D(ir_image,-1,kernel)
    IR_label = IR2Label(frame_ir_smooth, thresh = 100)
    IR_label_small_removed = bwareaopen(IR_label, 200).remove_small()
    return IR_label_small_removed

def draw_gt(org_image, window_frames_ir):
    seq_length = len(window_frames_ir)
    maskedSegFrame = np.zeros((1, seq_length , 512, 512))
    for count, ir_frame in enumerate(window_frames_ir):
        # print('IR Frames shape - ',ir_frame.shape)
        binary_label = IR2LabelSmooth(ir_frame)
        mask = cv2.resize(binary_label, (512, 512), interpolation=cv2.INTER_NEAREST)
        maskedSegFrame[:,count,:,:] = mask/255.0
    
    merged_label = majority_label(maskedSegFrame)
    merged_label = np.squeeze(merged_label, axis=0)

    # print(np.unique(merged_label))
    gt_overlapped = draw_boundaries(org_image, merged_label, color=(0, 255, 0))
    return gt_overlapped

def video_predict(seq_length, model, video_rgb_path, video_ir_path, out_path, out_fps=10):

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('M','J','P','G'), out_fps, (512,512))
    overlap_length = seq_length - 1
    cap = cv2.VideoCapture(video_rgb_path)
    # print('path existance - ', os.path.exists(video_ir_path))
    cap_ir = cv2.VideoCapture(video_ir_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = 0
    end_frame = seq_length
    # j = 1
    while start_frame < total_frames:
        window_frames = []
        window_frames_ir = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        cap_ir.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for i in range(seq_length):
            ret, frame = cap.read()
            ret_ir, frame_ir = cap_ir.read()
            if not ret: 
                break

            frame = frame[40:-20,340:-300,:]   #[20:,340:-280,:]
            img_rgb = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_CUBIC)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

            
            window_frames.append(img_rgb)
            window_frames_ir.append(frame_ir)
        # print(f"Window {start_frame}-{end_frame}: {len(window_frames)} frames")
        start_frame = start_frame + seq_length - overlap_length
        end_frame = start_frame + seq_length

        if len(window_frames) != seq_length:
            break

        window_output = predict_output(model, window_frames)
        window_output = window_output
        window_output = window_output.softmax(axis=1)
        window_output = window_output.argmax(axis=1)
        window_output = window_output[0,0,:,:]
        window_output = window_output.data.cpu().numpy()
        window_output = (window_output*255.0).astype(np.float32)
        # print('window_output - ', window_output.shape)
        overlapped_pred = draw_boundaries(window_frames[-1][:,:,::-1], window_output)
        # print('overlapped_pred - ', overlapped_pred.shape)
        # overlapped_gt = draw_gt(overlapped_pred, window_frames_ir)
        # cv2.imwrite('output/out.png', overlapped_gt)
        # print(overlapped_gt.shape)
        out.write(overlapped_pred)

    out.release()
    cv2.destroyAllWindows()

    print('Done!!!')


# checkpoint_path = 'exp2/epoch-104 loss-0.066056.pth'
# checkpoint_path = 'exp3/epoch-56 loss-0.070961.pth'
# checkpoint_path = 'exp5_seq20/epoch-44 loss-0.136743.pth'


root_rgb_path = '../../../datasets/smoke_pattern/imp_vid_parts/RGB/'
root_ir_path = '../../../datasets/smoke_pattern/imp_vid_parts/IR/'
root_out_path = 'for_paper'

make_dir(root_out_path)

seq_length=20
n_classes = 2
# from Model_cbam import seq_seg
from Model_cbam import seq_seg
model = seq_seg(n_classes, seq_length)
checkpoint_path = 'EXP1_CBAM_CBAM_DC_SH/epoch-157 loss-0.257161.pth'
model.load_state_dict(torch.load(checkpoint_path))
model.to(device)
model.eval()



# vid_names = ['3_1', '3_2', '3_3', '3_4', '3_5', '3_6', '3_7', 
#                   '3_8', '3_9', '4_1', '4_2', '5_1', '5_2', '6_1', 
#                   '6_2', '7_1', '7_2', '7_3', '7_4', '7_5', '7_6', '7_7']

vid_names = ['7_1']


for root_name in vid_names:
    print('Processing... {}'.format(root_name))
    rgb_name = root_name.split('_')[0] + '_RGB_' + root_name.split('_')[1] + '.mp4'
    ir_name = root_name.split('_')[0] + '_IR_' + root_name.split('_')[1] + '.mp4'
    out_name = root_name.split('_')[0] + '_OUT_' + root_name.split('_')[1] + '.avi'

    video_rgb_path = os.path.join(root_rgb_path, rgb_name)
    video_ir_path = os.path.join(root_ir_path, ir_name)
    out_path = os.path.join(root_out_path, out_name)

    video_predict(seq_length, model, video_rgb_path, video_ir_path, out_path, out_fps=10)
