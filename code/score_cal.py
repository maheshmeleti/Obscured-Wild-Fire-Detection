import os
import cv2
import numpy as np
import torch

import imgprocRoutines
from PIL import Image

from sklearn.metrics import confusion_matrix as cm
from pathlib import Path


from Model_cbam import seq_seg
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def majority_label(seq_mask):
    seq_length = seq_mask.shape[1]
    threshold = int(seq_length//2)

    summ = np.sum(seq_mask, axis=1)
    summ[summ >= threshold] = 255.0
    summ[summ < threshold] = 0

    return summ

def binarylabel(im_label,classes):
    im_dims = im_label.shape

    lab=np.zeros([im_dims[0],im_dims[1],len(classes)],dtype="uint8")
    for index, class_index in enumerate(classes):

        lab[im_label==class_index, index] = 1

    return lab

PNG_path = '../../../datasets/smoke_pattern/imp_vid_parts/Train_Val_20_neg/train/PNG_Images/'
Annot_path = '../../../datasets/smoke_pattern/imp_vid_parts/Train_Val_20_neg/train/Annotations/'
json_path = '../../../datasets/smoke_pattern/imp_vid_parts/Train_Val_20_neg/train/meta.json'

# PNG_path = '../../../datasets/smoke_pattern/imp_vid_parts/Train_Val_20_neg/val/PNG_Images/'
# Annot_path = '../../../datasets/smoke_pattern/imp_vid_parts/Train_Val_20_neg/val/Annotations/'
# json_path = '../../../datasets/smoke_pattern/imp_vid_parts/Train_Val_20_neg/val/meta.json'

checkpoint_path = 'EXP1_CBAM_CBAM_DC_SH/epoch-157 loss-0.257161.pth'
model = seq_seg(n_classes=2, seq_length=20)

pytorch_total_params = sum(p.numel() for p in model.parameters())

model.load_state_dict(torch.load(checkpoint_path))


model.to(device)
model.eval()

out_path = 'train_buffer'
# out_path = 'test_buffer'

make_dir(out_path)
pred_path = out_path + '/pred'
gt_path =  out_path + '/gt'

make_dir(pred_path)
make_dir(gt_path)

with open(os.path.join(out_path, "score.txt"), "a") as f:
    f.write('Total Parms - '+ str(pytorch_total_params//1000000)+' M\n')

seq_length = 20
n_classes = 2
for fldr in os.listdir(PNG_path):
    print(f'Processing...{fldr}')
    fldr_path = os.path.join(PNG_path, fldr)
    annt_path = os.path.join(Annot_path, fldr)
    
    rgbFrames = np.zeros((3, seq_length, 512, 512))
    maskedSegFrame = np.zeros((1, seq_length , 512, 512))
    final_label = np.zeros((n_classes, 1 , 512, 512))
    
    gt_outpath = os.path.join(gt_path, fldr + '.png')
    pred_outpath = os.path.join(pred_path, fldr + '.png')
    
    for count,i in enumerate(range(1, seq_length)):
        rgbpth = os.path.join(fldr_path, "{0:0=4d}".format(i) + '.png')
        img_rgb = cv2.imread(rgbpth)
        img_rgb = img_rgb[40:-20,340:-300,:]
        img_rgb = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_CUBIC)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        
        img_rgb = img_rgb/255.0
        img_rgb = np.rollaxis(img_rgb, -1, 0)
        rgbFrames[:, count, :, :] = (img_rgb)
        
        segpath = os.path.join(annt_path, "{0:0=4d}".format(i) + '.png')
        mask = cv2.imread(segpath)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        maskedSegFrame[:,count,:,:] = mask/255.0
        merged_label = majority_label(maskedSegFrame)
        
        #label = binarylabel(merged_label.squeeze(0), [0, 255])
        #label = np.rollaxis(label, -1, 0)
        #final_label[:,0,:,:] = label
        
    maskedSegFrame = torch.from_numpy(final_label)    
    rgbFrames = torch.from_numpy(rgbFrames)
    
    merged_label = np.squeeze(merged_label, axis=0)
    merged_label = (merged_label*255.0).astype(np.float32)
    
    rgbFrames = rgbFrames.unsqueeze(0).type(torch.FloatTensor).to(device)
    output = model(rgbFrames)
    output = output.softmax(axis=1)
    output = output.argmax(axis=1)
    output = output[0,0,:,:]
    output = output.data.cpu().numpy()
    output = (output*255.0).astype(np.float32)
    
    cv2.imwrite(gt_outpath, merged_label)
    cv2.imwrite(pred_outpath, output)

######################################## dumping images done #############################



gt_path = gt_path
op_path = pred_path

op_ext = '.png'
gt_ext = '.png'

op_premble = ''
gt_premble = ''

op_label = 255
gt_label = 255

#get all gt names
all_gt_names = [os.path.basename(f).split(gt_premble+gt_ext)[0] for f in os.listdir(gt_path) if f.endswith(gt_premble + gt_ext)]
# print(all_gt_names)

gt_blobs = 0
op_blobs = 0
tp_blobs = 0
fp_blobs = 0
fn_blobs = 0
tp_overlap_th = 0.2 #true positive overlap threshold, higher than this
fp_overlap_th = 0.2 #false positive overlap threshold, lower than this
tn_overlap_th = 0.2 # true negative overlap threshold, lowe than this


#if you want to remove small areas from output
remove_small_blobs = False
small_blob_th = 200 #pixels



for file_idx, file_name in enumerate(all_gt_names):
    
    #print(os.path.join( gt_path, file_name.split('.')[0] +  gt_premble + gt_ext))
    #print(os.path.join( op_path, file_name.split('.')[0] +  op_premble + op_ext))
    img_gt = cv2.imread( os.path.join( gt_path, file_name.split('.')[0]  + gt_premble + gt_ext), cv2.IMREAD_GRAYSCALE )
    img_op = cv2.imread( os.path.join( op_path, file_name.split('.')[0]  + op_premble +op_ext), cv2.IMREAD_GRAYSCALE )
    
    if img_gt is None:
        print( "Not Found : ", os.path.join( gt_path, file_name.split('.')[0]  + gt_premble + gt_ext) )
        continue
    if img_op is None:
        print( "Not Found : ", os.path.join( op_path, file_name.split('.')[0]  + op_premble +op_ext) )
        continue

    # img_gt = 255 * np.uint8(img_gt[:, :, 1] == gt_label)
    img_gt = 255*np.uint8(img_gt[:, :] == gt_label)
    # img_op[img_op != 0] = 255
    # img_op = 255 * np.uint8(img_op[:, :, 1] == op_label)
    img_op = 255*np.uint8(img_op[:, :] == op_label)

    
    img_gt = cv2.morphologyEx(img_gt, cv2.MORPH_CLOSE, imgprocRoutines.strel(size = 5).disk())
    #kernel = np.ones((3, 3), np.uint8)
    #img_gt = cv2.erode(img_gt, kernel, iterations=3)
    img_op = cv2.morphologyEx(img_op, cv2.MORPH_CLOSE, imgprocRoutines.strel(size = 5).disk())
    
    
    if remove_small_blobs:
        img_op = imgprocRoutines.bwareaopen(img_op, small_blob_th).remove_small()
        img_gt = imgprocRoutines.bwareaopen(img_gt, small_blob_th).remove_small()
    
    
    #get connected components
    cc_gt = cv2.connectedComponents(img_gt)
    cc_op = cv2.connectedComponents(img_op)
    
    gt_blobs = gt_blobs + cc_gt[0] - 1 #always 1 blob for bg
    op_blobs = op_blobs + cc_op[0] - 1 #always 1 blob for bg
    
    #first, check for true positives / how many gt blobs correctly detected
    for c in range(1, cc_gt[0]):
        #get gt component
        comp_gt = cc_gt[1] == c
        #overlap with op blob
        overlap = np.count_nonzero((comp_gt > 0) & (cc_op[1] > 0))/np.count_nonzero(comp_gt)
        
        if overlap >= tp_overlap_th:
            tp_blobs = tp_blobs + 1
        if overlap < tn_overlap_th:
            fn_blobs = fn_blobs + 1
#             print('fn img - ', file_name.split('.')[0] + op_premble + op_ext)
            
    #now check for false positives
    for c in range(1, cc_op[0]):
        #get op component
        comp_op = cc_op[1] == c
        #overlap with gt blob
        overlap = np.count_nonzero((comp_op > 0) & (cc_gt[1] > 0))/np.count_nonzero(comp_op)
        
        if overlap < fp_overlap_th:
            fp_blobs = fp_blobs + 1
#             print('fp img - ', file_name.split('.')[0]  + op_premble +op_ext)
    
    #print('Done : ', os.path.join( gt_path, file_name ))
    #print('gt_blobs = {0}, op_blobs = {1}, tp_blobs = {2}, fp_blobs = {3}, fn_blobs = {4}'.format(gt_blobs, op_blobs, tp_blobs, fp_blobs, fn_blobs))
    
print('gt_blobs = {0}, op_blobs = {1}, tp_blobs = {2}, fp_blobs = {3}, fn_blobs = {4}'
      .format(gt_blobs, op_blobs, tp_blobs, fp_blobs, fn_blobs))
precision = (tp_blobs / (tp_blobs + fp_blobs))
recall = (tp_blobs / (tp_blobs + fn_blobs))

print('precision - ', precision)
print('recall - ', recall)

with open(os.path.join(out_path, "score.txt"), "a") as f:
    f.write('gt_blobs = {0}, op_blobs = {1}, tp_blobs = {2}, fp_blobs = {3}, fn_blobs = {4}\n'.format(gt_blobs, op_blobs, tp_blobs, fp_blobs, fn_blobs))
    f.write(f'precision - {precision}\n')
    f.write(f'recall - {recall}\n')

############################## object detection metrics done ########################

def calculate_score(LABEL_PATH, OUTPUT_PATH, NO_OF_CLASSES = 2, LABEL_SUFFIX = '.png', OUTPUT_SUFFIX = '.png'):
    confusion_matrix = np.zeros([NO_OF_CLASSES, NO_OF_CLASSES])

    output_file_name_list = Path(OUTPUT_PATH).glob('*' + OUTPUT_SUFFIX)
    for output_file_name in output_file_name_list:
#         print(output_file_name)

        ground_truth_name = str(Path(LABEL_PATH) / (str(output_file_name.name).split(OUTPUT_SUFFIX)[0] + LABEL_SUFFIX))
        #print(ground_truth_name)
        output_image = cv2.imread(str(output_file_name), cv2.IMREAD_GRAYSCALE)
        ground_truth = cv2.imread(ground_truth_name, cv2.IMREAD_GRAYSCALE)
        output_image[output_image != 0] = 1
        ground_truth[ground_truth != 0] = 1


        confusion_matrix += cm(ground_truth.reshape(-1), output_image.reshape(-1))
#                                                      range(NO_OF_CLASSES))
        #print(confusion_matrix)

    total_predictions = np.sum(confusion_matrix)
    mean_accuracy = mean_iou = mean_dice = 0
    for class_id in range(0, NO_OF_CLASSES):
        # tn, fp, fn, tp = confusion_matrix.ravel()
        tp = confusion_matrix[class_id, class_id]
        fp = np.sum(confusion_matrix[: class_id, class_id]) + np.sum(confusion_matrix[class_id + 1:, class_id])
        fn = np.sum(confusion_matrix[class_id, : class_id]) + np.sum(confusion_matrix[class_id, class_id + 1:])
        tn = total_predictions - tp - fp - fn

        accuracy = (tp + tn) / (tn + fn + tp + fp)
        mean_accuracy += accuracy

        if ((tp + fp + fn) != 0):
            iou = (tp) / (tp + fp + fn)
            dice = (2 * tp) / (2 * tp + fp + fn)
        else:
            # When there are no positive samples and model is not having any false positive, we can not judge IOU or Dice score
            # In this senario we assume worst case IOU or Dice score. This also avoids 0/0 condition
            iou = 0.0
            dice = 0.0

        mean_iou += iou
        mean_dice += dice
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        missed_detections = fn / ( fn + tp)
        over_detections = fp / ( tp + fp)

        with open(os.path.join(out_path, "score.txt"), "a") as f:
            f.write("Sensitivity - {} specificity - {} missed_detections - {} over_detections - {}\n".format(sensitivity, specificity, missed_detections, over_detections))
            f.write("CLASS: {}: Accuracy: {}, IOU: {}, Dice: {}\n".format(class_id, accuracy, iou, dice))

        print("Sensitivity - {} specificity - {} missed_detections - {} over_detections - {}".format(sensitivity, specificity, missed_detections, over_detections))

        print("CLASS: {}: Accuracy: {}, IOU: {}, Dice: {}".format(class_id, accuracy, iou, dice))

    mean_accuracy = mean_accuracy / (NO_OF_CLASSES)
    mean_iou = mean_iou / (NO_OF_CLASSES)
    mean_dice = mean_dice / (NO_OF_CLASSES)
    with open(os.path.join(out_path, "score.txt"), "a") as f:
        f.write("Mean Accuracy: {}, Mean IOU: {}, Mean Dice: {}\n".format(mean_accuracy, mean_iou, mean_dice))
    
    print("Mean Accuracy: {}, Mean IOU: {}, Mean Dice: {}".format(mean_accuracy, mean_iou, mean_dice))

calculate_score(gt_path, op_path)