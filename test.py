import sys
import os
import os.path
import glob
import cv2
import numpy as np
import torch
import architecture as arch
import time
#b.py models/1x_Fatality_DeBlur_270000_G.pth 1 1
model_path = sys.argv[1]  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

number_files = len(os.listdir('LR/'))
intRemaining = number_files
test_img_folder = 'LR/*'
results_folder = 'results/*'

model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=int(sys.argv[2]), norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

print("There are " + str(number_files) + " files.")

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0

total_time_start = time.time()
for path in glob.glob(test_img_folder):
  skip = 0
  if int(sys.argv[3]) == 1:
    for result_images in glob.glob(results_folder):
      if os.path.splitext(os.path.basename(result_images))[0] == os.path.splitext(os.path.basename(path))[0]:
        skip = 1
        idx += 1
        print(str(idx) + '/' + str(number_files) + ' skipping: ' + os.path.splitext(os.path.basename(path))[0])
  if skip == 0 and int(sys.argv[3]) == 1:
    start = time.time()
    idx += 1
    base = os.path.splitext(os.path.basename(path))[0]
    print(str(idx) + '/' + str(number_files) + ' ' + path)
    # read image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite('results/{:s}.png'.format(base), output)
    end = time.time()
    total_time_end = time.time()
    time_per_frame = end - start
    total_elapsed_time = total_time_end - total_time_start
    est_duration = number_files * time_per_frame
    print('********** Time per frame: ' + str(time_per_frame) + 's Time left: ' + str(est_duration - total_elapsed_time) + 's **********')
  if int(sys.argv[3]) == 0:
    start = time.time()
    idx += 1
    base = os.path.splitext(os.path.basename(path))[0]
    print(str(idx) + '/' + str(number_files) + ' ' + path)
    # read image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite('results/{:s}_rlt.png'.format(base), output)
    end = time.time()
    total_time_end = time.time()
    time_per_frame = end - start
    total_elapsed_time = total_time_end - total_time_start
    est_duration = number_files * time_per_frame
    print('********** Time per frame: ' + str(time_per_frame) + 's Time left: ' + str(est_duration - total_elapsed_time) + 's **********')
