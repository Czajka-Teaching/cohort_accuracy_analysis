import argparse
from tqdm import tqdm
from audioop import avg
import resnet as ResNet
import cv2
import os
from PIL import Image
import torch.nn as nn
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torchvision.transforms as transforms


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    parser.add_argument('--group','-g', type=str,
                        help='Ethnic Group')

    parser.add_argument('--model_path','-path', type=str,
                        help='path to where the model is saved')
    
    parser.add_argument(
        '--image_list_path','-imgpath',
        type=str,
        default=None,
        help='path to txtfile')
    
    parser.add_argument(
        '--dest','-d',
        type=str,
        help='destination to save the the final average image and gradcam images')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    

    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}


    #Declaring the ResNet model whose results are to be visualized on:
    model = ResNet.ResNet50(2)
    print("Loading the pre-trained weights......")

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(args.model_path))
    else:
        model.load_state_dict(torch.load(args.model_path))

    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5069, 0.4518, 0.4377], std=[0.2684, 0.2402, 0.2336])])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Can pick model.layer4 or model.layer4[-1] for ResNet
    target_layers = [model.layer4]#[model.layer4[-1]]

    #loading the entire directories of file/txt
    #the list_txt is in the same format as training -> image1path label /newline image2path label
    loaded_txt = np.sort(np.loadtxt(args.image_list_path,dtype=str))
    
    avg_image_correct = None
    avg_grayscale_correct = None
    avg_image_incorrect = None
    avg_grayscale_incorrect = None
    correct_count =0
    incorrect_count=0

    total_images = len(loaded_txt)
    for images,labels in tqdm(loaded_txt):
        rgb_img = cv2.imread(images, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        # If targets is None, the highest scoring category (for every member in the batch) will be used.
        # targets = [e.g ClassifierOutputTarget(281)] --> otherwise
        #targets = [ClassifierOutputTarget(1)]
        targets = None
        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.
        cam_algorithm = methods[args.method]
        with cam_algorithm(model=model,
                        target_layers=target_layers,
                        use_cuda=args.use_cuda) as cam:

        
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]
        
        img = Image.open(images).convert("RGB")
        img = transform(img)
        img = img.view(-1, 3, 224, 224)
        img = img.to(device)
        outputs = model(img)
        _,pred = torch.max(outputs, 1)

        if int(pred) == int(labels):
            correct_count+=1
            if avg_image_correct is None: 
                avg_image_correct = rgb_img
            else:
                avg_image_correct += rgb_img
        
            if avg_grayscale_correct is None: 
                avg_grayscale_correct = grayscale_cam
            else:
                avg_grayscale_correct += grayscale_cam
        
        else:
            incorrect_count+=1
            if avg_image_incorrect is None: 
                avg_image_incorrect = rgb_img
            else:
                avg_image_incorrect += rgb_img
        
            if avg_grayscale_incorrect is None: 
                avg_grayscale_incorrect = grayscale_cam
            else:
                avg_grayscale_incorrect += grayscale_cam
        
    print("The results are for {}".format(args.group))
    print("Incorrect count: ",incorrect_count)
    print("Correct count: ",correct_count)
    print("Accuracy = ",correct_count/(correct_count+incorrect_count))

    # Getting the average Image value amd also the cam value for the entire dataset
    if correct_count>0:
        avg_image_correct /= correct_count
        avg_grayscale_correct /= correct_count


        ######## FOR CORRECT CLASS ##################
        #This is the function from the code repo itself
        cam_image_correct = show_cam_on_image(avg_image_correct,avg_grayscale_correct, use_rgb=True)
        #cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False)
        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image_correct = cv2.cvtColor(cam_image_correct, cv2.COLOR_RGB2BGR)

        #Convert-average image to 0-255 scale
        avg_image_correct = np.uint8(255 * avg_image_correct)
        # average_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        avg_image_correct = cv2.cvtColor(avg_image_correct, cv2.COLOR_RGB2BGR)

        print("Saving for the correct classes!!!!")
        cv2.imwrite(os.path.join(args.dest,'{}_average_image_correct.jpg'.format(args.group)), avg_image_correct)
        cv2.imwrite(os.path.join(args.dest,'{}_{}_cam_correct.jpg'.format(args.group,args.method)), cam_image_correct)


    if incorrect_count>0:
        avg_image_incorrect /= incorrect_count
        avg_grayscale_incorrect /= incorrect_count

        ######## FOR INCORRECT CLASS ##################
        cam_image_incorrect = show_cam_on_image(avg_image_incorrect,avg_grayscale_incorrect, use_rgb=True)
        #cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False)
        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image_incorrect = cv2.cvtColor(cam_image_incorrect, cv2.COLOR_RGB2BGR)

        #Convert-average image to 0-255 scale
        avg_image_incorrect = np.uint8(255 * avg_image_incorrect)
        # average_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        avg_image_incorrect = cv2.cvtColor(avg_image_incorrect, cv2.COLOR_RGB2BGR)

        print("Saving for the incorrect classes!!!!")
        cv2.imwrite(os.path.join(args.dest,'{}_average_image_incorrect.jpg'.format(args.group)), avg_image_incorrect)
        cv2.imwrite(os.path.join(args.dest,'{}_{}_cam_incorrect.jpg'.format(args.group,args.method)), cam_image_incorrect)

    #Example terminal script
    #python3 cam_run.py --use-cuda -imgpath ../txtfiles/morph/check.txt --method gradcam -g C_M

    #If wanted to show cam on images manually:
    # #manual processing
    # heatmap1 = cv2.applyColorMap(np.uint8(255 * avg_grayscale), cv2.COLORMAP_JET)
    # heatmap1 = cv2.cvtColor(heatmap1, cv2.COLOR_BGR2RGB)
    # heatmap1 = np.float32(heatmap1) / 255

    # cam_check = heatmap1 + avg_image
    # cam_check = cam_check / np.max(cam_check)
    # cam_check =  np.uint8(255 * cam_check)
    # cam_check = cv2.cvtColor(cam_check, cv2.COLOR_RGB2BGR)
