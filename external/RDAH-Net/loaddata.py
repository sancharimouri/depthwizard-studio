import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from nyu_transform import *
import cv2

class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train_pred, train_depth, train_gt, transform=None, solar= False):
        # txtfile = csv_file
        # head_path = os.path.dirname(txtfile)
        gt_list = train_gt
        pred_list = train_pred
     
        self.pred_list = pred_list
        self.depth_list = train_depth
        self.gt_list = gt_list
        self.transform = transform
        self.solar = solar


    def __getitem__(self, idx):
        # print(f"Loading idx: {idx}")
        image_name = self.pred_list[idx]
        rel_depth_name = self.depth_list[idx]
        depth_name = self.gt_list[idx]
        

        
        depth = cv2.imread(depth_name,-1)
        
        # print(depth.max())
        mask = (~np.isnan(depth)) &(depth!=-9999.0) 
        depth = np.where(mask, depth, 0)  
        depth = (depth).astype(np.uint16)
        # print(depth.min())
        #depth  = cv2.cvtColor(depth  , cv2.COLOR_BGR2GRAY)
    
        depth = Image.fromarray(depth)
        
        rel_depth = cv2.imread(rel_depth_name,-1)
    
        # print("before transform"+str(rel_depth.max()))
        rel_depth = np.where(mask, rel_depth, 0)  
        
        rel_depth = Image.fromarray(rel_depth)
        if 'RGB' in image_name:
            image = Image.open(image_name)
        else:
            image = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
            if len(image.shape) == 3 and image.shape[2] == 3 and image.dtype == np.uint16:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                minv, maxv = image.min(), image.max()
                image = (image-minv)/(maxv-minv+1e-7)
                image = (image*255).astype(np.uint8)
        
                image = Image.fromarray(image, mode="RGB")
                # image.save("/root/autodl-tmp/test.png")
                
        image_np = np.array(image)
        
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            # RGB 图像：过滤 (0,0,0) 全黑像素
            mask_image = ~np.all(image_np == 0, axis=2)
        else:
            print('please check!')
        mask = mask &mask_image
        
        sample = {'name':image_name,'image': image, 'rel_depth': rel_depth, 'depth': depth, 'mask':mask,'solar':[0.0, 0.0]}

        if self.transform:
            sample = self.transform(sample)
            # print("after transform"+ str(np.array(sample['rel_depth']).max()))
        return sample

    def __len__(self):
        return len(self.pred_list)
def getTrainingData(batch_size=64,train_pred='', train_depth='', train_gt='', solar= False):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}




    # csv = csv_data
    transformed_training_trans =  depthDataset(train_pred, train_depth, train_gt,
                                        transform=transforms.Compose([
                                            ToTensor(),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]), solar= solar)



    dataloader_training = DataLoader(transformed_training_trans, batch_size, shuffle=False, num_workers=12, pin_memory=False)

   
   

    return dataloader_training


def getTestingData(batch_size=3, test_pred='', test_depth ='', test_gt='', solar= False):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
   

    transformed_testing = depthDataset(test_pred, test_depth, test_gt,
                                       transform=transforms.Compose([
                                           # CenterCrop([440, 440],[440,440]),
                                           ToTensor(),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]), solar = solar)

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=12, pin_memory=False)

    return dataloader_testing
