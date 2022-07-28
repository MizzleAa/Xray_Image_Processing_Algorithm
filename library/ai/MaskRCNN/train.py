from library.ai.header import *
from library.ai.utils import *
from library.ai.MaskRCNN.dataset import *

from library.utils.io import *

def mask_rcnn(num_classes):
    # COCO 에서 미리 학습된 인스턴스 분할 모델을 읽어옵니다
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # 분류를 위한 입력 특징 차원을 얻습니다
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 미리 학습된 헤더를 새로운 것으로 바꿉니다
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 마스크 분류기를 위한 입력 특징들의 차원을 얻습니다
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # 마스크 예측기를 새로운 것으로 바꿉니다
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        # (역자주: 학습시 50% 확률로 학습 영상을 좌우 반전 변환합니다)
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def split_dataset(dataset):
    images = dataset["images"]
    categoires = dataset["categories"]
    annotations = dataset["annotations"]

    return images, categoires, annotations

def get_key_list(key, data_list):
    return [data[key] for data in data_list]

def run():
    absolute_path = "D:/workspace/Xray_Image_Processing_Algorithm/sample/ai/example_1"

    train_image_path = f"{absolute_path}/train/image"
    val_image_path = f"{absolute_path}/val/image"
    
    train_json_data = load_json(f"{absolute_path}/train/json","data.json")
    val_json_data = load_json(f"{absolute_path}/val/json", "data.json")
    
    save_model_path = f"{absolute_path}/epoch/model"
    
    num_epochs = 40
    
    optimizer_options = {
        "lr":0.005,
        "momentum":0.9
    }
    
    lr_scheduler_options = {
        "step_size":3,
        "gamma":0.1
    }
    
    train_dataloader = {
        "batch_size":3,
        "shuffle":True,
        "num_workers":0,
    }
    
    val_dataloader = {
        "batch_size":1,
        "shuffle":False,
        "num_workers":0,
    }
    
    train_image, train_categories, train_annotations = split_dataset(train_json_data)
    val_image, val_categories, val_annotations = split_dataset(val_json_data)
    
    train_image_files = get_key_list("file_name", train_image)
    val_image_files = get_key_list("file_name", val_image)
    
    num_classes = 1+len(train_categories)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = mask_rcnn(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=optimizer_options["lr"],momentum=optimizer_options["momentum"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=lr_scheduler_options["step_size"],gamma=lr_scheduler_options["gamma"])
    
    for epoch in range(num_epochs):
        #train_images
        # imgs=[] 
        train_dataset = Dataset(
            root=train_image_path,
            image_files=train_image_files,
            transforms=get_transform(train=False),
            dataset=train_json_data
        )
        
        val_dataset = Dataset(
            root=val_image_path,
            image_files=val_image_files,
            transforms=get_transform(train=False),
            dataset=val_json_data
        )
        # train
        indices = torch.randperm(len(train_dataset)).tolist()

        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
        # val
        indices = torch.randperm(len(val_dataset)).tolist()
        indices = np.linspace(0,len(val_dataset)-1,len(val_dataset))
        indices=list(indices)
        indices = [int(item) for item in indices]
        
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
        
        # train loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_dataloader["batch_size"],
            shuffle=train_dataloader["shuffle"],
            num_workers=train_dataloader["num_workers"],        
            collate_fn=utils.collate_fn    
        )
        
        # val loader
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_dataloader["batch_size"],
            shuffle=val_dataloader["shuffle"],
            num_workers=val_dataloader["num_workers"],        
            collate_fn=utils.collate_fn       
        )
        
        # train action
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        torch.save(model.state_dict(), f"{save_model_path}/epoch_{epoch}.pth" )
        lr_scheduler.step()
        
        evaluate(model, val_dataset, device=device)
        
# if __name__ == '__main__':
#     run()