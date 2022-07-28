from library.ai.header import *
from library.ai.utils import *

class Dataset(torch.utils.data.Dataset):
    '''
    @Params
        root : dataset root 경로
        images : image dataset file의 집합(cocodataset 의 images)
        transforms : dataset을 변형시킬 값
        dataset : cocodataset
    '''
    def __init__(self, root, image_files, transforms, dataset):
        
        self.root = root
        self.transforms = transforms
        
        #imgs
        self.image_files = image_files 
        
        # annot_image_id
        self.annotation_image_idxs = [ annotation["image_id"] for annotation in dataset["annotations"]] 
        
        #pc_train_python
        self.dataset = dataset 

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        
        # dataset 열기
        dataset = self.dataset
        
        # 해당 데이터 셋의 경로 지정
        image_path_name = os.path.join(self.root, self.image_files[idx])
        
        #현제 이미지 아이디
        image_id = -1
        
        # 이미지파일 열기
        image_data = Image.open(image_path_name)
        image_to_numpy = np.asarray(image_data)
        
        #이미지 세로 가로 크기
    
        # 이미지 정보
        images = dataset["images"]
        
        # 카테고리 정보
        categories = dataset["categories"]
        
        # 어노테이션 집합
        new_annotations = []
        
        keys = [ image_idx for image_idx, image in enumerate(dataset["images"]) if image["file_name"] == self.image_files[idx] ]
        # 이미지 데이터셋 정보 불러오기
        #for image_idx, image  in enumerate(dataset["images"]):
            #연제 불러온 파일 여부 확인
        for key in keys:
            image = copy.copy(images[key])
            if image["file_name"] == self.image_files[idx]:
                image_id = key
                
                for annotation_idx, annotation in enumerate(dataset["annotations"]):
                    #어노테이션에 포함된 image id와 이미지의 id 가 같으면
                    if image["id"] == annotation["image_id"]:
                        new_annotation = copy.copy(annotation)
                        #bbox : boxes 용
                        new_annotation["boxes"] = [ 
                            new_annotation["bbox"][0],
                            new_annotation["bbox"][1],
                            new_annotation["bbox"][0]+new_annotation["bbox"][2],
                            new_annotation["bbox"][1]+new_annotation["bbox"][3],
                        ]
                        
                        #mask
                        new_annotation["mask"] = self.make_mask(image_to_numpy, annotation)
                        
                        #crowd : 군중 상태 여부
                        new_annotation["iscrowd"] = int(new_annotation["iscrowd"])
                        
                        #append
                        new_annotations.append(new_annotation)
        
        #target data tensor화 수행
        
        target = {
            "image_id": torch.as_tensor([image_id]),
            "labels": torch.as_tensor(self.get_key_list("category_id", new_annotations), dtype=torch.int64),
            "boxes": torch.as_tensor(self.get_key_list("boxes", new_annotations), dtype=torch.float32),
            "masks": torch.as_tensor(self.get_key_list("mask",new_annotations)),
            "area": torch.as_tensor(self.get_key_list("area",new_annotations)),
            "iscrowd": torch.as_tensor(self.get_key_list("iscrowd",new_annotations), dtype=torch.int64)
        }
        
        #transform 검증 여부
        if self.transforms is not None:
            image_data, target = self.transforms(image_data, target)

        return image_data, target

    def get_key_list(self, key, data_list):
        return [data[key] for data in data_list]
        
    
    def make_mask(self, image_data, annotation):
        #이미지 세로 가로 크기
        height, width = image_data.shape[:2]
        
        #마스크 영역 초기화
        mask = np.zeros((height,width))
        
        #segmentation영역을 마스크 이미지화 수행
        for segmentation in annotation["segmentation"]:
            coodinate = self.coordinate_segm([segmentation])
            mask = cv2.fillPoly(
                mask, 
                [np.array(coodinate, np.int32)],
                (1,1)
            )
        
        result = mask.tolist()
        
        return result    
    
    def coordinate_segm(self, segmentations):
        return [((segmentations[0][i],segmentations[0][i+1])) for i in range(0,len(segmentations[0]),2)]
