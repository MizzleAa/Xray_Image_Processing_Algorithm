from library.utils.header import *
from library.utils.decorator import *
from library.utils.log import *

@dec_func_start_end
def ground_truth_view_run():
    # path = "F://daq/_train_/origin/ct_bettery_1"
    # path = "F://custom/_train_/origin2/append"
    path = "F://test"
    
    ground_truth_options = {
        "image_path": f"{path}/image",
        "json_path": f"{path}/json/data.json",
        "gt_path": f"{path}/test"
    }
    ground_truth_view(ground_truth_options)

def ground_truth_view(options):
    try:
        ground_truth = GroundTruth_View()
        ground_truth.run(options)
    except Exception as ex:
        logger_exception(ex)
            
class CV_View:
    def __init__(self):
        super().__init__()

    def run(self, data, options={"file_name": "test"}):
        file_name = options["file_name"]
        #8 bit 아니면 imshwo가 안됨
        data = np.array(data, dtype=np.uint8)
        # bgr to rgb 번환
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        cv2.imshow(file_name, data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class PLT_View:
    def __init__(self):
        super().__init__()

    def run(self, data, options={}):
        plt.imshow(data)
        plt.show()


class GroundTruth_View:
    def __init__(self):
        super().__init__()

    def run(self, options={}):
        image_path = options["image_path"]
        json_path = options["json_path"]
        gt_path = options["gt_path"]
        
        self.visualize(image_path=image_path,
                       json_path=json_path, gt_path=gt_path)

    def catid2name(self, ann_categories):
        dict = {}
        for cate in ann_categories:
            dict[str(cate['id'])] = cate['name']
        return dict

    def make_imagelist(self, image_path):
        for path, dir, files in os.walk(image_path):
            imagefiles = [os.path.join(path, filename) for filename in files
                          if filename.endswith('.jpg') or filename.endswith('.png')]
        return imagefiles
        
    def load_json(self, json_path):
        with open(json_path, 'r', encoding="UTF-8") as f:
            jsondata = json.load(f)
        ann_images = jsondata['images']
        ann_categories = jsondata['categories']
        ann_annotations = jsondata['annotations']
        return ann_images, ann_categories, ann_annotations

    def extract_json(self, imgfile, ann_images, ann_categories, ann_annotations):
        '''
        선택한 이미지의 label 데이터 추출
        :return: 이미지의 coco형식의 정보(images,categories,annotations)
        '''
        imgfile = imgfile.replace("\\", "/")
        image_name = imgfile.split("/")[-1]
        dict = {'images': [],
                'categories': [],
                'annotations': []}
        for ann_image in ann_images:
            if image_name == ann_image['file_name']:
                dict['images'].append(ann_image)
        dict['categories'].extend(ann_categories)
        
        for ann_annotation in ann_annotations:
            try:
                if dict['images'][0]['id'] == ann_annotation['image_id']:
                    dict['annotations'].append(ann_annotation)
            except Exception as ex:
                logger_exception(ex)
            

        return dict

    def _coordinate_segm(self, segm):
        '''
        segmentation 값을 x,y 형식의 좌표로 나타내는 메서드
        Args:
            segm: annotation의 segmentation value

        Returns: list[(x,y),(x,y),...]
        '''
        list = []
        for i in range(0, len(segm), 2):
            list.append((segm[i], segm[i + 1]))
        return list

    def _make_mask(self, cordinate_segm, image):
        '''
        segmentation 좌표를 mask로 만드는 메서드
        Args:
            cordinate_segm: segmentaion의 좌표
            image: mask를 그릴 원본 이미지

        Returns: mask(True,False 형식)

        '''
        h, w, c = image.shape
        mask = np.zeros((h, w, c))
        for idx, segm in enumerate(cordinate_segm):
            arr = np.array(segm, np.int32)
            if idx == 0:
                mask = cv2.fillPoly(mask, [arr], (1, 1, 1))
            else:
                mask = cv2.fillPoly(mask, [arr], (0, 0, 0))
        mask = mask[:, :, 0]
        mask = mask >= 1
        mask.astype(bool)
        return mask

    def search_category_key(self, annotation, categories):
        ctg_key = -1
        for idx, ctg in enumerate(categories):
            if ctg["id"] == annotation['category_id']:
                ctg_key = idx
                break
        return self._hex_to_rgb(categories[ctg_key]['color'])


    def _rgb_to_hex(self, RGB_color):        
        return '#'+hex(RGB_color[0])[2:]+hex(RGB_color[1])[2:]+hex(RGB_color[2])[2:]

    def _hex_to_rgb(self, hex_code):
         return tuple(int(hex_code[i:i + 2], 16) for i in (1, 3, 5))

    def draw_annotation(self, ex_dict, image, cat2name):
        '''
        추출된 label로 ground truth box와 segmentation을 그리는 메서드
        Args:
            ex_dict: 추출된 label
            image: mask를 그릴 원본 이미지
            cat2name : {key:category_id,value:category_name}인 딕셔너리
        Returns:
            image : ground truth bbox와 segmentation이 그려진 이미지
            label_text : label된 category
        '''
        bbox_color = (72, 101, 241)  # TODO  : 색상 수정
        annotations = ex_dict['annotations']
        categories = ex_dict['categories']

        mask_colors = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(len(annotations))
        ]
        for i, ann in enumerate(annotations):
            try:
                bbox = ann['bbox']
                catid = ann['category_id']
                #bbox_color = categories[catid]['color']
                # bbox_color = self.search_category_key(ann, categories)
                bbox = list(map(int, bbox))
                cv2.rectangle(image, bbox, color=bbox_color, thickness=2)
                label_text = cat2name[str(catid)]
                text_size, _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                text_w, text_h = text_size
                #cv2.rectangle(
                #    image, (bbox[0], bbox[1] - 4, text_w, text_h + 7), (0, 0, 0), -1)
                # cv2.putText(image, label_text, (bbox[0], bbox[1] + 10), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                #             bbox_color,
                #             1, cv2.LINE_AA)
                if "segmentation" in ann.keys():
                    segm = ann['segmentation']
                    segmentation = [self._coordinate_segm(s) for s in segm]
                    mask = self._make_mask(segmentation, image)
                    image[mask] = image[mask] * 0.5 + mask_colors[i] * 0.5
            except Exception as ex:
                logger_exception(ex)
            
        return image, label_text

    def makedirs(self, gt_path, ann_categories):
        for category in ann_categories:
            name = category['name']
            path = f'{gt_path}/{name}/'
            os.makedirs(path, exist_ok=True)

    def save_image(self, gt_path, label_text, image_name, result_image):
        cv2.imwrite(f'{gt_path}/{label_text}/{image_name}', result_image)

    def visualize(self, image_path, json_path, gt_path):
        try:
            shutil.rmtree(gt_path)
        except:
            pass
        try:
            image_files = self.make_imagelist(image_path)
            ann_images, ann_categories, ann_annotations = self.load_json(
                json_path=json_path)
            cat2name = self.catid2name(ann_categories)
            self.makedirs(gt_path=gt_path, ann_categories=ann_categories)
            
            for i, image_file in enumerate(image_files):
                try:
                    image_file = image_file.replace("\\", "/")
                    image_name = image_file.split("/")[-1]
                    image = cv2.imread(image_file).astype(np.uint8)
                    ex_dict = self.extract_json(
                        image_file, ann_images, ann_categories, ann_annotations)
                    result_image, label_text = self.draw_annotation(
                        ex_dict, image, cat2name=cat2name)
                    self.save_image(gt_path=gt_path, label_text=label_text,
                                    image_name=image_name, result_image=result_image)
                except Exception as ex:
                    memo = f"\ngt_path={gt_path}\nimage_name={image_name}"
                    logger_exception(ex,memo=memo)
        except Exception as ex:
            logger_exception(ex)