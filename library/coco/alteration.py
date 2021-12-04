from library.utils.header import *
from library.utils.decorator import *
from library.utils.log import *
from library.utils.io import *
from library.utils.progress import *

from library.coco.groundtruth import *
from library.coco.define import *

@dec_func_start_end
def append_grouping():
    """AppendGrouping 클레스 사용 예제

    .. note::

        AppendGrouping 클레스 사용 예시 입니다. 해당 클레스를 사용하기 위한 파일 전처리가 필요 하며, 해당 파일이 올바르게 수행되었는지 검증(ground truth)이 필요합니다.
        
    """
    grouping = AppendGrouping()

    # folder = "synthesis"
    folder = "multi_synthesis_20211130_2"
    
    origin_path = f"F://custom/_seperation_/{folder}"
    save_path = f"F://custom/_seperation_/append_{folder}"

    # origin_path = f"D://daq/_train_/classifier_{folder}"
    # save_path = f"D://daq/_train_/append_{folder}"

    # origin_path = f"E://aixac/_train_/{folder}"
    # save_path = f"E://aixac/_train_/append_{folder}"

    make_dir(save_path,{"is_remove":True})

    ben_json_path = f"E://daq/_json_"
    ben_json_name = f"category_ben.json"
    ben_json = load_json(ben_json_path,ben_json_name)

    dir_list_options = {
        "dir_path": origin_path
    }

    paths, _, _ = load_dir_list(dir_list_options)
    file_path_list = []
    json_list = []

    for path in paths:
        for ben in ben_json["ben_list"]:
            if path.split("/")[-1] != ben:
                file_path_list.append(f"{path}/image")
                json_list.append(f"{path}/json/data.json")

    file_path_list = sorted(set(file_path_list))
    json_list = sorted(set(json_list))

    options = {
        "file_path_list": file_path_list,
        "json_list": json_list,
        "save_file_path": f"{save_path}/image",
        "save_json_path": f"{save_path}/json",
    }
    grouping.run(options)
    
    ground_truth_options = {
        "image_path": f"{save_path}/image",
        "json_path": f"{save_path}/json/data.json",
        "gt_path": f"{save_path}/test"
    }
    ground_truth_view(ground_truth_options)
    
class AppendGrouping(Bone):
    """다수개의 분할된 학습 데이터를 하나의 학습 데이터로 병합
    
    .. note::

        다수개의 분류된 학습 데이터가 존재할 경우 해당 데이터를 하나의 학습 데이터로 병합 시킵니다. 데이터 포맷은 coco 데이터셋으로 정의된 json으로만 동작합니다.
    
    """
    
    def run(self, options):
        """해당 함수는 다음과 같이 정의됩니다.
        병합된 image,json 데이터는 하나의 폴더안에 저장됩니다. 

        Args:
            options (dict) : 입력값을 정의합니다.
        
        Return: 
            None

        Raise:
            fileio - 파일 입출력에 관한 에러

        options의 구성은 다음과 같이 정의 됩니다.

        .. code-block:: JSON

            "options": {
                "file_path_list": "image 파일 경로 목록",
                "json_list": "json 파일 경로 목록",
                "save_file_path": "image 파일 저장 경로",
                "save_json_path": "json 파일 저장 경로"
            }

        예시

        .. code-block:: python
        
            grouping = AppendGrouping()
            options = {
                "file_path_list": file_path_list,
                "json_list": json_list,
                "save_file_path": f"{save_path}/image",
                "save_json_path": f"{save_path}/json",
            }
            grouping.run(options)

        """
        try:
            file_path_lists = options["file_path_list"]
            json_lists = options["json_list"]
            save_file_path = options["save_file_path"]
            save_json_path = options["save_json_path"]

            make_dir(save_file_path, options={"is_remove":True})
            make_dir(save_json_path, options={"is_remove":True})
            
            image_id = 0
            annotation_id = 0

            json_result = {
                "images":[],
                "categories":[],
                "annotations":[],
            }

            for file_path_list, json_list in zip(file_path_lists, json_lists):

                path_options = {
                    "image_path" : file_path_list,
                    "json_path" : json_list,
                }
                self.change_image_path(path_options)

            categories_list = []
            json_data_list = []
            for _, json_list in enumerate(json_lists):
                json_path, json_file = self.split_path_name(json_list)
                json_data_options = {
                    "load_path":json_path,
                    "file_name":json_file
                }
                json_data = self.load_json(json_data_options)
                json_data_list.append(json_data)
                categories = self.get_categories(json_data)
                category_dict_list, categories_list = self.category_overlap_check(categories_list,categories)
            # print(category_dict_list)
            for _, json_data in enumerate(json_data_list):
                
                images = self.get_images(json_data)
                annotations = self.get_annotations(json_data)
                categories = self.get_categories(json_data)
                
                progress = Progress(max_num=len(images),work_name=__name__)
            
                for _, image in enumerate(images):
                    
                
                    copy_image = copy.copy(image)

                    for _, annotation in enumerate(annotations):
                        copy_annotation = copy.copy(annotation)
                        for ctg_idx, category in enumerate(category_dict_list):
                            annotation_category_id = annotation["category_id"]
                            ac_idx = -1
                            for nctg_idx, now_category in enumerate(categories):
                                if now_category["id"] == annotation_category_id:
                                    ac_idx = nctg_idx
                                    break
                                    
                            if ac_idx != -1 and image["id"] == annotation["image_id"] and category["name"] == categories[ac_idx]["name"]:
                                copy_annotation["image_id"] = image_id
                                copy_annotation["id"] = annotation_id
                                copy_annotation["category_id"] = ctg_idx
                                annotation_id += 1
                                json_result["annotations"].append(copy_annotation)
                                # del annotations[ann_idx]
                                break
                        
                    copy_image["id"] = image_id
                    copy_image["path"] = f"{save_file_path}/{image_id}.png"
                    copy_image["file_name"] = f"{image_id}.png"
                    copy_image["dataset_id"] = 0
                    image_id += 1
                    json_result["images"].append(copy_image)
                    
                    shutil.copy(image["path"], copy_image["path"])
                
                    progress.set_work_name(f" = {...}\n")
                    progress.update()
                
            categories = []
            for ctg_idx, category in enumerate(category_dict_list):
                value = {
                    "id":ctg_idx,
                    "name":category["name"],
                    "color":self.category_random_color()
                }
                categories.append(value)
            json_result["categories"] = categories
            
            save_options = {
                "load_path":save_json_path,
                "file_name":"data.json"
            }

            self.save_json(json_result, save_options)
                    
        except Exception as ex:
            logger_exception(ex)
    
            
    def make_id_name_dict(self, names):
        result = []
        for name in names:
            value = {
                "name":name,
                "id_list": []
            }
            result.append(value)
        return result

    def category_overlap_check(self, before_ctg, after_ctg):
        now_ctg = before_ctg + after_ctg
        new_ctg = list({v["name"]:v for v in now_ctg}.values())
        result_dict = self.make_id_name_dict([ n["name"] for n in new_ctg])
        
        for now in now_ctg:
            for result in result_dict:
                if now["name"] == result["name"]:
                    result["id_list"].append(now["id"])
                    result["id_list"] = list(set(result["id_list"]))
                    
        return result_dict, now_ctg

    def category_random_color(self):
        """카테고리 특정색을 랜덤으로 반환합니다.

        Args:
            None
        
        Return: 
            hex (str) : hex code 형식의 색상값을 반환합니다.

        Raise:
            None
        
        예시

        .. code-block:: python
        
            hex_color = category_random_color()

        """
        red = random.randint(70,220)
        blue = random.randint(70,220)
        green = random.randint(70,220)

        return '#'+hex(red)[2:]+hex(blue)[2:]+hex(green)[2:]

@dec_func_start_end
def change_category_name():
    """ChangeCategoryName 클레스 사용 예제

    .. note::

        ChangeCategoryName 클레스 사용 예시 입니다. 해당 클레스를 사용하기 위한 파일 전처리가 필요 하며, 해당 파일이 올바르게 수행되었는지 검증(ground truth)이 필요합니다.
        
    """

    category_name = ChangeCategoryName()
    # folder = "synthesis"
    folder = "multi_synthesis"
    # folder = "background_fix"

    # origin_path = f"E://daq/_train_/{folder}"
    # save_path = f"E://daq/_train_/classifier_{folder}"
    origin_path = f"D://daq/_train_/{folder}"
    save_path = f"D://daq/_train_/classifier_{folder}"

    group_json_path = f"E://daq/_json_"
    group_json_name = f"category_group.json"

    # group_json_path = f"E://daq/_json_"
    # group_json_name = f"category_group.json"

    make_dir(save_path,{"is_remove":True})
    group_json = load_json(group_json_path,group_json_name)

    fulls = category_name.copy_dir_list(folder,origin_path,save_path)

    progress = Progress(max_num=len(fulls),work_name=__name__)

    for full in fulls:
        progress.set_work_name(f" = {full}\n")
        progress.update()
        
        check, change_name = category_name.is_group(full, group_json["group_list"])

        if check:
            options = {
                "json_path": f"{full}/json/data.json",
                "save_json_path": f"{full}/json/data.json",
                "category_name_list": [
                    {
                        "name": change_name,
                        "supercategory": change_name
                    }
                ]
            }
            category_name.run(options)
            ground_truth_options = {
                "image_path": f"{full}/image",
                "json_path": f"{full}/json/data.json",
                "gt_path": f"{full}/test"
            }
            ground_truth_view(ground_truth_options)


@dec_func_start_end
def change_category_id():
    """ChangeCategoryID 클레스 사용 예제

    .. note::

        ChangeCategoryID 클레스 사용 예시 입니다. 해당 클레스를 사용하기 위한 파일 전처리가 필요 하며, 해당 파일이 올바르게 수행되었는지 검증(ground truth)이 필요합니다.
        
    """
    folder = "json_test"
    
    category_id = ChangeCategoryID()
    origin_path = f"F://custom/_seperation_/{folder}"
    save_path = f"F://custom/_seperation_/change_id_{folder}"

    group_json_path = f"E://aixac/_json_"
    group_json_name = f"20211130_category_id_list.json"

    make_dir(save_path,{"is_remove":True})
    change_category_id_list = load_json(group_json_path,group_json_name)

    fulls = category_id.copy_dir_list(folder,origin_path,save_path)
    
    for full in fulls:
        options = {
            "json_path": f"{full}/json/data.json",
            "save_json_path": f"{full}/json/data.json",
            "category_id_list": change_category_id_list
        }
        category_id.run(options)
        ground_truth_options = {
            "image_path": f"{full}/image",
            "json_path": f"{full}/json/data.json",
            "gt_path": f"{full}/test"
        }
        # ground_truth_view(ground_truth_options)
        
class ChangeCategoryID(Bone):
    
    def run(self, options):
        try:
            json_path = options["json_path"]
            save_json_path = options["save_json_path"]

            category_id_list = options["category_id_list"]["group_list"]

            json_path, json_file = self.split_path_name(json_path)
            json_data_options = {
                "load_path":json_path,
                "file_name":json_file
            }
            json_data = self.load_json(json_data_options)

            images = self.get_images(json_data)
            annotations = self.get_annotations(json_data)
            categories = self.get_categories(json_data)
            
            json_result = {
                "images":images,
                "categories":[],
                "annotations":[],
            }
            
            for _, (category, change_id_list) in enumerate(zip(categories,category_id_list)):
                id = change_id_list["id"]
                copy_category = copy.copy(category)
                copy_category["id"] = id
                copy_category["name"] = change_id_list["name"]
                
                json_result["categories"].append(copy_category)

                for annotation in annotations:
                    if annotation["category_id"] == category["id"]:
                        copy_annotation = copy.copy(annotation)
                        copy_annotation["category_id"] = id
                        json_result["annotations"].append(copy_annotation)

            json_path, json_file = self.split_path_name(save_json_path)
            make_dir(json_path)

            save_options = {
                "load_path":json_path,
                "file_name":json_file
            }
            self.save_json(json_result, save_options)
            
        except Exception as ex:
            logger_exception(ex)
            
    def copy_dir_list(self, folder, origin_path, save_path):
        """directory 목록을 복사

        Args:
            folder (str) : directory 목록 명칭를 정의합니다.
            origin_path (str) : 원본 directory 경로를 정의합니다.
            save_path (str) : 복사 directory 경로를 정의합니다.
        
        Return: 
            fulls (list) : 복사된 경로를 반환합니다.

        Raise:
            fileio
        
        예시

        .. code-block:: python

            category_name = ChangeCategoryName()
            folder = "multi_synthesis"
            origin_path = f"E://daq/_train_/{folder}"
            save_path = f"E://daq/_train_/classifier_{folder}"
            fulls = category_name.copy_dir_list(folder,origin_path,save_path)

        """
        fulls = []
        try:
            make_dir(save_path,{"is_remove":True})

            dir_list_options = {
                "dir_path": origin_path
            }
            fulls, _, _ = load_dir_list(dir_list_options)

            copy_inner_list = [
                "image",
                "json"
            ]

            for full in fulls:
                for inner in copy_inner_list:
                    objects_name = full.split("/")[-1]
                    dir_copy_options = {
                        "dir_path": f"{full}/{inner}",
                        "save_path": f"{save_path}/{objects_name}/{inner}",
                    }
                    copy_dir_list(dir_copy_options)
            
            dir_list_options = {
                "dir_path": f"{save_path}"
            }
            fulls, _, _ = load_dir_list(dir_list_options)

        except Exception as ex:
            logger_exception(ex)
            
        return fulls
    
    
    
class ChangeCategoryName(Bone):
    """그룹으로 지정된 카테고리 명칭 변경
    
    .. note::

        분할된 카테고리 명칭 및 변경사항이 있는 카테고리 명칭을 변경합니다.

    """
    
    def run(self, options):
        """해당 함수는 다음과 같이 정의됩니다.
        분할된 카테고리 명칭을 지정된 json 포맷에 따라 변경 합니다.

        Args:
            options (dict) : 입력값을 정의합니다.
        
        Return: 
            None

        Raise:
            fileio - 파일 입출력에 관한 에러

        options의 구성은 다음과 같이 정의 됩니다.

        .. code-block:: JSON

            "options": {
                "file_path_list": "image 파일 경로 목록",
                "json_list": "json 파일 경로 목록",
                "category_name_list": "변경할 카테고리 목록"
            }
        
        카테고리 분류를 위한 목록은 다음과 같이 정의 됩니다.

        .. code-block:: JSON

            "group_list": [
                {
                    "class": "knife",
                    "value": [
                        "knife",
                        "chefknife",
                        "fruitknife",
                        "jackknife",
                        "officeutilityknife",
                        "artknife",
                        "steakknife",
                        "swissarmyknife",
                        "breadknife"
                    ]
                },
                {
                    "class": "bettery",
                    "value": [
                        "bettery"
                    ]
                },
            ]


        예시

        .. code-block:: python
            
            category_name = ChangeCategoryName()
            ground_truth_options = {
                "image_path": f"{full}/image",
                "json_path": f"{full}/json/data.json",
                "gt_path": f"{full}/test"
            }
            ground_truth_view(ground_truth_options)

        """
        try:
            json_path = options["json_path"]
            save_json_path = options["save_json_path"]

            category_name_list = options["category_name_list"]

            json_path, json_file = self.split_path_name(json_path)
            json_data_options = {
                "load_path":json_path,
                "file_name":json_file
            }
            json_data = self.load_json(json_data_options)

            images = self.get_images(json_data)
            annotations = self.get_annotations(json_data)
            categories = self.get_categories(json_data)

            json_result = {
                "images":images,
                "categories":[],
                "annotations":[],
            }

            for cgt_idx, (category, change_name_list) in enumerate(zip(categories,category_name_list)):
                id = cgt_idx
                copy_category = copy.copy(category)
                copy_category["id"] = id
                copy_category["origin_name"] = category["name"]
                copy_category["name"] = change_name_list["name"]
                copy_category["supercategory"] = change_name_list["supercategory"]
                
                json_result["categories"].append(copy_category)

                for annotation in annotations:
                    if annotation["category_id"] == category["id"]:
                        copy_annotation = copy.copy(annotation)
                        copy_annotation["category_id"] = id
                        json_result["annotations"].append(copy_annotation)

            json_path, json_file = self.split_path_name(save_json_path)
            make_dir(json_path)

            save_options = {
                "load_path":json_path,
                "file_name":json_file
            }
            self.save_json(json_result, save_options)
            
        except Exception as ex:
            logger_exception(ex)
            
    def copy_dir_list(self, folder, origin_path, save_path):
        """directory 목록을 복사

        Args:
            folder (str) : directory 목록 명칭를 정의합니다.
            origin_path (str) : 원본 directory 경로를 정의합니다.
            save_path (str) : 복사 directory 경로를 정의합니다.
        
        Return: 
            fulls (list) : 복사된 경로를 반환합니다.

        Raise:
            fileio
        
        예시

        .. code-block:: python

            category_name = ChangeCategoryName()
            folder = "multi_synthesis"
            origin_path = f"E://daq/_train_/{folder}"
            save_path = f"E://daq/_train_/classifier_{folder}"
            fulls = category_name.copy_dir_list(folder,origin_path,save_path)

        """
        fulls = []
        try:
            make_dir(save_path,{"is_remove":True})

            dir_list_options = {
                "dir_path": origin_path
            }
            fulls, _, _ = load_dir_list(dir_list_options)

            copy_inner_list = [
                "image",
                "json"
            ]

            for full in fulls:
                for inner in copy_inner_list:
                    objects_name = full.split("/")[-1]
                    dir_copy_options = {
                        "dir_path": f"{full}/{inner}",
                        "save_path": f"{save_path}/{objects_name}/{inner}",
                    }
                    copy_dir_list(dir_copy_options)
            
            #TODO 경로 바꿔야됨 자동으로되게
            dir_list_options = {
                "dir_path": f"D://daq/_train_/classifier_{folder}"
                # "dir_path": f"E://ct/_train_/classifier_{folder}"
            }
            fulls, _, _ = load_dir_list(dir_list_options)

        except Exception as ex:
            logger_exception(ex)
            
        return fulls

    def is_group(self,full,group_list):
        """directory 목록을 복사

        Args:
            full (str) : 전체 파일 경로를 정의합니다.
            group_list (str) : 변경할 카테고리 목록으로 정의합니다.
        
        Return: 
            None

        Raise:
            None
        
        예시

        .. code-block:: python
            
            category_name = ChangeCategoryName()
            full = "E://test/xray_knife_a_1"
            check, change_name = category_name.is_group(full, group_json["group_list"])

        """

        name = full.split("/")[-1].split("_")[1]

        check = False
        change_name = ""
        for group_name in group_list:
            for value in group_name["value"]:
                if value == name:
                    change_name = group_name["class"]
                    check = True
                    break

        return check, change_name