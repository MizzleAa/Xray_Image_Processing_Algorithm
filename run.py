from library.coco.crop import *
from library.coco.background import *
from library.coco.rotate import *
from library.coco.synthesis import *
from library.coco.alteration import *
from library.coco.separation import *
from library.coco.align import *
from library.coco.rate import *
from library.coco.refine import *

from library.images.backgroundfix import *
from library.images.convert import *
from library.images.windowlevel import *
from library.images.blur import *
#from library.images.histogram import *

def main():
    
    # histogram 평횔비 
    # histogram_make()
    
    # window level 조절
    # windowlevel_make()
    
    # 16bit xray 이미지 컬러로 변환
    # xray_to_color_make()
    
    # 공백 이미지 생성
    # background_null_make()

    # 영상 배경 크기만 바꿈
    # background_fix_run()

    # groundtruth 검증 
    # ground_truth_view_run()
    
    # 자동 정제
    # make_auto_refine_image_annotation()
    
    # 임의 셈플본 출력(val용)
    # draw_image_annotation()
    
    # 존재하지 않은 이미지의 annotation을 제거
    # reside_image_annotation()

    # 이미지 비율 조절 데이터셋 생성(완료)
    # rate_image_annotation()

    # 잘린이미지 데이터셋 생성(완료)
    # crop_image_annotation()
    
    # 회전된 데이터셋 생성(완료)
    # rotate_image_annotation()
    
    # 배경크기 고정 이미지 데이터셋 생성(완료)
    # background_image_annotation()

    # 단품 합섬(완료)
    # single_synthesis_image_annotation()

    # 복합품 합성(완료)
    # multi_synthesis_image_annotation()

    # 카테고리 명칭 변경
    # change_category_name()
    
    # 카테고리 ID 변경
    # change_category_id()

    # image / json 합침
    # append_grouping()
    
    # image / json 분할
    # seperation_image_annotation()

    # smith xray to gray
    # image 8bit(color) 16bit로 변환
    # image_color_to_gray_make()
    
    # 8bit to 16bit
    # image_8bit_to_16bit_make()

    # 16bit to 8bit
    # image_16bit_to_8bit_make()
    
    pass

if __name__ == '__main__':
    main()
