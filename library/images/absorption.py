from library.utils.header import *
from library.utils.decorator import *

def absorption(first, second, options={}):
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
    axis = options["axis"]
    result = np.concatenate((first, second), axis)
    return result

