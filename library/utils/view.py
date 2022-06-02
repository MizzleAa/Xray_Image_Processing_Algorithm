from library.utils.header import *
from library.utils.decorator import *

def cv_view(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plt_view(image):
    plt.imshow(image)
    plt.show()