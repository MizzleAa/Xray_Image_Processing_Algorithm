from library.utils.header import *

class Progress:
    '''
        prgs = Progress(max_num=50)
        for i in range(50):
            prgs.update()
    '''
    def __init__(self,
                 max_num,
                 work_name=None,
                 bar_width=50):

        self.max_num = max_num
        self.bar_width = bar_width
        self.cnt = 0
        self.work_name = work_name
        self.file = sys.stdout
        self.init_bar()

    def init_bar(self):
        self.file.write(f'[{" "*self.bar_width}] 0/{self.max_num} work:{self.work_name}')
        self.file.flush()

    def set_work_name(self, work_name):
        self.work_name = work_name

    def update(self, cnt=1):
        self.cnt += cnt
        percentage = self.cnt/float(self.max_num)
        mark_width = int(self.bar_width*percentage)
        bar = '🥕'*mark_width + '  '*(self.bar_width-mark_width)
        # \r은 커서를 맨 앞으로 옮기는 기능
        self.file.write(f'\r[{bar}] {self.cnt}/{self.max_num} work:{self.work_name}')
        self.file.flush()


