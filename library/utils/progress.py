from library.utils.header import *

class Progress:
    def __init__(self,
                 max_num,
                 work_name=None,
                 bar_width=50):
        self.max_num = max_num
        self.bar_width = bar_width
        self.cnt = 0
        self.work_name = work_name
        self.start_time = time.time()
        self.file = sys.stdout
        self.init_bar()

    def init_bar(self):
        self.file.write(f'\n[{" " * self.bar_width}] 0/{self.max_num} work:{self.work_name} elapsed:0s ETA:0s')
        self.file.flush()

    @property
    def terminal_width(self):
        width, _ = get_terminal_size()
        return width

    def update(self, cnt=1):
        self.cnt += cnt
        percentage = self.cnt / float(self.max_num)
        elapsed = int(time.time() - self.start_time)
        # 비례식 [걸린시간 : x(남은시간) = 진행된 pecentage : 남은 percentage] 0.5는 왜 더하는지 모름
        eta = int(elapsed * (1 - percentage) / percentage + 0.5)
        # \r은 커서를 맨 앞으로 옮기는 기능
        msg = f'\r[{{}}] {self.cnt}/{self.max_num} work:{self.work_name} elapsed:{elapsed}s ETA:{eta}s'
        mark_width = int(self.bar_width * percentage)
        bar_width = min(self.bar_width,
                        int(self.terminal_width - len(msg)) + 2,
                        int(self.terminal_width * 0.6))
        bar = '>' * mark_width + ' ' * (bar_width - mark_width)
        self.file.write(msg.format(bar))
        self.file.flush()