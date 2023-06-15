import time
from concurrent.futures import ALL_COMPLETED, wait, ThreadPoolExecutor


class Test:

    def __init__(self, executor=None):
        self.l1 = []
        self.l2 = []
        self.executor = executor

    def test_case(self, l1, l2):
        if self.executor is not None:
            fs = [self.executor.submit(self.add_surr, item1, item2) for item1, item2 in zip(l1, l2)]
            wait(fs, timeout=None, return_when=ALL_COMPLETED)
        else:
            for item1, item2 in zip(l1, l2):
                self.add_surr(item1, item2)

    def add_surr(self, item1, item2):
        time.sleep(3)
        self.l1.append(item1)
        self.l2.append(item2)
        print(f'Added surr for ({item1}, {item2})\n')

    def __repr__(self):
        return f'l1: {self.l1}, l2: {self.l2}'

    def __str__(self):
        return self.__repr__()


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=4) as e:
        t = Test(executor=e)
        print(f'Parallel test before: {t}')
        t1 = time.time()
        t.test_case([1, 2, 3, 4, 5], [8, 9, 10, 11, 12])
        t2 = time.time()
        print(f'Parallel test after: {t}')

    tt = Test()
    print(f'Sequential test before: {tt}')
    t3 = time.time()
    tt.test_case([1, 2, 3, 4, 5], [8, 9, 10, 11, 12])
    t4 = time.time()
    print(f'Sequential test after: {tt}')

    print(f'Sequential time: {t4-t3} s.\nParallel time: {t2-t1} s.')
