import random
import numpy as np
from copy import deepcopy


class ClassIncrementalManager():
    def __init__(self, class_list: list[int], num_tasks: int, rand_seed: int = 0, shuffle=True):
        assert len(class_list) % num_tasks == 0, f"{len(class_list)}, {num_tasks}"
        self.__all_classes = deepcopy(class_list)
        self.__rng = random.Random(rand_seed)

        if shuffle:
            self.__rng.shuffle(class_list)

        self.__task_class_list = np.reshape(class_list, [num_tasks, -1]).tolist()

        self.__current_taskid = -1
        self.__num_tasks = num_tasks

        self.storage = {}

    @property
    def current_taskid(self) -> int:
        assert self.__current_taskid >= 0, "Not initialized"
        return self.__current_taskid

    @property
    def all_classes(self) -> list[int]:
        return self.__all_classes

    @property
    def num_tasks(self) -> int:
        return self.__num_tasks

    @property
    def current_task_classes(self) -> list[int]:
        return self.__task_class_list[self.current_taskid]

    @property
    def sofar_task_classes(self) -> list[list[int]]:
        extend = False
        classes = []
        for i in range(self.current_taskid + 1):
            if extend:
                classes.extend(self.__task_class_list[i])
            else:
                classes.append(self.__task_class_list[i])
        return classes

    def __iter__(self):
        return self

    def __next__(self):
        self.__current_taskid += 1
        if self.current_taskid >= len(self):
            self.__current_taskid = 0
            raise StopIteration()
        return self.current_taskid, self.current_task_classes

    def __len__(self) -> int:
        return self.num_tasks


if __name__ == "__main__":
    cl_man = ClassIncrementalManager(list(range(100)), 10, 0, shuffle=True)
    for taskid in cl_man:
        print(taskid, cl_man.current_task_classes)
