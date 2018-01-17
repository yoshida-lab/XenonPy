from torch import save, load
from torch.nn import Module


class CheckPoint(object):
    def __init__(self, model: Module, *extra_para_list):
        self.snapshots = []
        self.check_nums = 0
        self._model = model
        self.extra = dict()
        self._extra_para_list = extra_para_list
        for key in extra_para_list:
            self.extra[key] = None

    @property
    def extra_para_list(self):
        return self._extra_para_list

    @property
    def model(self):
        return self._model

    def __call__(self, **extra_paras):
        for k in extra_paras.keys():
            if k not in self.extra:
                raise ValueError('"{}" not in the extra parameter list'.format(k))
        for k in self._extra_para_list:
            if k not in extra_paras:
                raise ValueError('"{}" must be provide'.format(k))

        extra_paras['state_dict'] = self.model.state_dict()
        self.snapshots.append(extra_paras)
        self.check_nums += 1

    def save(self, snapshots, model: str = None):
        saver = dict(extra_para_list=self._extra_para_list,
                     check_nums=self.check_nums,
                     snapshots=self.snapshots)
        save(saver, snapshots)
        if model:
            save(self._model, model)

    def read(self, file_name):
        saver = load(file_name)
        extra_para_list = saver['extra_para_list']
        self._extra_para_list = extra_para_list
        check_nums = saver['check_nums']
        self.check_nums = check_nums
        snapshots = saver['snapshots']
        self.model.load_state_dict(snapshots[-1]['state_dict'])
        return self
