import numpy as np


class AHP:
    # total:一级权重, subList:二级权重, param:供应商参数
    def __init__(self, total, subList, param):
        self.totalWeight = total
        self.subWeightList = subList
        self.param = param

    def compute(self):
        cnt = 0
        for w in self.subWeightList:
            cnt += w.shape[0]

        if cnt != self.param.shape[1]:
            raise("subList.shape != param.shape")

        subScoreList = []
        cnt = 0
        for weight in self.subWeightList:
            subScoreList.append(
                np.dot(weight, self.param[:, cnt:cnt+weight.shape[0]].T))
            cnt += weight.shape[0]

        subScore = np.array(subScoreList)
        scores = np.dot(self.totalWeight, subScore)

        return scores
