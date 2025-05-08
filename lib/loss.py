import math
from turtle import forward

from numpy import argmax

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')
    
    def forward(self, output, target):
        mask = target.isnan()
        output, target = output[~mask], target[~mask]
        loss = self.criterion(output, target).mean()

        return loss

def logsumexp(input, dim=0):
    import pdb; pdb.set_trace()
    k = input.size()[dim]
    exp_value = torch.exp(input)
    sum_value = torch.sum(exp_value, dim=dim)
    return torch.log(sum_value)


class find_median(nn.Module):
    def __init__(self, timespan, k=100) -> None:
        super().__init__()
        self.timespan = timespan
        self.k = k
    
    def forward(self, input):
        '''
            input: N * k.  max approximately equal to logsumexp
            all of elements must larger than 0
            https://spaces.ac.cn/archives/6620
        '''
        
        input_list = torch.chunk(input, self.timespan)

        median_list = []
        for input_x in input_list:
            input_x = input_x * self.k
            step =  input_x.size()[0] // 2
            end = input_x.size()[0] % 2

            if end == 0:
                step = step -1
            
            for i in range(step):
                temp = F.softmax(input_x, dim=0)
                input_x = input_x - temp * input_x
            
            if end == 0:
                max_x1 = torch.logsumexp(input_x, dim=0) 
                temp = F.softmax(input_x, dim=0)
                input_x = input_x - temp * input_x
                max_x2 = torch.logsumexp(input_x, dim=0)
                median = (max_x1 + max_x2) / 2
            else:
                median = torch.logsumexp(input_x, dim=0)
            median_list.append(median / self.k)
        median = torch.stack(median_list)
        return median




if __name__ == '__main__':

    # x = torch.randn(8196, 75)
    from sklearn.metrics import mean_squared_error
    # for k in range(1, 100):
    #     x = torch.from_numpy(np.random.uniform(0, 1, (8196, 75)))
    #     value = x.max(dim=0)[0]
    #     x = x * k
    #     pred = torch.logsumexp(x, dim=0) / k

    #     # import pdb; pdb.set_trace()
    #     print(k, mean_squared_error(value, pred))
    
    
    for k in range(1, 100):

        x = torch.from_numpy(np.random.uniform(0, 1, (8196, 75)))

        value_list = []
        for input_x in torch.chunk(x, 10):
            value_list.append(input_x.median(dim=0)[0]) 

        median_func = find_median(timespan=10, k = 40)
        median = median_func(x)
        # import pdb; pdb.set_trace()
        print(k, mean_squared_error(torch.stack(value_list).view(-1), median.view(-1)))

    import pdb; pdb.set_trace()

    