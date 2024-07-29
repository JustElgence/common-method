import torch


# test torch.gather
x = torch.arange(12).reshape(2,3,2)
print(x,x.shape)
dd = x.sum(2)
print(dd)
x = x[dd != 1]
print(x)
quit()
gather_index = torch.LongTensor([[0, 1,2],[0, 1,2]])
gather_index = gather_index.unsqueeze(-1).expand(x.shape[0],3,x.shape[-1])
print(gather_index,gather_index.shape)

y = torch.gather(x, dim=1, index=gather_index)
print(y,y.shape)


# test torch.index_select
print('***************************分隔符**************************')
print('test torch.index_select ')
print('***************************分隔符**************************')
a = torch.linspace(1, 12, steps=12).view(3, 2, 2)
print(a,a.shape)
b = torch.index_select(a, 0, torch.tensor([0]))
print(b,b.shape)
quit()
print(a.index_select(0, torch.tensor([0, 2])))
c = torch.index_select(a, 1, torch.tensor([1, 3]))
print(c)


