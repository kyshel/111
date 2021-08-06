# snippets that assist devleop


# %%
# %%
# %%
# %% 




# %% calc dataset mean&std

transform = transforms.Compose([transforms.ToTensor(),])
dataset = Emoji(root='./data', train=True,
                    transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=1)


nimages = 0
mean = 0.0
var = 0.0
for i_batch, batch_target in enumerate(loader):
    batch = batch_target[0]
    # Rearrange batch to be the shape of [B, C, W * H]
    batch = batch.view(batch.size(0), batch.size(1), -1)
    # Update total number of images
    nimages += batch.size(0)
    # Compute mean and std here
    mean += batch.mean(2).sum(0) 
    var += batch.var(2).sum(0)

mean /= nimages
var /= nimages
std = torch.sqrt(var)

print(mean,std)
 











# %%



import numpy as np


np.random.RandomState(0)







