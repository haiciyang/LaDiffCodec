import torch
import numpy as np
from matplotlib import pyplot as plt

# file_name = 'entropy_results/0925_offenc_vae_mean.pt'
# file_name = 'entropy_results/1104_uncond_entropy.pt'
file_name = 'entropy_results/0925_offenc_vae_entropy.pt'

entropy_step = torch.load(file_name)

entropy_step = np.array(entropy_step)

cum_ent = np.cumsum(entropy_step)

print(entropy_step.shape)

gap_step = 30
entr_temp = np.array(entropy_step)[gap_step:]
cum_ent_temp = np.cumsum(entr_temp)

for i in range(len(entr_temp)):
    print(999-gap_step-i, entr_temp[i]/64/160, cum_ent_temp[i]/64/160, cum_ent_temp[i]/160*50)


draw_ent = cum_ent_temp[:-50]
plt.scatter(range(len(draw_ent)), draw_ent/160*50, s=1)
plt.savefig('entropy_results/bps-30-950.pdf')
plt.clf()

# plt.plot(range(1, 1000), cum_ent.squeeze()/(64*160))
# plt.savefig('dum_entropy.pdf')
# plt.clf()