import torch
import fairseq

N = 1
m = torch.load('bart.large/model.pt', map_location=torch.device('cpu'))
model = m['model']

for k in ['decoder.embed_tokens.weight', 'encoder.embed_tokens.weight']:
  par = model[k]
  x = par[-1]
  xx = x.unsqueeze(0).repeat(N, 1)
  par_aug = torch.cat((par, xx), dim=0)
  model[k] = par_aug

m['model'] = model
torch.save(m, 'bart.large/new_model.pt')
