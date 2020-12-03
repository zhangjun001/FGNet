import torch
from byol_pytorch import BYOL
from byol_pytorch.model import FCN, Predictor
from byol_pytorch.dataset import SegDataset

encoder = FCN(input_dim=3, output_classes=24)
predictor = Predictor(dim=24)

learner = BYOL(
    encoder=encoder,
    predictor=predictor,
    image_size = 1024,
    hidden_layer = 'avgpool',
    use_momentum = False
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

imagenet = SegDataset(root_dir="/images/HER2/images")
dataloader = torch.utils.data.DataLoader(imagenet, batch_size=2, shuffle=True, num_workers=0)
device = torch.device(2)

for _ in range(100):
    print("start epoch, ", _)
    for local_batch in dataloader:
        local_batch = local_batch.to(device)
        loss = learner(local_batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # learner.update_moving_average() # update moving average of target encoder
        torch.cuda.empty_cache()
    torch.save(encoder.state_dict(), 'checkpoints/improved-net_{}.pt'.format(_))

# save your improved network
