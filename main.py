import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchok.data.transforms import (
    Compose,
    PadIfNeeded,
    Crop,
    Normalize,
    ToTensorV2,
)

import wandb

from russian_htr import HandwrittenDataset, GroupSampler
from russian_htr import StringEncoder, Generator, count_parameters, TRGAN

BATCH_SIZE = 8

# datasets params
DS = "IAM"  # "IAM" or "CVL"
HEIGHT = 32
WIDTH = 192
DS_PATH = f"/home/d.nesterov/russian-htr/data/{DS}-32.pickle"
NUM_WRITERS = 339 if DS == "IAM" else 283


transform = Compose([
	PadIfNeeded(min_height=HEIGHT, min_width=WIDTH, border_mode=0, value=0),
	Crop(y_max=HEIGHT, x_max=WIDTH),
	Normalize(mean=[0.5], std=[0.5]),
	ToTensorV2()
])

ds_train = HandwrittenDataset(DS_PATH, mode='train', transform=transform)

sampler = GroupSampler(ds_train, BATCH_SIZE, drop_last=True)
dl = DataLoader(ds_train, batch_sampler=sampler)

vocab = ' !#%()*+,-./":;?\'&abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


generator_params = {
    'vocab_size': len(vocab),
    'nhead': 1,
    'hidden_dim': 512,
    'dim_feedforward': 512,
    'num_encoder_layers': 1,
    'num_decoder_layers': 1
}
# gen = Generator(**generator_params)

lrs = {
    'g_lr': 5e-05,
    'ocr_lr': 5e-05,
    'd_lr': 5e-05,
    'w_lr': 5e-05
}

EPOCHS = 10
DEVICE = 'cuda:0'
NUM_CRITIC_GOCR_TRAIN = 2
NUM_CRITIC_DOCR_TRAIN = 1
NUM_CRITIC_GWL_TRAIN = 2
NUM_CRITIC_DWL_TRAIN = 1

model = TRGAN(generator_params, device=DEVICE, batch_size=BATCH_SIZE, output_dim=NUM_WRITERS, **lrs)
print(count_parameters(model))

wandb.init(project="hwt", name = "init")

for epoch in range(EPOCHS):
        start_time = time.time()

        for i, batch in tqdm(enumerate(dl)):
            batch['image_orig'] = ds_train.random_sample_by_wid(int(batch['writer_id'][0].numpy()))['image']
            # print(batch['image_orig'].shape)
            batch['image_orig'] = batch['image_orig'].repeat(BATCH_SIZE, 1, 1, 1)
            # print(batch['image_orig'].shape)

            if (i % NUM_CRITIC_GOCR_TRAIN) == 0:
                model._set_input(batch)
                model.optimize_G_only()
                model.optimize_G_step()

            # print("ONE")

            if (i % NUM_CRITIC_DOCR_TRAIN) == 0:
                model._set_input(batch)
                model.optimize_D_OCR()
                model.optimize_D_OCR_step()

            # print("TWO")

            if (i % NUM_CRITIC_GWL_TRAIN) == 0:
                model._set_input(batch)
                model.optimize_G_WL()
                model.optimize_G_step()

            # print("THREE")

            if (i % NUM_CRITIC_DWL_TRAIN) == 0:
                model._set_input(batch)
                model.optimize_D_WL()
                model.optimize_D_WL_step()

            # print("FOUR")

            print(model.get_current_losses())

            if i % 4 == 0:
                model.eval()
                page = model._generate_page(model.sdata, model.input['width'])
                wandb.log({ "result":[wandb.Image(page, caption="page")] })
                model.train()

        end_time = time.time()
        data_val = next(iter(ds_train))
        losses = model.get_current_losses()
        # page = model._generate_page(model.sdata, model.input['width'])
        # page_val = model._generate_page(data_val['image'].to(DEVICE), data_val['width'])


        wandb.log({'loss-G': losses['G'],
                    'loss-D': losses['D'],
                    'loss-Dfake': losses['Dfake'],
                    'loss-Dreal': losses['Dreal'],
                    'loss-OCR_fake': losses['OCR_fake'],
                    'loss-OCR_real': losses['OCR_real'],
                    'loss-w_fake': losses['w_fake'],
                    'loss-w_real': losses['w_real'],
                    'epoch' : epoch,
                    'timeperepoch': end_time-start_time,
                    })

        # wandb.log({ "result":[wandb.Image(page, caption="page"), wandb.Image(page_val, caption="page_val")], })

        print({'EPOCH':epoch, 'TIME':end_time-start_time, 'LOSSES': losses})

        # if epoch % SAVE_MODEL == 0: torch.save(model.state_dict(), MODEL_PATH+ '/model.pth')
        # if epoch % SAVE_MODEL_HISTORY == 0: torch.save(model.state_dict(), MODEL_PATH+ '/model'+str(epoch)+'.pth')
