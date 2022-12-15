import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from torchok.losses import CTCLoss

from .generator import Generator
from .gans import WriterDiscriminator, Discriminator
from .string_encoder import StringEncoder
from .crnn import CRNN
from .vocab import VOCABULARY


IMG_HEIGHT = 32
NUM_WORDS = 3

# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real, len_text_fake, len_text, mask_loss):
    mask_real = torch.ones(dis_real.shape).to(dis_real.device)
    mask_fake = torch.ones(dis_fake.shape).to(dis_fake.device)
    if mask_loss and len(dis_fake.shape)>2:
        for i in range(len(len_text)):
            mask_real[i, :, :, len_text[i]:] = 0
            mask_fake[i, :, :, len_text_fake[i]:] = 0
    loss_real = torch.sum(F.relu(1. - dis_real * mask_real))/torch.sum(mask_real)
    loss_fake = torch.sum(F.relu(1. + dis_fake * mask_fake))/torch.sum(mask_fake)
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake, len_text_fake, mask_loss):
    mask_fake = torch.ones(dis_fake.shape).to(dis_fake.device)
    if mask_loss and len(dis_fake.shape)>2:
        for i in range(len(len_text_fake)):
            mask_fake[i, :, :, len_text_fake[i]:] = 0
    loss = -torch.sum(dis_fake * mask_fake) / torch.sum(mask_fake)
    return loss


class TRGAN(nn.Module):
    def __init__(self, gen_params, device, g_lr, ocr_lr, d_lr, w_lr, batch_size, output_dim, resolution: int = 16):
        super().__init__()

        self.epsilon = 1e-7
        self.resolution = resolution
        self.batch_size = batch_size
        self.device = device
        self.netG = Generator(**gen_params).to(device)

        # removed nn.DataParallel from here
        self.netD = Discriminator().to(device)
        self.netW = WriterDiscriminator(output_dim).to(device)

        self.netconverter = StringEncoder(VOCABULARY)
        self.netOCR = CRNN(len(VOCABULARY), gen_params['hidden_dim']).to(device)
        self.OCR_criterion = CTCLoss(zero_infinity=True, reduction='none')

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=g_lr, betas=(0.0, 0.999))
        self.optimizer_OCR = torch.optim.Adam(self.netOCR.parameters(), lr=ocr_lr, betas=(0.0, 0.999))

        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=d_lr, betas=(0.0, 0.999))

        self.optimizer_wl = torch.optim.Adam(self.netW.parameters(), lr=w_lr, betas=(0.0, 0.999))


        self.optimizers = [self.optimizer_G, self.optimizer_OCR, self.optimizer_D, self.optimizer_wl]

        self.optimizer_G.zero_grad()
        self.optimizer_OCR.zero_grad()
        self.optimizer_D.zero_grad()
        self.optimizer_wl.zero_grad()

        self.loss_G = 0
        self.loss_D = 0
        self.loss_Dfake = 0
        self.loss_Dreal = 0
        self.loss_OCR_fake = 0
        self.loss_OCR_real = 0
        self.loss_w_fake = 0
        self.loss_w_real = 0
        self.Lcycle1 = 0
        self.Lcycle2 = 0
        self.lda1 = 0
        self.lda2 = 0
        self.KLD = 0


        with open("/home/d.nesterov/russian-htr/data/query_words_eng.txt") as f:
            words = f.read().splitlines()
            self.lex = list(filter(lambda x: len(x) < 20, words))
        # lex = []
        # for word in self.lex:
        #     try:
        #         word = word.decode("utf-8")
        #     except:
        #         continue
        #     if len(word) < 20:
        #         lex.append(word)
        # self.lex = lex

        sample = 'Hello, this is sample text to reproduce'

        self.text = [j for j in sample.split()]
        self.eval_text_encode, self.eval_len_text = self.netconverter.encode(self.text)
        self.eval_text_encode = self.eval_text_encode.to(device).repeat(batch_size, 1, 1).flatten(1, 2)

    @torch.no_grad()
    def _generate_page(self, ST, SLEN, eval_text_encode = None, eval_len_text = None):
        if eval_text_encode == None:
            eval_text_encode = self.eval_text_encode
        if eval_len_text == None:
            eval_len_text = self.eval_len_text

        words = [word for word in np.random.choice(self.lex, self.batch_size, replace=False)]
        text_encode_fake, len_text_fake = self.netconverter.encode(words)
        text_encode_fake = text_encode_fake.to(self.device).repeat(self.batch_size, 1, 1).flatten(1, 2)

        # print(ST.shape, text_encode_fake.shape)
        # self.fakes = self.netG.forward_queries(ST, eval_text_encode)
        self.fakes = self.netG(ST, text_encode_fake).detach()
        # print(self.fakes.shape)

        page1s = []
        page2s = []
        for batch_idx in range(self.batch_size):
            word_t = []
            word_l = []

            gap = np.ones([IMG_HEIGHT, 16])  # IMG_HEIGHT = 32

            line_wids = []

            for idx, fake_ in enumerate(self.fakes):

                word_t.append((fake_[0,:len_text_fake[idx] * self.resolution].cpu().numpy() + 1)/2)
                word_t.append(gap.copy())

                if len(word_t) == 16 or idx == len(self.fakes) - 1:
                    # shapes = [a.shape for a in word_t]
                    # print(f"SHAPES ONE: {shapes}")
                    line_ = np.concatenate(word_t, -1)

                    word_l.append(line_)
                    line_wids.append(line_.shape[1])

                    word_t = []

            gap_h = np.ones([16, max(line_wids)])

            page_= []

            for l in word_l:

                pad_ = np.ones([IMG_HEIGHT,max(line_wids) - l.shape[1]])

                page_.append(np.concatenate([l, pad_], 1))
                page_.append(gap_h)

            # for a in page_:
            #     print(a.shape)
            page1 = np.concatenate(page_, 0)


            word_t = []
            word_l = []

            gap = np.ones([IMG_HEIGHT, 16])

            line_wids = []

            sdata_ = [i.unsqueeze(1) for i in torch.unbind(ST, 1)]

            for idx, st in enumerate((sdata_)):

                # print(SLEN.shape, SLEN)
                word_t.append((st[idx, 0,:,:int(SLEN.cpu().numpy()[batch_idx])].squeeze().cpu().numpy()+1)/2)
                if word_t[-1].shape[0] == 16:
                    print(word_t[-1].shape, st.shape)
                word_t.append(gap.copy())
                # print(word_t[-1].shape)

                if len(word_t) == 16 or idx == len(sdata_) - 1:
                    # shapes = [a.shape for a in word_t]
                    # print(f"SHAPES TWO: {shapes}")
                    line_ = np.concatenate(word_t, -1)

                    word_l.append(line_)
                    line_wids.append(line_.shape[1])

                    word_t = []


            gap_h = np.ones([16, max(line_wids)])

            page_= []

            for l in word_l:

                pad_ = np.ones([IMG_HEIGHT,max(line_wids) - l.shape[1]])

                page_.append(np.concatenate([l, pad_], 1))
                page_.append(gap_h)

            page2 = np.concatenate(page_, 0)

            merge_w_size =  max(page1.shape[0], page2.shape[0])

            if page1.shape[0] != merge_w_size:
                page1 = np.concatenate([page1, np.ones([merge_w_size-page1.shape[0], page1.shape[1]])], 0)

            if page2.shape[0] != merge_w_size:
                page2 = np.concatenate([page2, np.ones([merge_w_size-page2.shape[0], page2.shape[1]])], 0)

            page1s.append(page1)
            page2s.append(page2)

            #page = np.concatenate([page2, page1], 1)

        page1s_ = np.concatenate(page1s,0)
        max_wid = max([i.shape[1] for i in page2s])
        padded_page2s = []

        for pair in page2s:
            padded_page2s.append(np.concatenate([pair, np.ones([ pair.shape[0], max_wid-pair.shape[1]])], 1))

        padded_page2s_ =  np.concatenate(padded_page2s,0)

        return np.concatenate([padded_page2s_, page1s_], 1)

    def get_current_losses(self):

        losses = {}

        losses['G'] = self.loss_G
        losses['D'] = self.loss_D
        losses['Dfake'] = self.loss_Dfake
        losses['Dreal'] = self.loss_Dreal
        losses['OCR_fake'] = self.loss_OCR_fake
        losses['OCR_real'] = self.loss_OCR_real
        losses['w_fake'] = self.loss_w_fake
        losses['w_real'] = self.loss_w_real
        losses['cycle1'] = self.Lcycle1
        losses['cycle2'] = self.Lcycle2
        losses['lda1'] = self.lda1
        losses['lda2'] = self.lda2
        losses['KLD'] = self.KLD

        return losses

    def _set_input(self, input):
        self.input = input

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self):
        self.real = self.input['image_orig'].to(self.device)
        self.label = self.input['target']
        self.sdata = self.input['image'].to(self.device)
        self.ST_LEN = self.input['width']
        self.wcl = self.input['writer_id'].squeeze().to(self.device)

        self.text_encode, self.len_text = self.netconverter.encode(self.label)
        self.text_encode = self.text_encode.to(self.device).detach()
        self.len_text = self.len_text.detach()

        self.words = [word for word in np.random.choice(self.lex, self.batch_size, replace=False)]
        self.text_encode_fake, self.len_text_fake = self.netconverter.encode(self.words)
        self.text_encode_fake = self.text_encode_fake.to(self.device).repeat(self.batch_size, 1, 1).flatten(1, 2)

        # self.text_encode_fake_js = []

        # for _ in range(NUM_WORDS - 1):
        #     self.words_j = [word for word in np.random.choice(self.lex, self.batch_size, replace=False)]
        #     self.text_encode_fake_j, self.len_text_fake_j = self.netconverter.encode(self.words_j)
        #     self.text_encode_fake_j = self.text_encode_fake_j.to(self.device)
        #     self.text_encode_fake_js.append(self.text_encode_fake_j)

        # self.fake = self.netG(self.sdata, self.text_encode_fake, self.text_encode_fake_js)
        print(self.sdata.shape, self.text_encode_fake.shape)
        self.fake = self.netG(self.sdata, self.text_encode_fake)

    def backward_D_OCR(self):
        # print(f"REAL IN backward_D_OC: {self.real.shape}, {self.fake.shape}")
        pred_real = self.netD(self.real.detach())

        # print("FAKE IN backward_D_OCR")
        pred_fake = self.netD(**{'x': self.fake.detach()})

        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(), self.len_text.detach(), True)

        self.loss_D = self.loss_Dreal + self.loss_Dfake

        self.pred_real_OCR = self.netOCR(self.real.detach())
        preds_size = torch.IntTensor([self.pred_real_OCR.size(0)] * self.batch_size).detach()

        # print(self.pred_real_OCR.shape, self.text_encode.detach().shape, preds_size)

        loss_OCR_real = self.OCR_criterion(self.pred_real_OCR, self.text_encode.detach(), preds_size, self.len_text.detach())
        self.loss_OCR_real = torch.mean(loss_OCR_real[~torch.isnan(loss_OCR_real)])

        loss_total = self.loss_D + self.loss_OCR_real
        # backward
        loss_total.backward()
        for param in self.netOCR.parameters():
            param.grad[param.grad!=param.grad]=0
            param.grad[torch.isnan(param.grad)]=0
            param.grad[torch.isinf(param.grad)]=0

        return loss_total


    def backward_D_WL(self):
        # Real
        pred_real = self.netD(self.real.detach())

        pred_fake = self.netD(**{'x': self.fake.detach()})


        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(), self.len_text.detach(), True)

        self.loss_D = self.loss_Dreal + self.loss_Dfake


        self.loss_w_real = self.netW(self.real.detach(), self.wcl).mean()
        # total loss
        loss_total = self.loss_D + self.loss_w_real

        # backward
        loss_total.backward()


        return loss_total

    def optimize_D_WL(self):
        self.forward()
        self.set_requires_grad([self.netD], True)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], True)

        self.optimizer_D.zero_grad()
        self.optimizer_wl.zero_grad()

        self.backward_D_WL()


    def backward_D_OCR_WL(self):
        # Real
        # print(f"REAL in backward_D_OCR_WL")
        pred_real = self.netD(**{'x': self.real.detach(), 'z': self.real_z_mean.detach()})

        # Fake
        # try:
        # print(f"FAKE in backward_D_OCR_WL")
        pred_fake = self.netD(**{'x': self.fake.detach(), 'z': self.z.detach()})
        # except:
        #     print('a')

        # Combined loss
        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(), self.len_text.detach(), mask_loss=True)

        self.loss_D = self.loss_Dreal + self.loss_Dfake
        # OCR loss on real data
        self.pred_real_OCR = self.netOCR(self.real.detach())
        preds_size = torch.IntTensor([self.pred_real_OCR.size(0)] * self.batch_size).detach()
        loss_OCR_real = self.OCR_criterion(self.pred_real_OCR, self.text_encode.detach(), preds_size, self.len_text.detach())
        self.loss_OCR_real = torch.mean(loss_OCR_real[~torch.isnan(loss_OCR_real)])
        # total loss
        self.loss_w_real = self.netW(self.real.detach(), self.wcl)
        loss_total = self.loss_D + self.loss_OCR_real + self.loss_w_real

        # backward
        loss_total.backward()
        for param in self.netOCR.parameters():
            param.grad[param.grad!=param.grad]=0
            param.grad[torch.isnan(param.grad)]=0
            param.grad[torch.isinf(param.grad)]=0

        return loss_total

    def optimize_D_WL_step(self):
        self.optimizer_D.step()
        self.optimizer_wl.step()
        self.optimizer_D.zero_grad()
        self.optimizer_wl.zero_grad()

    def backward_OCR(self):
        # OCR loss on real data
        self.pred_real_OCR = self.netOCR(self.real.detach())
        preds_size = torch.IntTensor([self.pred_real_OCR.size(0)] * self.batch_size).detach()
        loss_OCR_real = self.OCR_criterion(self.pred_real_OCR, self.text_encode.detach(), preds_size, self.len_text.detach())
        self.loss_OCR_real = torch.mean(loss_OCR_real[~torch.isnan(loss_OCR_real)])

        # backward
        self.loss_OCR_real.backward()
        for param in self.netOCR.parameters():
            param.grad[param.grad!=param.grad]=0
            param.grad[torch.isnan(param.grad)]=0
            param.grad[torch.isinf(param.grad)]=0

        return self.loss_OCR_real

    def backward_D(self):
        # Real
        # print(f"REAL in backward_D")
        pred_real = self.netD(**{'x': self.real.detach(), 'z': self.real_z_mean.detach()})

        # print(f"FAKE in backward_D")
        pred_fake = self.netD(**{'x': self.fake.detach(), 'z': self.z.detach()})

        # Combined loss
        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(), self.len_text.detach(), mask_loss=True)
        self.loss_D = self.loss_Dreal + self.loss_Dfake

        # backward
        self.loss_D.backward()

        return self.loss_D

    def backward_G_only(self):

        self.gb_alpha = 0.7
        #self.Lcycle1 = self.Lcycle1.mean()
        #self.Lcycle2 = self.Lcycle2.mean()
        # print(f"FAKE in backward_G_only")
        self.loss_G = loss_hinge_gen(self.netD(**{'x': self.fake}), self.len_text_fake.detach(), True).mean()
        # print(f"OUT IN FAKE in backward_G_only")

        pred_fake_OCR = self.netOCR(self.fake)
        preds_size = torch.IntTensor([pred_fake_OCR.size(0)] * self.batch_size).detach()
        loss_OCR_fake = self.OCR_criterion(pred_fake_OCR, self.text_encode_fake.detach(), preds_size, self.len_text_fake.detach())
        self.loss_OCR_fake = torch.mean(loss_OCR_fake[~torch.isnan(loss_OCR_fake)])

        self.loss_G = self.loss_G + self.Lcycle1 + self.Lcycle2 + self.lda1 + self.lda2 - self.KLD

        self.loss_T = self.loss_G + self.loss_OCR_fake



        grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, retain_graph=True)[0]


        self.loss_grad_fake_OCR = 10**6*torch.mean(grad_fake_OCR**2)
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, retain_graph=True)[0]
        self.loss_grad_fake_adv = 10**6*torch.mean(grad_fake_adv**2)


        self.loss_T.backward(retain_graph=True)


        grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=True, retain_graph=True)[0]
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=True, retain_graph=True)[0]


        a = self.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon+torch.std(grad_fake_OCR))


        # if a is None:
        #     print(self.loss_OCR_fake, self.loss_G, torch.std(grad_fake_adv), torch.std(grad_fake_OCR))
        # if a>1000 or a<0.0001:
        #     print(a)


        self.loss_OCR_fake = a.detach() * self.loss_OCR_fake

        self.loss_T = self.loss_G + self.loss_OCR_fake


        self.loss_T.backward(retain_graph=True)
        grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=False, retain_graph=True)[0]
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=False, retain_graph=True)[0]
        self.loss_grad_fake_OCR = 10 ** 6 * torch.mean(grad_fake_OCR ** 2)
        self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)

        with torch.no_grad():
            self.loss_T.backward()

        # if any(torch.isnan(loss_OCR_fake)) or torch.isnan(self.loss_G):
        #     print('loss OCR fake: ', loss_OCR_fake, ' loss_G: ', self.loss_G, ' words: ', self.words)
        #     sys.exit()

    def backward_G_WL(self):

        self.gb_alpha = 0.7
        #self.Lcycle1 = self.Lcycle1.mean()
        #self.Lcycle2 = self.Lcycle2.mean()

        # print("FAKE in backward_G_WL")
        self.loss_G = loss_hinge_gen(self.netD(self.fake), self.len_text_fake.detach(), True).mean()

        self.loss_w_fake = self.netW(self.fake, self.wcl).mean()

        self.loss_G = self.loss_G + self.Lcycle1 + self.Lcycle2 + self.lda1 + self.lda2 - self.KLD

        self.loss_T = self.loss_G + self.loss_w_fake


        self.loss_T.backward(retain_graph=True)


        grad_fake_WL = torch.autograd.grad(self.loss_w_fake, self.fake, create_graph=True, retain_graph=True)[0]
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=True, retain_graph=True)[0]


        a = self.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon+torch.std(grad_fake_WL))

        # if a is None:
        #     print(self.loss_w_fake, self.loss_G, torch.std(grad_fake_adv), torch.std(grad_fake_WL))
        # if a>1000 or a<0.0001:
        #     print(a)

        self.loss_w_fake = a.detach() * self.loss_w_fake

        self.loss_T = self.loss_G + self.loss_w_fake

        self.loss_T.backward(retain_graph=True)
        grad_fake_WL = torch.autograd.grad(self.loss_w_fake, self.fake, create_graph=False, retain_graph=True)[0]
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=False, retain_graph=True)[0]
        self.loss_grad_fake_WL = 10 ** 6 * torch.mean(grad_fake_WL ** 2)
        self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)

        with torch.no_grad():
            self.loss_T.backward()

    def backward_G(self):
        self.gb_alpha = 0.7
        # print("FAKE IN backward_G")
        self.loss_G = loss_hinge_gen(self.netD(self.fake), self.len_text_fake.detach(), mask_loss=True)
        # OCR loss on real data

        pred_fake_OCR = self.netOCR(self.fake)
        preds_size = torch.IntTensor([pred_fake_OCR.size(0)] * self.batch_size).detach()
        loss_OCR_fake = self.OCR_criterion(pred_fake_OCR, self.text_encode_fake.detach(), preds_size, self.len_text_fake.detach())
        self.loss_OCR_fake = torch.mean(loss_OCR_fake[~torch.isnan(loss_OCR_fake)])

        self.loss_w_fake = self.netW(self.fake, self.wcl)
        #self.loss_OCR_fake = self.loss_OCR_fake + self.loss_w_fake
        # total loss

       # l1 = self.params[0]*self.loss_G
       # l2 = self.params[0]*self.loss_OCR_fake
        #l3 = self.params[0]*self.loss_w_fake

        self.loss_G_ = 10 * self.loss_G + self.loss_w_fake
        self.loss_T = self.loss_G_ + self.loss_OCR_fake

        grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, retain_graph=True)[0]

        self.loss_grad_fake_OCR = 10 ** 6 * torch.mean(grad_fake_OCR ** 2)
        grad_fake_adv = torch.autograd.grad(self.loss_G_, self.fake, retain_graph=True)[0]
        self.loss_grad_fake_adv = 10 ** 6*torch.mean(grad_fake_adv ** 2)

        self.loss_T.backward(retain_graph=True)

        grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=True, retain_graph=True)[0]
        grad_fake_adv = torch.autograd.grad(self.loss_G_, self.fake, create_graph=True, retain_graph=True)[0]
        #grad_fake_wl = torch.autograd.grad(self.loss_w_fake, self.fake, create_graph=True, retain_graph=True)[0]


        a = self.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon + torch.std(grad_fake_OCR))


        #a0 = self.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon+torch.std(grad_fake_wl))

        # if a is None:
        #     print(self.loss_OCR_fake, self.loss_G_, torch.std(grad_fake_adv), torch.std(grad_fake_OCR))
        # if a>1000 or a<0.0001:
        #     print(a)
        # b = self.gb_alpha * (torch.mean(grad_fake_adv) -
        #                                 torch.div(torch.std(grad_fake_adv), self.epsilon+torch.std(grad_fake_OCR))*
        #                                 torch.mean(grad_fake_OCR))
        # self.loss_OCR_fake = a.detach() * self.loss_OCR_fake + b.detach() * torch.sum(self.fake)
        self.loss_OCR_fake = a.detach() * self.loss_OCR_fake
        #self.loss_w_fake = a0.detach() * self.loss_w_fake

        # loss weight
        onlyOCR = 1
        self.loss_T = (1 - onlyOCR) * self.loss_G_ + self.loss_OCR_fake # + self.loss_w_fake
        self.loss_T.backward(retain_graph=True)

        grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=False, retain_graph=True)[0]
        grad_fake_adv = torch.autograd.grad(self.loss_G_, self.fake, create_graph=False, retain_graph=True)[0]
        self.loss_grad_fake_OCR = 10 ** 6 * torch.mean(grad_fake_OCR ** 2)
        self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)
        with torch.no_grad():
            self.loss_T.backward()

        # ???
        # if self.opt.clip_grad > 0:
        #      clip_grad_norm_(self.netG.parameters(), self.opt.clip_grad)

        # if any(torch.isnan(loss_OCR_fake)) or torch.isnan(self.loss_G_):
        #     print('loss OCR fake: ', loss_OCR_fake, ' loss_G: ', self.loss_G, ' words: ', self.words)
        #     sys.exit()

    def optimize_D_OCR(self):
        self.forward()
        self.set_requires_grad([self.netD], True)
        self.set_requires_grad([self.netOCR], True)
        self.optimizer_D.zero_grad()

        #if self.opt.OCR_init in ['glorot', 'xavier', 'ortho', 'N02']:
        self.optimizer_OCR.zero_grad()
        self.backward_D_OCR()

    def optimize_D_OCR_step(self):
        self.optimizer_D.step()

        self.optimizer_OCR.step()
        self.optimizer_D.zero_grad()
        self.optimizer_OCR.zero_grad()

    def optimize_G_WL(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], False)
        self.backward_G_WL()

    def optimize_G_only(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], False)
        self.backward_G_only()

    def optimize_G_step(self):
        self.optimizer_G.step()
        self.optimizer_G.zero_grad()
