import torch
import itertools
from util.image_pool import ImageMaskPool, ImagePool
from .base_model import BaseModel
from . import networks 
from . import uag_networks as uag
from STEGO.src.train_segmentation import LitUnsupervisedSegmenter
from STEGO.src.crf import dense_crf
from STEGO.src.utils import unnorm, remove_axes
import matplotlib.pyplot as plt

class UAGGANModel(BaseModel):
    '''
      An implement of the UAGGAN model.
  
      Paper: Unsupervised Attention-guided Image-to-Image Translation, NIPS 2018.
             https://arxiv.org/pdf/1806.02311.pdf
    '''
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--lambda_B', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')
        return parser

    def __init__(self, opt):
        """Initialize the UAGGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'D_B', 'G_A', 'G_B', 'cycle_A', 'cycle_B']
        self.visual_names = ['real_A', 'att_A_viz', 'fake_B', 'masked_fake_B', 
                             'real_B', 'att_B_viz', 'fake_A', 'masked_fake_A']
        if self.isTrain:
            self.model_names = ['G_att_A', 'G_att_B', 'G_img_A', 'G_img_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_att_A', 'G_att_B', 'G_img_A', 'G_img_B']
        

        self.netG_att_A = uag.define_net_att(opt.input_nc,
                                             opt.ngf,
                                             norm=opt.norm,
                                             init_type=opt.init_type,
                                             init_gain=opt.init_gain,
                                             gpu_ids=opt.gpu_ids)

        self.netG_att_B = uag.define_net_att(opt.input_nc,
                                             opt.ngf,
                                             norm=opt.norm,
                                             init_type=opt.init_type,
                                             init_gain=opt.init_gain,
                                             gpu_ids=opt.gpu_ids)

        self.netG_img_A = uag.define_net_img(opt.input_nc,
                                             opt.output_nc,
                                             opt.ngf,
                                             norm=opt.norm,
                                             init_type=opt.init_type,
                                             init_gain=opt.init_gain,
                                             gpu_ids=opt.gpu_ids)
        
        self.netG_img_B = uag.define_net_img(opt.input_nc,
                                             opt.output_nc,
                                             opt.ngf,
                                             norm=opt.norm,
                                             init_type=opt.init_type,
                                             init_gain=opt.init_gain,
                                             gpu_ids=opt.gpu_ids)


        if self.isTrain:
            self.netD_A = uag.define_net_dis(opt.input_nc,
                                             opt.ndf,
                                             norm=opt.norm,
                                             init_type=opt.init_type,
                                             init_gain=opt.init_gain,
                                             gpu_ids=opt.gpu_ids)

            self.netD_B = uag.define_net_dis(opt.input_nc,
                                             opt.ndf,
                                             norm=opt.norm,
                                             init_type=opt.init_type,
                                             init_gain=opt.init_gain,
                                             gpu_ids=opt.gpu_ids)

            self.stego_model = LitUnsupervisedSegmenter.load_from_checkpoint(opt.stego).cuda()
            
            print(self.stego_model)
        
        
        
        if self.isTrain:
            if self.opt.use_mask_for_D:
                self.masked_fake_A_pool = ImageMaskPool(opt.pool_size)
                self.masked_fake_B_pool = ImageMaskPool(opt.pool_size)
            else:
                self.masked_fake_A_pool = ImagePool(opt.pool_size)
                self.masked_fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                    self.netG_att_A.parameters(), 
                    self.netG_att_B.parameters(),
                    self.netG_img_A.parameters(), 
                    self.netG_img_B.parameters()), 
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(
                    self.netD_A.parameters(), 
                    self.netD_B.parameters()), 
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # G(A) -> B
        #genero immagine fake nel modo classico
        self.fake_B = self.netG_img_A(self.real_A)
        
        #utilizzo stego per il background subtraction (img_fake*mask + (1-mask)*img_real) ---- (1-mask)=background
        with torch.no_grad():
          self.code_A = self.stego_model(self.real_A)
          self.linear_probs_A = torch.log_softmax(self.stego_model.linear_probe(self.code_A), dim=1)
          self.single_img_A = self.real_A[0]
          self.linear_pred_A = dense_crf(self.single_img_A, self.linear_probs_A[0]).argmax(0)
          self.mask_A = (self.linear_pred_A == 7)*1
          #ho la maschera, la converto in pytorch e genero quindi l'attenzione
          self.att_A = torch.tensor(self.mask_A).cuda()
        #calcolo quindi l'immagine fake con il background subtraction
        self.masked_fake_B = self.fake_B*self.att_A + self.real_A*(1-self.att_A)
        #dato che l'attenzione (la maschera) è una rete pre addestrata, è necessario calcolarla solo
        #una volta, nel ciclo di ricostruzione dell'immagine iniziale sarà infatti la stessa maschera
        #che verrà applicata (prima la ricalcolavo perchè avevo due reti)
        
        # cycle G(G(A)) -> A
        #self.cycle_att_B = self.netG_att_B(self.masked_fake_B)
        self.cycle_att_B = self.att_A #per quanto spiegato prima
        self.cycle_fake_A = self.netG_img_B(self.masked_fake_B)
        self.cycle_masked_fake_A = self.cycle_fake_A*self.cycle_att_B + self.masked_fake_B*(1-self.cycle_att_B)


        # G(B) -> A
        self.fake_A = self.netG_img_B(self.real_B)

        with torch.no_grad():
          self.code_B = self.stego_model(self.real_B)
          self.linear_probs_B = torch.log_softmax(self.stego_model.linear_probe(self.code_B), dim=1)
          self.single_img_B = self.real_B[0]
          self.linear_pred_B = dense_crf(self.single_img_B, self.linear_probs_B[0]).argmax(0)
          self.mask_B = (self.linear_pred_B == 7)*1
          #ho la maschera, la converto in pytorch e genero quindi l'attenzione
          self.att_B = torch.tensor(self.mask_B).cuda()

        self.masked_fake_A = self.fake_A*self.att_B + self.real_B*(1-self.att_B)

        # cycle G(G(B)) -> B
        #self.cycle_att_A = self.netG_att_A(self.masked_fake_A)
        self.cycle_att_A = self.att_B
        self.cycle_fake_B = self.netG_img_A(self.masked_fake_A)
        self.cycle_masked_fake_B = self.cycle_fake_B*self.cycle_att_A + self.masked_fake_A*(1-self.cycle_att_A)

        # just for visualization
        self.att_A_viz, self.att_B_viz = (torch.reshape(self.att_A,(1,1,256,256))-0.5)/0.5, (torch.reshape(self.att_B,(1,1,256,256))-0.5)/0.5


        # Solo per visualizzare durante debugging iniziale
        # with torch.no_grad():
        #     fig, ax = plt.subplots(1,4, figsize=(5*3,5))
        #     ax[0].imshow(unnorm(self.real_A).squeeze().permute(1,2,0).cpu().numpy())
        #     ax[0].set_title("original")
        #     ax[1].imshow(self.att_A.cpu().numpy())
        #     ax[1].set_title("mask")
        #     ax[2].imshow(unnorm(self.masked_fake_B)[0].permute(1,2,0).cpu())
        #     ax[2].set_title("masked")
        #     ax[3].imshow(unnorm(self.cycle_masked_fake_A)[0].permute(1,2,0).cpu())
        #     ax[3].set_title("cycle masked")
        #     remove_axes(ax)
        #     fig.savefig('A-B.png')
        # with torch.no_grad():
        #     fig, ax = plt.subplots(1,4, figsize=(5*3,5))
        #     ax[0].imshow(unnorm(self.real_B).squeeze().permute(1,2,0).cpu().numpy())
        #     ax[0].set_title("original")
        #     ax[1].imshow(self.att_B.cpu().numpy())
        #     ax[1].set_title("mask")
        #     ax[2].imshow(unnorm(self.masked_fake_A)[0].permute(1,2,0).cpu())
        #     ax[2].set_title("masked")
        #     ax[3].imshow(unnorm(self.cycle_masked_fake_B)[0].permute(1,2,0).cpu())
        #     ax[3].set_title("cycle masked")
        #     remove_axes(ax)
        #     fig.savefig('B-A.png')

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        if self.opt.use_mask_for_D:
            masked_fake_B, att_A = self.masked_fake_B_pool.query(self.masked_fake_B, self.att_A)
            masked_fake_B *= att_A
        else:
            masked_fake_B = self.masked_fake_B_pool.query(self.masked_fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, masked_fake_B)

        if self.opt.use_mask_for_D:
            masked_fake_A, att_B = self.masked_fake_A_pool.query(self.masked_fake_A, self.att_B)
            masked_fake_A *= att_B
        else:
            masked_fake_A = self.masked_fake_A_pool.query(self.masked_fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, masked_fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # GAN loss D_A(G(A))
        masked_fake_B = self.masked_fake_B
        if self.opt.use_mask_for_D:
            masked_fake_B *= self.att_A
        self.loss_G_A = self.criterionGAN(self.netD_A(self.masked_fake_B), True)
        # GAN loss D_B(G(B))
        masked_fake_A = self.masked_fake_A
        if self.opt.use_mask_for_D:
            masked_fake_A *= self.att_B
        self.loss_G_B = self.criterionGAN(self.netD_B(self.masked_fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.cycle_masked_fake_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.cycle_masked_fake_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
        self.loss_G.backward()
    
    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        nets = [self.netD_A, self.netD_B]
        if epoch > 60 and self.opt.use_early_stopping:
            nets += [self.netG_att_A, self.netG_att_B] # Ds require no gradients when optimizing Gs
        self.set_requires_grad(nets, False)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()        # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights