import wandb
from comet_ml import start
from comet_ml.integration.pytorch import log_model
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["WANDB_API_KEY"] = ""

from pathlib import Path
import time
from data.dataset import TextDataset, TextDatasetval
from models import create_model
import torch
import cv2
import os
import numpy as np
from itertools import cycle
from scipy import linalg
from models.model import TRGAN
from params import *
from torch import nn


experiment = start(
  api_key="6xk1Nmcm6P2OmkiUlYSqe4IqV",
  project_name="bangla-handwriting",
  workspace="rabib-jahin"
)
def freeze_for_finetuning(model):
    """
    Freezes the entire TRGAN model except for the core encoder-decoder
    pipeline within the Generator (netG).
    
    This keeps the following modules trainable:
    - netG.Feat_Encoder
    - netG.encoder (Transformer Encoder)
    - netG.decoder (Transformer Decoder)
    - netG.DEC (FCN Decoder)
    
    And freezes:
    - netD (Discriminator)
    - netW (WDiscriminator)
    - netOCR (OCR Network)
    - Other small layers in netG
    """
    # First, freeze all parameters in the entire model
    for param in model.parameters():
        param.requires_grad = False

    # Then, unfreeze only the parameters of the generator's encoder-decoder parts
    for name, param in model.named_parameters():
        if 'netG.Feat_Encoder' in name or \
           'netG.encoder' in name or \
           'netG.decoder' in name or \
           'netG.DEC' in name:
            param.requires_grad = True
def main():

    # wandb.init(project="bangla-final", name = EXP_NAME)

    init_project()
    model_path = 'model450_pretrained_BN.pth'
    torch.cuda.empty_cache()

    TextDatasetObj = TextDataset(num_examples = NUM_EXAMPLES)
    dataset = torch.utils.data.DataLoader(
                TextDatasetObj,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                
                pin_memory=True, drop_last=True,
                collate_fn=TextDatasetObj.collate_fn)

    TextDatasetObjval = TextDatasetval(num_examples = NUM_EXAMPLES)
    datasetval = torch.utils.data.DataLoader(
                TextDatasetObjval,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True, drop_last=True,
                collate_fn=TextDatasetObjval.collate_fn)

    model = TRGAN()
    # model.load_state_dict(torch.load(model_path))

    os.makedirs('saved_models', exist_ok = True)
    MODEL_PATH = os.path.join('saved2/saved_models', EXP_NAME)
    MODEL_PATH2 = os.path.join('saved_models', EXP_NAME)
    if os.path.isdir(MODEL_PATH2) and RESUME: 
        model.load_state_dict(torch.load(MODEL_PATH2+'/model.pth'))
        
        print (MODEL_PATH2+' : Model loaded Successfully')
    else: 
        if not os.path.isdir(MODEL_PATH): os.mkdir(MODEL_PATH)


    # freeze_for_finetuning(model)
# 
    model.save_images_for_fid_calculation(dataset, 1000)

    return

    for epoch in range(EPOCHS):    


        

        start_time = time.time()
        
        for i,data in enumerate(dataset): 

            if (i % NUM_CRITIC_GOCR_TRAIN) == 0:

                model._set_input(data)
                model.optimize_G_only()
                model.optimize_G_step()

            if (i % NUM_CRITIC_DOCR_TRAIN) == 0:

                model._set_input(data)
                model.optimize_D_OCR()
                model.optimize_D_OCR_step()

            if (i % NUM_CRITIC_GWL_TRAIN) == 0:

                model._set_input(data)
                model.optimize_G_WL()
                model.optimize_G_step()
# 
            if (i % NUM_CRITIC_DWL_TRAIN) == 0:

                model._set_input(data)
                model.optimize_D_WL()
                model.optimize_D_WL_step()

        end_time = time.time()
        data_val = next(iter(datasetval))
        losses = model.get_current_losses()
        page = model._generate_page(model.sdata, model.input['swids'])
        page_val = model._generate_page(data_val['simg'].to(DEVICE), data_val['swids'])
        experiment.log_metric("loss-G", losses['G'], step=epoch)
        experiment.log_metric("loss-D", losses['D'], step=epoch)
        experiment.log_metric("loss-Dfake", losses['Dfake'], step=epoch)
        experiment.log_metric("loss-Dreal", losses['Dreal'], step=epoch)
        experiment.log_metric("loss-OCR_fake", losses['OCR_fake'], step=epoch)
        experiment.log_metric("loss-OCR_real", losses['OCR_real'], step=epoch)
        experiment.log_metric("loss-w_fake", losses['w_fake'], step=epoch)
        experiment.log_metric("loss-w_real", losses['w_real'], step=epoch)
        experiment.log_metric("epoch", epoch, step=epoch)
        experiment.log_metric("timeperepoch", end_time - start_time, step=epoch)

  
        
        # wandb.log({'loss-G': losses['G'],
        #             'loss-D': losses['D'], 
        #             'loss-Dfake': losses['Dfake'],
        #             'loss-Dreal': losses['Dreal'],
        #             'loss-OCR_fake': losses['OCR_fake'],
        #             'loss-OCR_real': losses['OCR_real'],
        #             'loss-w_fake': losses['w_fake'],
        #             'loss-w_real': losses['w_real'],
        #             'epoch' : epoch,
        #             'timeperepoch': end_time-start_time,
                    
        #             })

                    
        
        # wandb.log({ "result":[wandb.Image(page, caption="page"),wandb.Image(page_val, caption="page_val")],
        #             })
        experiment.log_image(page, name="page")
        experiment.log_image(page_val, name="page_val")

        

        print ({'EPOCH':epoch, 'TIME':end_time-start_time, 'LOSSES': losses})

        if epoch % SAVE_MODEL == 0: torch.save(model.state_dict(), MODEL_PATH2+ '/model.pth')
        if epoch % SAVE_MODEL_HISTORY == 0: torch.save(model.state_dict(), MODEL_PATH2+ '/model'+str(epoch)+'.pth')


if __name__ == "__main__":
    
    main()
