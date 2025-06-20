# Adapted from: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any
from dataclasses import dataclass, asdict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from audiotools import AudioSignal
from torchmetrics import MinMetric, MeanMetric

from src.models.moshi.modules import VocosBackbone, StreamingLogMelSpectrogram, ISTFTHead
from src.models.components.discriminator import Discriminator
from src.models.components.losses import L1Loss, MultiScaleSTFTLoss, MelSpectrogramLoss, GANLoss, LossPED, HuBERTLoss, SpeakerLoss
from src.models.utils.data_classes import VocoderFeatureArgs, VocosVocoderArgs, ISTFTHeadArgs, Lambdas



class VocosVocoderModule(pl.LightningModule):
    """Example of LightningModule for MNIST classification.
    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)
    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        encoder_dim: int = 512,
        vocoder_feature_args: VocoderFeatureArgs = VocoderFeatureArgs(),
        vocoder_args: VocosVocoderArgs = VocosVocoderArgs(),
        istft_head_args: ISTFTHeadArgs = ISTFTHeadArgs(),
        lambdas: Lambdas = Lambdas(),
        learning_rate: float = 0.0001,
        lr_betas: list = [0.8, 0.99],
        lr_sch_gamma: float = 0.999996,
        decoder_pretrained: bool = False,
        decoder_path: str = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        # Multiple optimizers need manual optimization
        self.automatic_optimization = False

        self.feature_extractor = StreamingLogMelSpectrogram(**asdict(vocoder_feature_args))
        vocos_backbone = VocosBackbone(**asdict(vocoder_args))
        istft_head = ISTFTHead(**asdict(istft_head_args))
        self.decoder = nn.Sequential(
            vocos_backbone,
            istft_head
        )
        
        ## Discriminators
        self.discriminator = Discriminator()

        # Load pretrained decoder/discriminator
        if self.hparams.decoder_pretrained:
            model_state_dict = torch.load(self.hparams.decoder_path, map_location=self.device)["state_dict"]
            # Filter out the keys that belong to the decoder.
            decoder_state = {
                k.replace("decoder.", ""): v
                for k, v in model_state_dict.items()
                if k.startswith("decoder.")
            }
            self.decoder.load_state_dict(decoder_state, strict=False)

            # Filter out the keys that belong to the discriminator.
            discriminator_state = {
                k.replace("discriminator.", ""): v
                for k, v in model_state_dict.items()
                if k.startswith("discriminator.")
            }
            self.discriminator.load_state_dict(discriminator_state, strict=False)

        # loss function
        self.lambdas = asdict(lambdas)
        self.waveform_loss = L1Loss()
        self.stft_loss = MultiScaleSTFTLoss()
        self.mel_loss = MelSpectrogramLoss()
        self.gan_loss = GANLoss(self.discriminator)
        # self.hubert_loss = HuBERTLoss(device=self.device)
        # self.speaker_loss = SpeakerLoss(device=self.device)


        # for averaging loss across batches
        loss_metric = MeanMetric()
        self.train_stft_loss = loss_metric.clone()
        self.train_mel_loss = loss_metric.clone()
        self.train_waveform_loss = loss_metric.clone()
        self.train_adv_gen_loss = loss_metric.clone()
        self.train_adv_feat_loss = loss_metric.clone()
        self.train_adv_d_loss = loss_metric.clone()
        # self.train_hubert_loss = loss_metric.clone()
        # self.train_speaker_loss = loss_metric.clone()


        self.val_stft_loss = loss_metric.clone()
        self.val_mel_loss = loss_metric.clone()
        self.val_waveform_loss = loss_metric.clone()

        self.test_loss = loss_metric.clone()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.decoder.parameters(), self.hparams.learning_rate, betas=self.hparams.lr_betas)
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), self.hparams.learning_rate, betas=self.hparams.lr_betas)

        sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.hparams.lr_sch_gamma)
        sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=self.hparams.lr_sch_gamma)
        return [opt_g, opt_d], [sch_g, sch_d]


    def forward(self, x: torch.Tensor):
        audio = self.decoder(x)
        return audio

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_loss_best.reset()

    def model_step(self, batch: Any):
        y, _ = batch
        y = y.unsqueeze(1)

        mel = self.feature_extractor(y)

        y_g_hat = self.decoder(mel)

        assert y_g_hat.shape == y.shape, f"Mismatch in vocoder output shape - {y_g_hat.shape} != {y.shape}"

        recons = AudioSignal(y_g_hat, self.hparams.sample_rate)
        target = AudioSignal(y, self.hparams.sample_rate)

        return recons, target


    def training_step(self, batch: Any, batch_idx: int):
        # Retrieve optimizers and lr schedulers
        opt_g, opt_d = self.optimizers()
        

        # Forward pass
        recons, signal = self.model_step(batch)
        batch_size = recons.shape[0]

        output = {}
        #### Discriminator Update ####
        opt_d.zero_grad()

        disc_loss = self.gan_loss.discriminator_loss(recons, signal)
        output["adv/d_loss"] = disc_loss
        
        # Manually backprop for discriminator
        self.manual_backward(disc_loss)
        opt_d.step()
        
        
        # Manually step the discriminator scheduler (if updating every step)
        # sch_d.step()

        #### Generator Update ####
        opt_g.zero_grad()

        # output["hubert_loss"] = self.hubert_loss(recons, signal)
        # output["speaker_loss"] = self.speaker_loss(recons, spkr_embed)
        output["stft_loss"] = self.stft_loss(recons, signal)
        output["mel_loss"] = self.mel_loss(recons, signal)
        output["waveform_loss"] = self.waveform_loss(recons, signal)
        (
            output["adv_gen_loss"],
            output["adv_feat_loss"],
        ) = self.gan_loss.generator_loss(recons, signal)
        gen_loss = sum([v * output[k] for k, v in self.lambdas.items() if k in output])
        output["loss"] = gen_loss
        
        # Manually backprop for generator
        self.manual_backward(gen_loss)
        opt_g.step()
        
        # Manually step the generator scheduler (if updating every step)
        # sch_g.step()

        
        # update and log metrics
        self.train_stft_loss(output["stft_loss"])
        self.train_mel_loss(output["mel_loss"])
        self.train_waveform_loss(output["waveform_loss"])
        self.train_adv_gen_loss(output["adv_gen_loss"])
        self.train_adv_feat_loss(output["adv_feat_loss"])
        self.train_adv_d_loss(output["adv/d_loss"])
        # self.train_hubert_loss(output["hubert_loss"])
        # self.train_speaker_loss(output["speaker_loss"])

        self.log("train/stft/loss", self.train_stft_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train/mel/loss", output["mel_loss"], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/waveform/loss", self.train_waveform_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train/adv/gen_loss", self.train_adv_gen_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train/adv/feat_loss", self.train_adv_feat_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train/total_gen_loss", output["loss"], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/adv/d_loss", self.train_adv_d_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        # self.log("train/hubert_loss", self.train_hubert_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        # self.log("train/speaker_loss", self.train_speaker_loss, on_step=False, on_epoch=True, batch_size=batch_size)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return output


    def on_train_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs
        sch_g, sch_d = self.lr_schedulers()
        sch_g.step()
        sch_d.step()


    def validation_step(self, batch: Any, batch_idx: int):
        recons, signal = self.model_step(batch)
        batch_size = recons.shape[0]

        # update and log metrics
        stft_loss = self.stft_loss(recons, signal)
        mel_loss = self.mel_loss(recons, signal)
        waveform_loss = self.waveform_loss(recons, signal)
        loss = mel_loss

        self.val_stft_loss(stft_loss)
        self.val_mel_loss(mel_loss)
        self.val_waveform_loss(waveform_loss)

        self.log("val/stft/loss", self.val_stft_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val/mel/loss", self.val_mel_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val/waveform/loss", self.val_waveform_loss, on_step=False, on_epoch=True, batch_size=batch_size)

        return {
            "loss": loss,
            "stft/loss": stft_loss,
            "mel/loss": mel_loss,
            "waveform/loss": waveform_loss
        }

    def on_validation_epoch_end(self):
        mel_loss = self.val_mel_loss.compute()  # get current val acc
        self.val_loss_best(mel_loss)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute())

    def test_step(self, batch: Any, batch_idx: int):
        recons, signal = self.model_step(batch)

        loss = self.mel_loss(recons, signal)
        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True)
        return {"loss": loss}

    def on_test_epoch_end(self):
        pass


if __name__ == "__main__":
    m = VocosVocoderModule()
    print(m)