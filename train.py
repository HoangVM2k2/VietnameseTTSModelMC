import os
import time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import commons
import utils
from data_utils import (
    TextAudioLoader, 
    TextAudioCollate,
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler,
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import get_symbols


torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()


def should_stop(time_limit) -> bool:
    if time_limit is None:
        return False
    
    global start_time
    duration = time.time() - start_time
    if duration >= time_limit:
        return True
    return False


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    # os.environ["MASTER_PORT"] = str(random.randint(0, 65535))
    os.environ["MASTER_PORT"] = "12345"

    hps = utils.get_hparams()
    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )


def run(rank, n_gpus, hps):
    writer, writer_eval = None, None
    if rank == 0:
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    try:
        _run(rank, n_gpus, hps, writer, writer_eval)
    except KeyboardInterrupt as e:
        if rank == 0:
            print(e.args)
    finally:
        if rank == 0:
            writer.close()
            writer_eval.close()


def _run(rank, n_gpus, hps, writer, writer_eval):
    print("| Start at rank", rank)
    
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(utils.beautify_hparams(hps.to_dict()))
        utils.check_git_hash(hps.model_dir)

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    if hps.mode == "single":
        TargetDataset = TextAudioLoader
        TargetCollator = TextAudioCollate
    else:
        TargetDataset = TextAudioSpeakerLoader
        TargetCollator = TextAudioSpeakerCollate

    if rank == 0:
        logger.info("| Loading training dataset")
    train_dataset = TargetDataset(hps.data.training_files, hps.data)
    if rank == 0:
        logger.info("| Loading data sampler")
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    if rank == 0:
        logger.info("| Loading data collator")
    collate_fn = TargetCollator()
    if rank == 0:
        logger.info("| Loading data loader")
    train_loader = DataLoader(
        train_dataset,
        num_workers=hps.train.num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )
    if rank == 0:
        logger.info("| Loading evaluation dataset")
        eval_dataset = TargetDataset(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=hps.train.num_workers,
            shuffle=False,
            batch_size=hps.train.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    if rank == 0:
        logger.info("| Loading models")

    if hps.mode == "single":
        n_speakers = 0
    else:
        n_speakers = hps.data.n_speakers

    net_g = SynthesizerTrn(
        len(get_symbols(hps.data.lang)),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=n_speakers,
        **hps.model
    ).cuda(rank)

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])

    if rank == 0:
        logger.info("| Try to restore checkpoints")

    try:
        _, _, _, start_epoch, global_step = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        _, _, _, start_epoch, global_step = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )
    except:
        start_epoch = 1
        global_step = 0

    if rank == 0:
        logger.info(f"| Start training from epoch {start_epoch}, step {global_step}")

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=start_epoch - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=start_epoch - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(start_epoch, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, eval_loader],
                logger,
                [writer, writer_eval],
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
            )
        if rank == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                global_step,
                os.path.join(hps.model_dir, "G_end_epoch.pth".format(global_step)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                global_step,
                os.path.join(hps.model_dir, "D_end_epoch.pth".format(global_step)),
            )
        scheduler_g.step()
        scheduler_d.step()

        if should_stop(hps.train.get("time_limit")):
            return


def train_and_evaluate(
    rank, 
    epoch, 
    hps, 
    nets: tuple[SynthesizerTrn, MultiPeriodDiscriminator], 
    optims: tuple[torch.optim.AdamW, torch.optim.AdamW], 
    schedulers: tuple[torch.optim.lr_scheduler.ExponentialLR, torch.optim.lr_scheduler.ExponentialLR], 
    scaler: GradScaler, 
    loaders, 
    logger, 
    writers
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    optim_g.zero_grad()
    optim_d.zero_grad()
    grad_norm_d = 0
    grad_norm_g = 0
    max_norm = torch.inf if hps.train.max_norm is None else hps.train.max_norm

    for batch_idx, batch in enumerate(train_loader):
        if hps.mode == "single":
            x, x_lengths, spec, spec_lengths, y, y_lengths = batch
        else:
            x, x_lengths, spec, spec_lengths, y, y_lengths, speakers = batch

        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(
            rank, non_blocking=True
        )
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(
            rank, non_blocking=True
        )
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
            rank, non_blocking=True
        )
        if hps.mode == "multiple":
            speakers = speakers.cuda(rank, non_blocking=True)
        else:
            speakers = None

        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
            ) = net_g(x, x_lengths, spec, spec_lengths, sid=speakers)

            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
                loss_disc_all /= hps.train.accumulation_steps

        scaler.scale(loss_disc_all).backward()

        if (global_step + 1) % hps.train.accumulation_steps == 0:
            if hps.train.use_clip_grad_norm is True:
                scaler.unscale_(optim_d)
                # grad_norm_d = commons.clip_grad_value_(net_d.parameters(), hps.train.max_norm)
                grad_norm_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), max_norm)
            scaler.step(optim_d)

        # Train generator
        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                loss_gen_all /= hps.train.accumulation_steps

        scaler.scale(loss_gen_all).backward()

        if (global_step + 1) % hps.train.accumulation_steps == 0:
            if hps.train.use_clip_grad_norm is True:
                scaler.unscale_(optim_g)
                # grad_norm_g = commons.clip_grad_value_(net_g.parameters(), hps.train.max_norm)
                grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm)
            scaler.step(optim_g)
            scaler.update()

            optim_g.zero_grad()
            optim_d.zero_grad()
            
            scheduler_g.step()
            scheduler_d.step()

        # Logging and saving checkpoint
        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/dur": loss_dur,
                        "loss/g/kl": loss_kl,
                    }
                )

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    "all/attn": utils.plot_alignment_to_numpy(
                        attn[0, 0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            if should_stop(hps.train.time_limit):
                global_step += 1
                return

            if global_step % hps.train.eval_interval == 0 and global_step != 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    global_step,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    global_step,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
        global_step += 1

        if should_stop(hps.train.time_limit):
            return

    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if hps.mode == "single":
                x, x_lengths, spec, spec_lengths, y, y_lengths = batch
            else:
                x, x_lengths, spec, spec_lengths, y, y_lengths, speakers = batch

            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)
            if hps.mode == "multiple":
                speakers = speakers.cuda(0)
            else:
                speakers = None
            
            # remove else
            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]
            if hps.mode == "multiple":
                speakers = speakers[:1]
            break

        y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, speakers, max_len=1000)
        y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
    image_dict = {
        "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {"gen/audio": y_hat[0, :, : y_hat_lengths[0]]}
    if global_step == 0:
        image_dict.update(
            {"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())}
        )
        audio_dict.update({"gt/audio": y[0, :, : y_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()


if __name__ == "__main__":
    main()
