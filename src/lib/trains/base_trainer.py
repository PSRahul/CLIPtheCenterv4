from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter
from models.clip.clip_model import CLIPModel
from models.decode import ctdet_decode
from models.clip.embedder import Embedder
class ModelWithLoss(torch.nn.Module):
  def __init__(self, model, loss,opt,clip_model=None,embedder=None):
    super(ModelWithLoss, self).__init__()
    self.model = model
    self.loss = loss
    self.opt=opt
    self.clip_model=clip_model
    self.embedder=embedder
  
  def forward(self, batch):
    outputs = self.model(batch['input'])

    outputs=outputs[0]
    if self.opt.clip_encoder:
      with torch.no_grad():
        hm = outputs['hm'].detach().sigmoid()
        dets = ctdet_decode(
          heat=hm, wh=outputs['wh'].detach(), reg=outputs['reg'].detach(),
          cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.clip_topk)
        dets = dets.reshape((dets.shape[0] * dets.shape[1], dets.shape[2]))
        dets[:, :4] *= self.opt.down_ratio
        clip_embedding, dets = self.clip_model(batch, dets)
        dets[:, :4] /= self.opt.down_ratio
      model_embedding = self.embedder(outputs['clip'], dets)
      outputs["model_embedding"]=model_embedding
      outputs["clip_embedding"]=clip_embedding
    outputs=[outputs]
    loss, loss_stats = self.loss(outputs, batch)
    return outputs, loss, loss_stats

class BaseTrainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.clip_model = None
    self.embedder = None
    if opt.clip_encoder:
      self.clip_model=CLIPModel(self.opt)
      self.embedder=Embedder(self.opt)
      self.clip_model.to("cuda")
      self.embedder.to("cuda")
    self.model_with_loss = ModelWithLoss(model, self.loss, opt,self.clip_model,self.embedder)

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.model_with_loss = DataParallel(
        self.model_with_loss, device_ids=gpus, 
        chunk_sizes=chunk_sizes).to(device)
    else:
      self.model_with_loss = self.model_with_loss.to(device)
    
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader):
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)

      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)    
      output, loss, loss_stats = model_with_loss(batch)


      loss = loss.mean()
      if phase == 'train':
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['input'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      if not opt.hide_data_time:
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
      else:
        bar.next()
      
      if opt.debug > 0:
        self.debug(batch, output, iter_id)
      
      if opt.test:
        self.save_result(output, batch, results)
      del output, loss, loss_stats
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results
  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)