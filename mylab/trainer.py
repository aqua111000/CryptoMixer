from .utils import setup_seed, count_parameters
import torch
from tqdm import tqdm
import numpy as np
import time


def evaluate(model, dataloader, args, loss_fn, metrics = {}, data_already_to_cuda = False, multi_class = False):
    
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    pred = []
    label = []
    
    if args.distribute:
        dataloader.sampler.set_epoch(0)
        
    with torch.no_grad():    
        with tqdm(total = len(dataloader)) as _tqdm:
            _tqdm.set_description('evaluate: ')  # 设置前缀 一般为epoch的信息

            [i.reset() for i in metrics.values()]
            cum_loss = 0
            cum_num = 0
            step = 0

            for batch in dataloader:

                if multi_class:
                    model_input = batch[:-2]
                    y_true = batch[-1]
                    y_class =  batch[-2]
                else:
                    model_input = batch[:-1]
                    y_true = batch[-1]
                    y_class =  y_true
                    
                if not data_already_to_cuda:
                    model_input = [i.to(device) for i in model_input]
                    y_true = y_true.to(device)
                    y_class = y_class.to(device)
                    
                y_pred = model(*model_input)

                loss = loss_fn(y_pred, y_true)

                if args.distribute:
                    loss = loss.mean()

                pred.append(y_pred)
                label.append(y_class)
                _tqdm.update(1)  # 设置你每一次想让进度条更新的iteration 大小

                step += 1
    result_metrics = {}
    pred = torch.cat(pred, dim = 0)
    label = torch.cat(label, dim = 0)
    loss = loss_fn(pred, label.float())
    for i in metrics.items():
        result_metrics['val_' + i[0]] = '{:.6f}'.format(i[1](pred, torch.argmax(label, dim = 1)))
    result_metrics['val_loss'] = '{:.6f}'.format(loss.detach().cpu().numpy())
    print(pred.shape)
    print(label.shape)
    print(result_metrics)
    return result_metrics


def train(model, dataloader, opt, loss_fn, args,
          val_dataloader = None,
          test_dataloader = None,
          lr_s = None, 
          early_stop_s = None,
          metrics = {},
          max_iter_for_debug = 20,
          debug = False,
          data_already_to_cuda = False,
          multi_class = False,
          loss_figure = None,
          print_alpha = False,
         ):
    
    setup_seed(args.seed)
    device = torch.device(args.device)
    model.to(device)
    if args.distribute:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.output_device], output_device=args.output_device, find_unused_parameters=True)
    
    params = count_parameters(model)
    print('Parameters of model: ', params)

    break_all = False

    if print_alpha:
        model.print_alpha()
        model.print_attn_weight()
        
    with torch.autograd.set_detect_anomaly(True):
    
        for epoch in range(args.epochs):
            
            opt.zero_grad()
            model.train()
            if args.distribute:
                dataloader.sampler.set_epoch(epoch)
                
            with tqdm(total = len(dataloader)) as _tqdm:
                _tqdm.set_description('epoch train: {}/{}'.format(epoch + 1, args.epochs))  # 设置前缀 一般为epoch的信息
                
                [i.reset() for i in metrics.values()]
                cum_loss = 0
                cum_num = 0
                step = 0
                
                model.train()
                for batch in dataloader:
    
                    if multi_class:
                        model_input = batch[:-2]
                        y_true = batch[-1]
                        y_class =  batch[-2]
                    else:
                        model_input = batch[:-1]
                        y_true = batch[-1]
                        y_class =  y_true
                        
                    if not data_already_to_cuda:
                        model_input = [i.to(device) for i in model_input]
                        y_true = y_true.to(device)
                        y_class = y_class.to(device)
                        
                    y_pred = model(*model_input)
                    
                    loss = loss_fn(y_pred, y_true)  
    
                    if args.distribute:
                        loss = loss.mean()
                    
                    loss.backward()
    
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
                    
                    opt.step()
                    opt.zero_grad()
                        
                    [i.update(y_pred, torch.argmax(y_class.int(), dim = 1)) for i in metrics.values()]
    
                    result_metrics = {}
                    for i in metrics.items():
                        result_metrics[i[0]] = '{:.6f}'.format(i[1].compute())
                    cum_loss += loss.item()
                    cum_num += 1
                    result_metrics['loss'] = cum_loss / cum_num
                    _tqdm.set_postfix(**result_metrics)
                    _tqdm.update(1)
                    step += 1
                    
                    if lr_s is not None:
                        lr_s.step(step / len(dataloader) + epoch)
    
    
                    if debug:
                        if step > max_iter_for_debug:
                            break_all = True
                            break
    
                if print_alpha:
                    model.print_alpha()
                    model.print_attn_weight()
                
                            
            if break_all:
                break
                #  torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            
            if val_dataloader is not None:
                val_result_metrics = evaluate(model, val_dataloader, args, loss_fn, metrics = metrics, data_already_to_cuda = data_already_to_cuda, multi_class = multi_class) 
                for i in val_result_metrics:
                    result_metrics[i] = val_result_metrics[i]
                if loss_figure is not None:
                    loss_figure.add_loss(result_metrics['val_loss'])
                    
            if lr_s is not None:
                print(lr_s.get_last_lr())
            
            if early_stop_s is not None:
                if early_stop_s(result_metrics, epoch):
                    break
            
    if early_stop_s is not None and (not debug):
        early_stop_s.restore()
    if loss_figure is not None:
        args.now_figure_path = loss_figure.draw_model()
    if not debug:
        print('start test')
        result_metrics = evaluate(model, test_dataloader, args, loss_fn, metrics = metrics, data_already_to_cuda = data_already_to_cuda, multi_class = multi_class)
        print(result_metrics)
        return result_metrics
    

def test_infer_time(model, dataloader, args):
    
    setup_seed(args.seed)
    device = torch.device(args.device)
    model.to(device)

    model.eval()

    repetitions = 300

    for batch in dataloader:
        model_input = batch[:-1]
        y_true = batch[-1]
        y_class =  y_true

        model_input = [i.to(device)[0:1] for i in model_input]
        y_true = y_true.to(device)
        y_class = y_class.to(device)
        break

    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = model(*model_input)
    
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))
    
    print('testing ...\n')
    with torch.no_grad():
        for rep in tqdm(range(repetitions)):
            starter.record()
            _ = model(*model_input)
            ender.record()
            torch.cuda.synchronize() # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
            timings[rep] = curr_time
    
    avg = timings.sum()/repetitions
        
    return avg
    