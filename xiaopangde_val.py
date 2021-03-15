def eval(self, epoch):
    torch.backends.cudnn.benchmark = True
    self.n_val = len(self.valset)
    self.print("{0:<22s} : {1:} ".format('valset sample', self.n_val))
    self.print("<-------------Evaluate the model-------------->")
    # Evaluate one epoch
    measures, fps = self.eval_epoch(epoch)
    acc = measures['a1']
    self.print('The {}th epoch, fps {:4.2f} | {}'.format(epoch, fps, measures))
    # Save the checkpoint
    self.save(epoch, acc)
    return measures

def eval_epoch(self, epoch):
    device = torch.device('cuda' if self.use_gpu else 'cpu')
    # self.net.to(device)
    self.net.cuda()
    # self.criterion.to(device)
    self.criterion.cuda()

    self.net.eval()
    val_total_time = 0
    # measure_list = ['a1', 'a2', 'a3', 'rmse', 'rmse_log', 'log10', 'abs_rel', 'sq_rel']
    measures = {key: 0 for key in measure_list}
    with torch.no_grad():
        sys.stdout.flush()
        tbar = tqdm(self.valloader)
        rand = np.random.randint(len(self.valloader))
        # if self.params.use_seg:
        score = self.eval_func_seg(self.params.seg_classes)
        score.reset()
        for step, data in enumerate(tbar):
            images, depths, labels = data[0].cuda(), data[1].cuda(), data[2]
            # forward
            before_op_time = time.time()
            pred = self.net(images)
            pred_depths = pred['depth_y']
            # y = self.net(images)
            if self.params.classifier == 'OR':
                pred_depths = self.net.inference(pred_depths)
            duration = time.time() - before_op_time
            val_total_time += duration
            # ratios = []
            # ratios.append(ratio)
            # pred_depths[i,:,:,:] *= ratio
            # # accuracy_depth
            pred_depths[pred_depths < 0] = 0
            pred_depths[pred_depths > 10.0] = 10.0
            pred_depths[torch.isinf(pred_depths)] = 10.0
            pred_depths[torch.isnan(pred_depths)] = 0

            new_depth = self.eval_func(depths, pred_depths)

            # accuracy_seg
            pred_labels = torch.argmax(pred['seg_y'], dim=1, keepdim=True)
            pred_labels = pred_labels.data.cpu().numpy()
            labels = labels.numpy()
            score.add_batch(labels, pred_labels)

            for k, v in new_depth.items():
                measures[k] += v.item()
            # for i in range(depths.shape[0]):
            #     pred_depths[i, :, :, :] /= ratios[i]
            # display images
            if step == rand and self.disp_func is not None:
                for i in range(depths.shape[0]):
                    print('the median of pred_depth: %f\n the median of depth: %f' % (torch.median(
                        pred_depths[i, :, :, :]).data, torch.median(depths[i, :, :, :]).data))
                    print('the maximum of pred_depth: %f\n the maximum of depth: %f' % (
                        torch.max(pred_depths[i, :, :, :]).data, torch.max(depths[i, :, :, :]).data))
                    print('the minimum of pred_depth: %f\n the minimum of depth: %f' % (
                        torch.min(pred_depths[i, :, :, :]).data, torch.min(depths[i, :, :, :]).data))
                visuals = {'inputs': images, 'seg_aff_map': pred['seg_aff_map'],
                           'depth_aff_map': pred['depth_aff_map'],
                           'labels': depths, 'depths': pred_depths, 'seg_labels': labels,
                           'segs': pred_labels}
                # visuals = {'inputs': images, 'seg_aff_map': pred['seg_aff_map'], 'depth_aff_map': pred['depth_aff_map'],
                #            'labels': depths, 'depths': pred_depths}
                self.disp_func(self.writer, visuals, epoch)
            print_str = 'Test step [{}/{}].'.format(step + 1, len(self.valloader))
            tbar.set_description(print_str)
        measures = {key: round(value / self.n_val, 5) for key, value in measures.items()}

        Acc = score.Pixel_Accuracy()
        # Acc_class = score.Pixel_Accuracy_Class()
        mIoU = score.Mean_Intersection_over_Union()
        FWIoU = score.Frequency_Weighted_Intersection_over_Union()
        measures['Acc'] = Acc
        measures['mIoU'] = mIoU
        measures['FWIoU'] = FWIoU
    fps = self.n_val / val_total_time
    if self.params.use_seg:
        return measures, fps
    measures = {key: round(value / self.n_val, 5) for key, value in measures.items()}
    return measures, fps

