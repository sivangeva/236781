from subprocess import check_call, check_output
import numpy as np
import torch
from attacks.attack import Attack
import time
from tqdm import tqdm
import cv2


class APGD(Attack):
    def __init__(
            self,
            model,
            criterion,
            test_criterion,
            data_shape,
            norm='Linf',
            n_iter=20,
            rho=.75,
            n_restarts=1,
            alpha=None,
            rand_init=False,
            sample_window_size=None,
            sample_window_stride=None,
            pert_padding=(0, 0),
            init_pert_path=None,
            init_pert_transform=None):
        super(APGD, self).__init__(model, criterion, test_criterion, norm, data_shape,
                                   sample_window_size, sample_window_stride,
                                   pert_padding)

        self.alpha = alpha
        self.thr_decr = rho
        self.n_restarts = n_restarts
        self.n_iter = n_iter

        self.rand_init = rand_init

        self.init_pert = None
        if init_pert_path is not None:
            self.init_pert = cv2.cvtColor(cv2.imread(init_pert_path), cv2.COLOR_BGR2RGB)
            if init_pert_transform is None:
                self.init_pert = torch.tensor(self.init_pert).unsqueeze(0)
            else:
                self.init_pert = init_pert_transform({'img': self.init_pert})['img'].unsqueeze(0)
        
        # add
        self.alpha_last_changed = False
        self.check_point_step = 10
        self.last_checkpoint_pert = None
        self.last_checkpoint_pert_diff_list = None



    def calc_sample_grad_single(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                            scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2, device=None):
        pert = pert.detach()
        pert.requires_grad_()
        img1_adv, img2_adv, output_adv = self.perturb_model_single(pert, img1_I0, img2_I0,
                                                                   intrinsic_I0,
                                                                   img1_delta, img2_delta,
                                                                   scale,
                                                                   mask1, mask2,
                                                                   perspective1,
                                                                   perspective2,
                                                                   device)


        loss = self.criterion(output_adv, scale.to(device), y.to(device), target_pose.to(device), clean_flow.to(device))
        #print('====> SIVAN, you are inside calc_sample_grad_single, this is the lost!!!!')
        #print(loss)
        ## TODO: Softmax instead of sum:
        ## [100, 200, 100, 400]
        loss_sum = loss.sum(dim=0)
        grad = torch.autograd.grad(loss_sum, [pert])[0].detach()



        del img1_adv
        del img2_adv
        del output_adv
        #del loss
        #del loss_sum
        torch.cuda.empty_cache()
        #print('====> SIVAN, you are inside calc_sample_grad_single')
        #print('the loss is = ')
        #print(loss_sum)
        return grad, loss_sum


    def calc_sample_grad_split(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                               scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2, device=None):
        sample_data_ind = list(range(img1_I0.shape[0] + 1))
        window_start_list = sample_data_ind[0::self.sample_window_stride]
        window_end_list = sample_data_ind[self.sample_window_size::self.sample_window_stride]

        if window_end_list[-1] != sample_data_ind[-1]:
            window_end_list.append(sample_data_ind[-1])
        grad = torch.zeros_like(pert, requires_grad=False)
        grad_multiplicity = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)

        for window_idx, window_end in enumerate(window_end_list):
            window_start = window_start_list[window_idx]
            grad_multiplicity[window_start:window_end] += 1

            pert_window = pert[window_start:window_end].clone().detach()
            img1_I0_window = img1_I0[window_start:window_end].clone().detach()
            img2_I0_window = img2_I0[window_start:window_end].clone().detach()
            intrinsic_I0_window = intrinsic_I0[window_start:window_end].clone().detach()
            img1_delta_window = img1_delta[window_start:window_end].clone().detach()
            img2_delta_window = img2_delta[window_start:window_end].clone().detach()
            scale_window = scale[window_start:window_end].clone().detach()
            y_window = y[window_start:window_end].clone().detach()
            clean_flow_window = clean_flow[window_start:window_end].clone().detach()
            target_pose_window = target_pose.clone().detach()
            perspective1_window = perspective1[window_start:window_end].clone().detach()
            perspective2_window = perspective2[window_start:window_end].clone().detach()
            mask1_window = mask1[window_start:window_end].clone().detach()
            mask2_window = mask2[window_start:window_end].clone().detach()

            grad_window, loss_window = self.calc_sample_grad_single(pert_window,
                                                                    img1_I0_window,
                                                                    img2_I0_window,
                                                                    intrinsic_I0_window,
                                                                    img1_delta_window,
                                                                    img2_delta_window,
                                                                    scale_window,
                                                                    y_window,
                                                                    clean_flow_window,
                                                                    target_pose_window,
                                                                    perspective1_window,
                                                                    perspective2_window,
                                                                    mask1_window,
                                                                    mask2_window,
                                                                    device=device)
            with torch.no_grad():
                grad[window_start:window_end] += grad_window

            del grad_window
            del pert_window
            del img1_I0_window
            del img2_I0_window
            del intrinsic_I0_window
            del scale_window
            del y_window
            del clean_flow_window
            del target_pose_window
            del perspective1_window
            del perspective2_window
            del mask1_window
            del mask2_window
            torch.cuda.empty_cache()
        grad_multiplicity_expand = grad_multiplicity.view(-1, 1, 1, 1).expand(grad.shape)
        grad = grad / grad_multiplicity_expand
        del grad_multiplicity
        del grad_multiplicity_expand
        torch.cuda.empty_cache()
        return grad.to(device), loss_window



    def perturb(self, data_loader, y_list, eps,
                                   targeted=False, device=None, eval_data_loader=None, eval_y_list=None):

        a_abs = np.abs(eps / self.n_iter) if self.alpha is None else np.abs(self.alpha)
        multiplier = -1 if targeted else 1
        print("computing APGD attack with parameters:")
        print("attack random restarts: " + str(self.n_restarts))
        print("attack epochs: " + str(self.n_iter))
        print("attack norm: " + str(self.norm))
        print("attack epsilon norm limitation: " + str(eps))
        print("attack step size: " + str(a_abs))

        data_shape, dtype, eval_data_loader, eval_y_list, clean_flow_list, \
        eval_clean_loss_list, traj_clean_loss_mean_list, clean_loss_sum, \
        best_pert, best_loss_list, best_loss_sum, all_loss, all_best_loss = \
            self.compute_clean_baseline(data_loader, y_list, eval_data_loader, eval_y_list, device=device)
        

        for rest in tqdm(range(self.n_restarts)):
            print("restarting attack optimization, restart number: " + str(rest))
            opt_start_time = time.time()

            pert = torch.zeros_like(best_pert)

            if self.init_pert is not None:
                print(" perturbation initialized from provided image")
                pert = self.init_pert.to(best_pert)
            elif self.rand_init:
                print(" perturbation initialized randomly")
                pert = self.random_initialization(pert, eps)
            else:
                print(" perturbation initialized to zero")

            pert = self.project(pert, eps)
            pert_list = [torch.zeros_like(best_pert), torch.zeros_like(best_pert)]
            
            # initialization before entering the iteration loops
            a_abs_apgd = a_abs
            prev_iter_loss = 0
            counter_loss_increase = 0
            check_point_prev_loss = float('-inf')
            for k in tqdm(range(self.n_iter)):
                print('===> SIVAN, you are inside apgd attack opt')
                print(" attack optimization epoch: " + str(k))
                iter_start_time = time.time()
                pert_regulation = 0.7 # set regulization 
                    
                # get grad and loss
                grad_tot, loss_total = self.gradient_ascent_step_apgd(pert, data_shape, data_loader, 
                                            y_list, clean_flow_list,device=device) 
                print('===> SIVAN, this is loss_total:')
                print(loss_total)
                if loss_total>prev_iter_loss:
                    counter_loss_increase +=1
                prev_iter_loss = loss_total

                epoch_start_checkpoint = 70
                #check and update alpha checkpoint
                if k % self.check_point_step == 0 and k > epoch_start_checkpoint - 1:
                    
                    a_abs_apgd, alpha_changed = self.update_alpha(k, loss_total, counter_loss_increase, 
                                        check_point_prev_loss, a_abs_apgd)
                    print('===> SIVAN, alpha is updated!!: ')
                    print('===>this is alpha = ')
                    print(a_abs_apgd)
                    check_point_prev_loss = loss_total
                    counter_loss_increase = 0
                    #self.last_checkpoint_pert = None
                    #self.last_checkpoint_pert_diff_list = None
                    # creat final pert
                    if k == 0:
                        pert_diff = pert - torch.zeros_like(best_pert) 
                    else:
                        pert_diff = pert_list[1] - pert_list[0]
                    # go back to last pert if needed by conditions of alpha update
                    if alpha_changed and epoch_start_checkpoint > 70:
                        pert = self.last_checkpoint_pert
                        pert_list = self.last_checkpoint_pert_diff_list
                    else:
                        pert = self.apgd_pert_update(pert, multiplier, grad_tot, a_abs_apgd, eps, pert_diff, pert_regulation)

                    # update list 
                    pert_list[0] = pert_list[1]
                    pert_list[1] = pert
                    # update checkpoint x to go back if needed
                    self.last_checkpoint_pert = pert
                    self.last_checkpoint_pert_diff_list = pert_diff

                else:
                    # creat final pert
                    if k == 0:
                        pert_diff = pert - torch.zeros_like(best_pert) 
                    else:
                        pert_diff = pert_list[1] - pert_list[0]
                    pert = self.apgd_pert_update(pert, multiplier, grad_tot, a_abs_apgd, eps, pert_diff, pert_regulation)
                    # update list 
                    pert_list[0] = pert_list[1]
                    pert_list[1] = pert



                step_runtime = time.time() - iter_start_time
                print(" optimization epoch finished, epoch runtime: " + str(step_runtime))

                print(" evaluating perturbation")
                eval_start_time = time.time()

                with torch.no_grad():
                    eval_loss_tot, eval_loss_list = self.attack_eval(pert, data_shape, eval_data_loader, eval_y_list,
                                                                     device)

                    if eval_loss_tot > best_loss_sum:
                        best_pert = pert.clone().detach()
                        best_loss_list = eval_loss_list
                        best_loss_sum = eval_loss_tot
                    all_loss.append(eval_loss_list)
                    all_best_loss.append(best_loss_list)
                    traj_loss_mean_list = np.mean(eval_loss_list, axis=0)
                    traj_best_loss_mean_list = np.mean(best_loss_list, axis=0)

                    eval_runtime = time.time() - eval_start_time
                    print(" evaluation finished, evaluation runtime: " + str(eval_runtime))
                    print(" current trajectories loss mean list:")
                    print(" " + str(traj_loss_mean_list))
                    print(" current trajectories best loss mean list:")
                    print(" " + str(traj_best_loss_mean_list))
                    print(" trajectories clean loss mean list:")
                    print(" " + str(traj_clean_loss_mean_list))
                    print(" current trajectories loss sum:")
                    print(" " + str(eval_loss_tot))
                    print(" current trajectories best loss sum:")
                    print(" " + str(best_loss_sum))
                    print(" trajectories clean loss sum:")
                    print(" " + str(clean_loss_sum))
                    del eval_loss_tot
                    del eval_loss_list
                    torch.cuda.empty_cache()

            opt_runtime = time.time() - opt_start_time
            print("optimization restart finished, optimization runtime: " + str(opt_runtime))
        return best_pert.detach(), eval_clean_loss_list, all_loss, all_best_loss

    def update_alpha(self, k, loss_total, counter_loss_increase, check_point_prev_loss, a_abs_apgd):
        # eta_0 = epsilon*2 
        # checkpoints %%20 == 0 
        print('===> SIVAN, you are inside update_alpha')
        rho = 0.75
        alpha_last_changed = self.alpha_last_changed
        check_point_step = self.check_point_step

        # condition 1
        if counter_loss_increase/ check_point_step < rho:
            condition_1 = True
        else:
            condition_1 = False
        print('===> SIVAN, condition 1 counter_loss_increase')
        print(counter_loss_increase)
        print('condition1 is ' + str(condition_1))

        #condition 2
        if k > check_point_step:
            if (alpha_last_changed == False) and (check_point_prev_loss >= loss_total):
                condition_2 = True
            else:
                condition_2 = False
        else:
            condition_2 = False

        print('===> SIVAN, condition 2 heck_point_prev_loss = ')
        print(check_point_prev_loss)
        print('condition2 is ' + str(condition_2))   
        alpha_changed = condition_1 or condition_2
        if alpha_changed:
            self.alpha_last_changed = True
            return a_abs_apgd/2, alpha_changed
        else:
            self.alpha_last_changed = False
            return a_abs_apgd, alpha_changed


    # TODO : we must check this project function as it takes the eps!!!! 
    # could be that we need to use other function for the pert_final
    def apgd_pert_update(self, pert, multiplier, grad_tot, a_abs_apgd, eps, pert_diff, pert_regulation):
            print('===> SIVAN, you are inside apgd_pert_update')
            with torch.no_grad():
                # the sign is because we are using Linf.
                # z notation in the paper
                pert_temp = pert + multiplier * a_abs_apgd * torch.sign(grad_tot)
                pert_temp = self.project(pert_temp, eps)

                # x notation in the paper  
                pert_final = self.project(pert + (pert_regulation*(pert_temp - pert)) + ((1-pert_regulation)*(pert_diff)), eps)

                # lets check if we use just Yaniv:
                #pert_final = self.project((pert_regulation*(pert_temp)) + ((1-pert_regulation)*(pert_diff)), eps)
            return pert_final