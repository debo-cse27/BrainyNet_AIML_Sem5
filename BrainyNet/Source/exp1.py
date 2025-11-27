"""
Experiment 1: Training Time Variation
Variations: 0.25, 0.5, 1, 2, 3, 4, 5 hours
Feature: Logs u,v,w,c,p errors at regular intervals (epochs).
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import scipy.io
import time
import sys
import os
import pyvista as pv
from pyevtk.hl import gridToVTK

from utilities import neural_net, Navier_Stokes_3D, \
                    tf_session, mean_squared_error, relative_error

class HFM(object):
    def __init__(self, t_data, x_data, y_data, z_data, c_data,
                       t_eqns, x_eqns, y_eqns, z_eqns,
                       layers, batch_size,
                       Pec, Rey):
        
        # specs
        self.layers = layers
        self.batch_size = batch_size
        
        # flow properties
        self.Pec = Pec
        self.Rey = Rey
        
        # data
        [self.t_data, self.x_data, self.y_data, self.z_data, self.c_data] = [t_data, x_data, y_data, z_data, c_data]
        [self.t_eqns, self.x_eqns, self.y_eqns, self.z_eqns] = [t_eqns, x_eqns, y_eqns, z_eqns]
        
        # placeholders
        [self.t_data_tf, self.x_data_tf, self.y_data_tf, self.z_data_tf, self.c_data_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(5)]
        [self.t_eqns_tf, self.x_eqns_tf, self.y_eqns_tf, self.z_eqns_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
        
        # physics "uninformed" neural networks
        self.net_cuvwp = neural_net(self.t_data, self.x_data, self.y_data, self.z_data, layers = self.layers)
        
        [self.c_data_pred,
         self.u_data_pred,
         self.v_data_pred,
         self.w_data_pred,
         self.p_data_pred] = self.net_cuvwp(self.t_data_tf,
                                            self.x_data_tf,
                                            self.y_data_tf,
                                            self.z_data_tf)
         
        # physics "informed" neural networks
        [self.c_eqns_pred,
         self.u_eqns_pred,
         self.v_eqns_pred,
         self.w_eqns_pred,
         self.p_eqns_pred] = self.net_cuvwp(self.t_eqns_tf,
                                            self.x_eqns_tf,
                                            self.y_eqns_tf,
                                            self.z_eqns_tf)
        
        [self.e1_eqns_pred,
         self.e2_eqns_pred,
         self.e3_eqns_pred,
         self.e4_eqns_pred,
         self.e5_eqns_pred] = Navier_Stokes_3D(self.c_eqns_pred,
                                               self.u_eqns_pred,
                                               self.v_eqns_pred,
                                               self.w_eqns_pred,
                                               self.p_eqns_pred,
                                               self.t_eqns_tf,
                                               self.x_eqns_tf,
                                               self.y_eqns_tf,
                                               self.z_eqns_tf,
                                               self.Pec,
                                               self.Rey)
        
        # loss
        self.loss = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                    mean_squared_error(self.e1_eqns_pred, 0.0) + \
                    mean_squared_error(self.e2_eqns_pred, 0.0) + \
                    mean_squared_error(self.e3_eqns_pred, 0.0) + \
                    mean_squared_error(self.e4_eqns_pred, 0.0) + \
                    mean_squared_error(self.e5_eqns_pred, 0.0)
        
        # optimizers
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        
        self.sess = tf_session()
    
    # Modified train to accept test data for epoch-wise logging
    def train(self, total_time, learning_rate, t_star, x_star, y_star, z_star, c_star, u_star, v_star, w_star, p_star, log_file):
        
        N_data = self.t_data.shape[0]
        N_eqns = self.t_eqns.shape[0]
        
        start_time = time.time()
        running_time = 0
        it = 0
        
        # Create CSV for logging
        with open(log_file, "w") as f:
            f.write("Iter,Time_Hr,Loss,Error_C,Error_U,Error_V,Error_W,Error_P\n")

        print('It, Loss, Time(s), Running Time(h), Learning Rate')
        while running_time < total_time:
            
            idx_data = np.random.choice(N_data, self.batch_size)
            idx_eqns = np.random.choice(N_eqns, self.batch_size)
            
            (t_data_batch, x_data_batch, y_data_batch, z_data_batch, c_data_batch) = \
                (self.t_data[idx_data,:], self.x_data[idx_data,:], self.y_data[idx_data,:], self.z_data[idx_data,:], self.c_data[idx_data,:])

            (t_eqns_batch, x_eqns_batch, y_eqns_batch, z_eqns_batch) = \
                (self.t_eqns[idx_eqns,:], self.x_eqns[idx_eqns,:], self.y_eqns[idx_eqns,:], self.z_eqns[idx_eqns,:])

            tf_dict = {self.t_data_tf: t_data_batch,
                       self.x_data_tf: x_data_batch,
                       self.y_data_tf: y_data_batch,
                       self.z_data_tf: z_data_batch,
                       self.c_data_tf: c_data_batch,
                       self.t_eqns_tf: t_eqns_batch,
                       self.x_eqns_tf: x_eqns_batch,
                       self.y_eqns_tf: y_eqns_batch,
                       self.z_eqns_tf: z_eqns_batch,
                       self.learning_rate: learning_rate}
            
            self.sess.run([self.train_op], tf_dict)
            
            # Print & Log
            if it % 100 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed/3600.0
                [loss_value] = self.sess.run([self.loss], tf_dict)
                
                # -- Calculate Epoch Errors (using a representative snapshot 10 for speed) --
                snap = 10
                t_test = t_star[:,snap:snap+1]
                x_test = x_star[:,snap:snap+1]
                y_test = y_star[:,snap:snap+1]
                z_test = z_star[:,snap:snap+1]
                
                c_p, u_p, v_p, w_p, p_p = self.predict(t_test, x_test, y_test, z_test)
                
                # Ground truth for this snap
                c_gt = c_star[:,snap:snap+1]; u_gt = u_star[:,snap:snap+1]
                v_gt = v_star[:,snap:snap+1]; w_gt = w_star[:,snap:snap+1]
                p_gt = p_star[:,snap:snap+1]
                
                e_c = relative_error(c_p, c_gt)
                e_u = relative_error(u_p, u_gt)
                e_v = relative_error(v_p, v_gt)
                e_w = relative_error(w_p, w_gt)
                e_p = relative_error(p_p - np.mean(p_p), p_gt - np.mean(p_gt))
                
                with open(log_file, "a") as f:
                    f.write(f"{it},{running_time:.4f},{loss_value:.4e},{e_c:.4e},{e_u:.4e},{e_v:.4e},{e_w:.4e},{e_p:.4e}\n")

                print('%d,%.3e, %.2f, %.2f, %.1e' %(it, loss_value, elapsed, running_time, learning_rate))
                sys.stdout.flush()
                start_time = time.time()
            it += 1
    
    def predict(self, t_star, x_star, y_star, z_star):
        
        tf_dict = {self.t_data_tf: t_star, self.x_data_tf: x_star, self.y_data_tf: y_star, self.z_data_tf: z_star}
        
        c_star = self.sess.run(self.c_data_pred, tf_dict)
        u_star = self.sess.run(self.u_data_pred, tf_dict)
        v_star = self.sess.run(self.v_data_pred, tf_dict)
        w_star = self.sess.run(self.w_data_pred, tf_dict)
        p_star = self.sess.run(self.p_data_pred, tf_dict)
        
        return c_star, u_star, v_star, w_star, p_star

if __name__ == "__main__": 
    results_dir = "../Results/Experiment1_Time"
    if not os.path.exists(results_dir): os.makedirs(results_dir)

    time_variations = [0.25, 0.5, 1, 2, 3, 4, 5] 

    layers = [4] + 10*[5*50] + [5]
    
    # Load Data
    data = scipy.io.loadmat('../Data/sortedfifty_data.mat')
        
    t_star = data['t_star']; x_star = data['x_star']; y_star = data['y_star']; z_star = data['z_star']
    T = t_star.shape[0]; N = x_star.shape[0]
        
    U_star = data['U_star']; V_star = data['V_star']; W_star = data['W_star']
    P_star = data['P_star']; C_star = data['C_star']    
    
    # Rearrange Data 
    T_star = np.tile(t_star, (1,N)).T
    X_star = np.tile(x_star, (1,T))
    Y_star = np.tile(y_star, (1,T))
    Z_star = np.tile(z_star, (1,T))
    
    T_data = T
    N_data = N
    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_data, replace=False)
    
    t_data = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    x_data = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    y_data = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    z_data = Z_star[:, idx_t][idx_x,:].flatten()[:,None]
    c_data = C_star[:, idx_t][idx_x,:].flatten()[:,None]
    
    T_eqns = T
    N_eqns = N
    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_eqns-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_eqns, replace=False)
    t_eqns = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    x_eqns = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    y_eqns = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    z_eqns = Z_star[:, idx_t][idx_x,:].flatten()[:,None]
        
    for train_time in time_variations:
        print(f"\n================ STARTING RUN: Time = {train_time} hours ================\n")
        tf.reset_default_graph()
        
        model = HFM(t_data, x_data, y_data, z_data, c_data,
                    t_eqns, x_eqns, y_eqns, z_eqns,
                    layers, batch_size=10000,
                    Pec = 1.0/0.0101822, Rey = 1.0/0.0101822)
        
        log_name = f"{results_dir}/log_{train_time}h.csv"
        model.train(total_time=train_time, learning_rate=1e-3, 
                    t_star=T_star, x_star=X_star, y_star=Y_star, z_star=Z_star,
                    c_star=C_star, u_star=U_star, v_star=V_star, w_star=W_star, p_star=P_star,
                    log_file=log_name)

        # Test Data (Snapshot 10)
        snap = 10
        t_test = T_star[:,snap:snap+1]
        x_test = X_star[:,snap:snap+1]
        y_test = Y_star[:,snap:snap+1]
        z_test = Z_star[:,snap:snap+1]
        
        c_test = C_star[:,snap:snap+1]
        u_test = U_star[:,snap:snap+1]
        v_test = V_star[:,snap:snap+1]
        w_test = W_star[:,snap:snap+1]
        p_test = P_star[:,snap:snap+1]
        
        # Prediction
        c_pred, u_pred, v_pred, w_pred, p_pred = model.predict(t_test, x_test, y_test, z_test)
        
        # Error
        error_c = relative_error(c_pred, c_test)
        error_u = relative_error(u_pred, u_test)
        error_v = relative_error(v_pred, v_test)
        error_w = relative_error(w_pred, w_test)
        error_p = relative_error(p_pred - np.mean(p_pred), p_test - np.mean(p_test))

        print('Error c: %e' % (error_c))
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('Error w: %e' % (error_w))
        print('Error p: %e' % (error_p))
            
        ################# Save Data ###########################
        
        # Save output arrays
        C_pred_all = 0*C_star
        U_pred_all = 0*U_star
        V_pred_all = 0*V_star
        W_pred_all = 0*W_star
        P_pred_all = 0*P_star
        
        # Directory for this sub-experiment vtk
        vtk_dir = f"{results_dir}/vtk_{train_time}h"
        if not os.path.exists(vtk_dir): os.makedirs(vtk_dir)

        # Loop for saving VTK same as original file
        for snap_idx in range(0, t_star.shape[0]):
            t_test = T_star[:,snap_idx:snap_idx+1]
            x_test = X_star[:,snap_idx:snap_idx+1]
            y_test = Y_star[:,snap_idx:snap_idx+1]
            z_test = Z_star[:,snap_idx:snap_idx+1]
            
            c_test = C_star[:,snap_idx:snap_idx+1]
            u_test = U_star[:,snap_idx:snap_idx+1]
            v_test = V_star[:,snap_idx:snap_idx+1]
            w_test = W_star[:,snap_idx:snap_idx+1]
            p_test = P_star[:,snap_idx:snap_idx+1]
        
            # Prediction
            c_pred, u_pred, v_pred, w_pred, p_pred = model.predict(t_test, x_test, y_test, z_test)
            
            C_pred_all[:,snap_idx:snap_idx+1] = c_pred
            U_pred_all[:,snap_idx:snap_idx+1] = u_pred
            V_pred_all[:,snap_idx:snap_idx+1] = v_pred
            W_pred_all[:,snap_idx:snap_idx+1] = w_pred
            P_pred_all[:,snap_idx:snap_idx+1] = p_pred
        
            # Save grid
            points = np.hstack((x_test, y_test, z_test))
            grid = pv.PolyData(points)

            grid.point_data["C_test"] = c_test
            grid.point_data["U_test"] = u_test
            grid.point_data["V_test"] = v_test
            grid.point_data["W_test"] = w_test
            grid.point_data["P_test"] = p_test
            
            grid.point_data["C_pred"] = c_pred
            grid.point_data["U_pred"] = u_pred
            grid.point_data["V_pred"] = v_pred
            grid.point_data["W_pred"] = w_pred
            grid.point_data["P_pred"] = p_pred
            
            grid.point_data["Error_C"] = c_pred - c_test
            grid.point_data["Error_U"] = u_pred - u_test
            grid.point_data["Error_V"] = v_pred - v_test
            grid.point_data["Error_W"] = w_pred - w_test
            grid.point_data["Error_P"] = p_pred - p_test

            grid.save(f"{vtk_dir}/allresult_{snap_idx}.vtk")
        
        scipy.io.savemat(f"{results_dir}/results_{train_time}h.mat",
                         {'C_pred':C_pred_all, 'U_pred':U_pred_all, 'V_pred':V_pred_all, 'W_pred':W_pred_all, 'P_pred':P_pred_all})