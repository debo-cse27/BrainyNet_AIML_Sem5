"""
- Experiment 3: Learning Rate Dependency (Static & Dynamic)

Experiment 3: Learning Rate Dependency
    Goal: Find the optimal learning rate.
    Method: Use a fixed total_time (e.g., 1.0 hour).
    Static LR Test: Test learning_rate = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4].
    Dynamic LR Test (Recommended): Implement a learning rate schedule. This is a major improvement. 
    For example, use tf.train.exponential_decay to start at 1e-3 and decay to 1e-5 over the training run.
    Deliverable: Plots of final error vs. static learning_rate. A separate plot showing the error convergence of the best static LR vs. the new dynamic LR schedule.
"""
"""
RUN SCRIPT FOR MEMBER B (Training Tuner)
- Experiment 3: Learning Rate Dependency (Static & Dynamic)
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import scipy.io
import time
import sys
import os
import pathlib
import pyvista as pv # Import pyvista

# Assume utilities.py is in the same directory
from utilities import neural_net, Navier_Stokes_3D, \
                    tf_session, mean_squared_error, relative_error

class HFM(object):
    # --- (HFM Class from previous script) ---
    def __init__(self, t_data, x_data, y_data, z_data, c_data,
                       u_data, v_data, w_data, p_data,
                       t_eqns, x_eqns, y_eqns, z_eqns,
                       layers, batch_size,
                       Pec, Rey): 
        
        self.layers = layers
        self.batch_size = batch_size
        self.Pec = Pec
        self.Rey = Rey
        
        [self.t_data, self.x_data, self.y_data, self.z_data, self.c_data,
         self.u_data, self.v_data, self.w_data, self.p_data] = [
                t_data, x_data, y_data, z_data, c_data,
                u_data, v_data, w_data, p_data]
        [self.t_eqns, self.x_eqns, self.y_eqns, self.z_eqns] = [t_eqns, x_eqns, y_eqns, z_eqns]
        
        [self.t_data_tf, self.x_data_tf, self.y_data_tf, self.z_data_tf, self.c_data_tf,
         self.u_data_tf, self.v_data_tf, self.w_data_tf, self.p_data_tf] = [
                tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(9)]
        [self.t_eqns_tf, self.x_eqns_tf, self.y_eqns_tf, self.z_eqns_tf] = [
                tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
        
        self.net_cuvwp = neural_net(self.t_data, self.x_data, self.y_data, self.z_data,
                                    layers=self.layers)
        
        [self.c_data_pred,
         self.u_data_pred,
         self.v_data_pred,
         self.w_data_pred,
         self.p_data_pred] = self.net_cuvwp(self.t_data_tf,
                                            self.x_data_tf,
                                            self.y_data_tf,
                                            self.z_data_tf)
         
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
        
        self.loss_c_data = mean_squared_error(self.c_data_pred, self.c_data_tf)
        self.loss_u_data = mean_squared_error(self.u_data_pred, self.u_data_tf)
        self.loss_v_data = mean_squared_error(self.v_data_pred, self.v_data_tf)
        self.loss_w_data = mean_squared_error(self.w_data_pred, self.w_data_tf)
        self.loss_p_data = mean_squared_error(self.p_data_pred, self.p_data_tf)
        
        self.loss_e1 = mean_squared_error(self.e1_eqns_pred, 0.0)
        self.loss_e2 = mean_squared_error(self.e2_eqns_pred, 0.0)
        self.loss_e3 = mean_squared_error(self.e3_eqns_pred, 0.0)
        self.loss_e4 = mean_squared_error(self.e4_eqns_pred, 0.0)
        self.loss_e5 = mean_squared_error(self.e5_eqns_pred, 0.0)
        
        self.loss_data = (0.01 * self.loss_c_data + 
                          1.0 * self.loss_u_data + 
                          1.0 * self.loss_v_data + 
                          1.0 * self.loss_w_data +
                          1.0 * self.loss_p_data)
                           
        self.loss_physics = self.loss_e1 + self.loss_e2 + self.loss_e3 + self.loss_e4 + self.loss_e5

        self.lambda_physics_tf = tf.placeholder(tf.float32, shape=[]) 

        self.loss = (1.0 - self.lambda_physics_tf) * self.loss_data + \
                    self.lambda_physics_tf * self.loss_physics
        
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        
        self.sess = tf_session()
    
    def train(self, total_time, learning_rate, lambda_physics=0.5, lr_schedule=None, n_iter_print=10):
        
        N_data = self.t_data.shape[0]
        N_eqns = self.t_eqns.shape[0]
        
        start_time = time.time()
        running_time = 0
        it = 0
        print('It, Loss, Time(s), Running Time(h), Learning Rate')
        while running_time < total_time:
            
            idx_data = np.random.choice(N_data, self.batch_size)
            idx_eqns = np.random.choice(N_eqns, self.batch_size)
            
            (t_data_batch,
             x_data_batch,
             y_data_batch,
             z_data_batch,
             c_data_batch,
             u_data_batch,
             v_data_batch,
             w_data_batch,
             p_data_batch) = (self.t_data[idx_data,:],
                              self.x_data[idx_data,:],
                              self.y_data[idx_data,:],
                              self.z_data[idx_data,:],
                              self.c_data[idx_data,:],
                              self.u_data[idx_data,:],
                              self.v_data[idx_data,:],
                              self.w_data[idx_data,:],
                              self.p_data[idx_data,:])

            (t_eqns_batch,
             x_eqns_batch,
             y_eqns_batch,
             z_eqns_batch) = (self.t_eqns[idx_eqns,:],
                              self.x_eqns[idx_eqns,:],
                              self.y_eqns[idx_eqns,:],
                              self.z_eqns[idx_eqns,:])
            
            current_lr = learning_rate
            if lr_schedule == 'exponential_decay':
                current_lr = learning_rate * (0.01 ** (running_time / total_time))
            
            tf_dict = {self.t_data_tf: t_data_batch,
                       self.x_data_tf: x_data_batch,
                       self.y_data_tf: y_data_batch,
                       self.z_data_tf: z_data_batch,
                       self.c_data_tf: c_data_batch,
                       self.u_data_tf: u_data_batch,
                       self.v_data_tf: v_data_batch,
                       self.w_data_tf: w_data_batch,
                       self.p_data_tf: p_data_batch,
                       self.t_eqns_tf: t_eqns_batch,
                       self.x_eqns_tf: x_eqns_batch,
                       self.y_eqns_tf: y_eqns_batch,
                       self.z_eqns_tf: z_eqns_batch,
                       self.learning_rate: current_lr,
                       self.lambda_physics_tf: lambda_physics}
            
            self.sess.run([self.train_op], tf_dict)
            
            if it % n_iter_print == 0:
                elapsed = time.time() - start_time
                running_time += elapsed/3600.0
                [loss_value,
                 learning_rate_value] = self.sess.run([self.loss,
                                                       self.learning_rate], tf_dict)
                print('%d,%.3e, %.2f, %.2f, %.1e'
                      %(it, loss_value, elapsed, running_time, learning_rate_value))
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

# --- Function to run a single experiment ---
def run_experiment(config):
    
    tf.reset_default_graph() 
    
    print(f"\n--- Starting Experiment: {config['name']} ---")
    
    results_dir = config['results_dir']
    vtk_dir = os.path.join(results_dir, "VTK_files")
    pathlib.Path(vtk_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {results_dir}")
    
    data = scipy.io.loadmat(config['data_path'])
    
    t_star = data['t_star'] # T x 1
    x_star = data['x_star'] # N x 1
    y_star = data['y_star'] # N x 1
    z_star = data['z_star'] # N x 1

    T = t_star.shape[0]
    N = x_star.shape[0]
        
    U_star = data['U_star'] # N x T
    V_star = data['V_star'] # N x T
    W_star = data['W_star'] # N x T
    P_star = data['P_star'] # N x T
    C_star = data['C_star'] # N x T    
    
    T_star = np.tile(t_star, (1,N)).T # N x T
    X_star = np.tile(x_star, (1,T)) # N x T
    Y_star = np.tile(y_star, (1,T)) # N x T
    Z_star = np.tile(z_star, (1,T)) # N x T
    
    T_data = T
    N_data = config.get('N_data', N) 
    N_eqns = config.get('N_eqns', N) 
    
    if N_data > N: N_data = N
    if N_eqns > N: N_eqns = N

    idx_t_data = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] )
    idx_x_data = np.random.choice(N, N_data, replace=False)
    t_data = T_star[:, idx_t_data][idx_x_data,:].flatten()[:,None]
    x_data = X_star[:, idx_t_data][idx_x_data,:].flatten()[:,None]
    y_data = Y_star[:, idx_t_data][idx_x_data,:].flatten()[:,None]
    z_data = Z_star[:, idx_t_data][idx_x_data,:].flatten()[:,None]
    c_data = C_star[:, idx_t_data][idx_x_data,:].flatten()[:,None]
    u_data = U_star[:, idx_t_data][idx_x_data,:].flatten()[:,None]
    v_data = V_star[:, idx_t_data][idx_x_data,:].flatten()[:,None]
    w_data = W_star[:, idx_t_data][idx_x_data,:].flatten()[:,None]
    p_data = P_star[:, idx_t_data][idx_x_data,:].flatten()[:,None]
    
    T_eqns = T
    idx_t_eqns = np.concatenate([np.array([0]), np.random.choice(T-2, T_eqns-2, replace=False)+1, np.array([T-1])] )
    idx_x_eqns = np.random.choice(N, N_eqns, replace=False)
    t_eqns = T_star[:, idx_t_eqns][idx_x_eqns,:].flatten()[:,None]
    x_eqns = X_star[:, idx_t_eqns][idx_x_eqns,:].flatten()[:,None]
    y_eqns = Y_star[:, idx_t_eqns][idx_x_eqns,:].flatten()[:,None]
    z_eqns = Z_star[:, idx_t_eqns][idx_x_eqns,:].flatten()[:,None]
        
    # Initialize Model
    model = HFM(t_data, x_data, y_data, z_data, c_data,
                u_data, v_data, w_data, p_data,
                t_eqns, x_eqns, y_eqns, z_eqns,
                config['layers'], config['batch_size'],
                Pec = 1.0/0.0101822, Rey = 1.0/0.0101822)
    
    # Train Model
    model.train(total_time = config['total_time'], 
                learning_rate = config['learning_rate'], 
                lambda_physics = config['lambda_physics'],
                lr_schedule = config.get('lr_schedule', None),
                n_iter_print = 500) 
    
    # Test Data (Snapshot 10)
    snap = np.array([10])
    t_test = T_star[:,snap]
    x_test = X_star[:,snap]
    y_test = Y_star[:,snap]
    z_test = Z_star[:,snap]
    c_test = C_star[:,snap]
    u_test = U_star[:,snap]
    v_test = V_star[:,snap]
    w_test = W_star[:,snap]
    p_test = P_star[:,snap]
    
    # Prediction
    c_pred, u_pred, v_pred, w_pred, p_pred = model.predict(t_test, x_test, y_test, z_test)
    
    # Error
    error_c = relative_error(c_pred, c_test)
    error_u = relative_error(u_pred, u_test)
    error_v = relative_error(v_pred, v_test)
    error_w = relative_error(w_pred, w_test)
    error_p = relative_error(p_pred - np.mean(p_pred), p_test - np.mean(p_test))

    print('Final Error c: %e' % (error_c))
    print('Final Error u: %e' % (error_u))
    print('Final Error v: %e' % (error_v))
    print('Final Error w: %e' % (error_w))
    print('Final Error p: %e' % (error_p))
        
    # Save VTK files
    points = np.hstack((x_star, y_star, z_star))
    grid = pv.PolyData(points)
    
    for snap in range(0,t_star.shape[0]):
        t_test = T_star[:,snap:snap+1]
        x_test = X_star[:,snap:snap+1]
        y_test = Y_star[:,snap:snap+1]
        z_test = Z_star[:,snap:snap+1]
        
        c_test = C_star[:,snap:snap+1]
        u_test = U_star[:,snap:snap+1]
        v_test = V_star[:,snap:snap+1]
        w_test = W_star[:,snap:snap+1]
        p_test = P_star[:,snap:snap+1]
    
        c_pred, u_pred, v_pred, w_pred, p_pred = model.predict(t_test, x_test, y_test, z_test)
        
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
        grid.point_data["Error_U"] = u_pred - u_test
        grid.point_data["Error_P"] = p_pred - p_test

        grid.save(os.path.join(vtk_dir, f"allresult_{snap}.vtp"))
    
    print(f"VTK files saved to {vtk_dir}")
    
    # Save .mat file
    mat_path = os.path.join(results_dir, "predictions.mat")
    scipy.io.savemat(mat_path,
                     {'c_pred':c_pred, 'u_pred':u_pred, 'v_pred':v_pred, 'w_pred':w_pred, 'p_pred':p_pred,
                      'error_c': error_c, 'error_u': error_u, 'error_v': error_v, 'error_w': error_w, 'error_p': error_p})
    print(f".mat file saved to {mat_path}")
    
    return (error_c, error_u, error_v, error_w, error_p)

# --- Main execution block for Member B ---
if __name__ == "__main__": 
    
    # Baseline settings for Member B's experiments
    BASE_CONFIG = {
        'layers': [4] + 10*[5*50] + [5], # Baseline architecture
        'lambda_physics': 0.5, 
        'batch_size': 10000,
        'data_path': '../Data/sortedfifty_data.mat'
    }
    
    # Log file
    log_file_path = "../Results/Member_B_LR_results.txt"
    pathlib.Path("../Results/").mkdir(exist_ok=True)
    with open(log_file_path, 'w') as f:
        f.write("MEMBER B EXPERIMENT 3 (LR) LOG\n")
        f.write("="*30 + "\n")
        f.write("Experiment, Error_c, Error_u, Error_v, Error_w, Error_p\n")

    # --- Experiment 3: Static LR Test ---
    print("\n" + "="*50)
    print("MEMBER B: STARTING EXPERIMENT 3 (STATIC LR TEST)")
    print("="*50)
    
    lr_tests = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]

    for lr_val in lr_tests:
        config = BASE_CONFIG.copy()
        config['name'] = f"Exp3_StaticLR_{lr_val:.0e}"
        config['total_time'] = 1.0 # Fixed time for this test
        config['learning_rate'] = lr_val
        config['results_dir'] = f"../Results/Member_B/LR/{config['name']}" # Subfolder
        
        errors = run_experiment(config)
        
        with open(log_file_path, 'a') as f:
            f.write(f"{config['name']}, {errors[0]}, {errors[1]}, {errors[2]}, {errors[3]}, {errors[4]}\n")

    # --- Experiment 3: Dynamic LR Test ---
    print("\n" + "="*50)
    print("MEMBER B: STARTING EXPERIMENT 3 (DYNAMIC LR TEST)")
    print("="*50)
    
    config = BASE_CONFIG.copy()
    config['name'] = "Exp3_DynamicLR_1e-3_to_1e-5"
    config['total_time'] = 1.0 # Fixed time
    config['learning_rate'] = 1e-3 # Starting LR
    config['lr_schedule'] = 'exponential_decay'
    config['results_dir'] = f"../Results/Member_B/LR/{config['name']}" # Subfolder
    
    errors = run_experiment(config)
    
    with open(log_file_path, 'a') as f:
        f.write(f"{config['name']}, {errors[0]}, {errors[1]}, {errors[2]}, {errors[3]}, {errors[4]}\n")

    print("\n" + "="*50)
    print("MEMBER B: LEARNING RATE EXPERIMENTS COMPLETE.")
    print(f"Results summary saved to {log_file_path}")
    print("="*50)