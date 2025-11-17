"""
Optimizer: Test Adam (current) vs. L-BFGS. (Note: L-BFGS is a second-order optimizer often used after Adam to fine-tune the solution. 
You may need to modify the train loop).
"""

"""
RUN SCRIPT FOR MEMBER A (Model Architect)
- Experiment 6: Optimizer Test (Adam vs. Adam + L-BFGS)

NOTE: Uses a custom L-BFGS optimizer wrapper that works with TensorFlow 2.x
(compatible replacement for tensorflow.contrib.opt.ScipyOptimizerInterface)
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import scipy.io
import scipy.optimize
import time
import sys
import os
import pathlib
import pyvista as pv

from utilities import neural_net, Navier_Stokes_3D, \
                    tf_session, mean_squared_error, relative_error

# Custom L-BFGS optimizer wrapper (replacement for tensorflow.contrib.opt.ScipyOptimizerInterface)
class ScipyOptimizerInterface:
    """
    Wrapper for scipy.optimize.minimize to work with TensorFlow variables.
    This replaces tensorflow.contrib.opt.ScipyOptimizerInterface which was removed in TF 2.0.
    """
    def __init__(self, loss, method='L-BFGS-B', options=None):
        self.loss = loss
        self.method = method
        self.options = options or {}
        self.callback_fn = None
        
    def minimize(self, sess, feed_dict=None, fetches=None, loss_callback=None):
        """
        Minimize the loss using scipy.optimize.minimize.
        
        Args:
            sess: TensorFlow session
            feed_dict: Dictionary of feed values for placeholders
            fetches: Optional list of tensors to fetch (not used, kept for compatibility)
            loss_callback: Optional callback function called with loss value at each iteration
        """
        self.sess = sess
        self.feed_dict = feed_dict or {}
        self.loss_callback = loss_callback
        
        # Get all trainable variables
        self.vars = tf.trainable_variables()
        self.var_shapes = [var.get_shape().as_list() for var in self.vars]
        self.var_sizes = [np.prod(shape) for shape in self.var_shapes]
        self.var_starts = np.cumsum([0] + self.var_sizes[:-1])
        
        # Initial parameter values
        x0 = self._pack_vars()
        
        # Define objective function
        def objective(x):
            self._unpack_vars(x)
            loss_val = sess.run(self.loss, feed_dict=self.feed_dict)
            if self.loss_callback:
                self.loss_callback(loss_val)
            return loss_val
        
        # Define gradient function
        def gradient(x):
            self._unpack_vars(x)
            grads = sess.run(tf.gradients(self.loss, self.vars), feed_dict=self.feed_dict)
            return self._pack_grads(grads)
        
        # Run optimization
        result = scipy.optimize.minimize(
            objective,
            x0,
            method=self.method,
            jac=gradient,
            options=self.options,
            callback=None
        )
        
        # Restore final values
        self._unpack_vars(result.x)
        
        return result
    
    def _pack_vars(self):
        """Pack all variables into a flat numpy array."""
        values = self.sess.run(self.vars)
        return np.concatenate([v.flatten() for v in values])
    
    def _unpack_vars(self, x):
        """Unpack a flat numpy array into variables."""
        for var, start, size, shape in zip(self.vars, self.var_starts, self.var_sizes, self.var_shapes):
            var_val = x[start:start+size].reshape(shape)
            self.sess.run(var.assign(var_val))
    
    def _pack_grads(self, grads):
        """Pack gradients into a flat numpy array."""
        return np.concatenate([g.flatten() for g in grads])

# Alias for compatibility
lbfgs_optimizer = ScipyOptimizerInterface

class HFM(object):
    def __init__(self, t_data, x_data, y_data, z_data, c_data,
                       u_data, v_data, w_data, p_data,
                       t_eqns, x_eqns, y_eqns, z_eqns,
                       layers, batch_size,
                       Pec, Rey,
                       activation=tf.nn.tanh):
        
        self.layers = layers
        self.batch_size = batch_size
        self.Pec = Pec
        self.Rey = Rey
        
        # Store all data for L-BFGS
        self.all_t_data, self.all_x_data, self.all_y_data, self.all_z_data = t_data, x_data, y_data, z_data
        self.all_c_data, self.all_u_data, self.all_v_data, self.all_w_data, self.all_p_data = c_data, u_data, v_data, w_data, p_data
        self.all_t_eqns, self.all_x_eqns, self.all_y_eqns, self.all_z_eqns = t_eqns, x_eqns, y_eqns, z_eqns

        # Data for batching (Adam)
        self.t_data, self.x_data, self.y_data, self.z_data = t_data, x_data, y_data, z_data
        self.c_data, self.u_data, self.v_data, self.w_data, self.p_data = c_data, u_data, v_data, w_data, p_data
        self.t_eqns, self.x_eqns, self.y_eqns, self.z_eqns = t_eqns, x_eqns, y_eqns, z_eqns

        self.N_data = t_data.shape[0]
        self.N_eqns = t_eqns.shape[0]

        [self.t_data_tf, self.x_data_tf, self.y_data_tf, self.z_data_tf, self.c_data_tf,
         self.u_data_tf, self.v_data_tf, self.w_data_tf, self.p_data_tf] = [
                tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(9)]
        [self.t_eqns_tf, self.x_eqns_tf, self.y_eqns_tf, self.z_eqns_tf] = [
                tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
        
        self.net_cuvwp = neural_net(self.t_data, self.x_data, self.y_data, self.z_data,
                                    layers=self.layers, activation=activation)
        
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
        
        # --- MODIFIED: Define both optimizers ---
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer_adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op_adam = self.optimizer_adam.minimize(self.loss)
        
        self.optimizer_lbfgs = lbfgs_optimizer(
                self.loss, 
                method='L-BFGS-B', 
                options={'maxiter': 50000, 
                         'maxfun': 50000,
                         'maxcor': 50,
                         'maxls': 50,
                         'ftol': 1.0 * np.finfo(float).eps})
        
        self.sess = tf_session()
        self.lbfgs_step = 0 # Counter for L-BFGS steps
    
    def train_adam(self, total_time, learning_rate, lambda_physics=0.5, lr_schedule=None, n_iter_print=10):
        N_data = self.N_data
        N_eqns = self.N_eqns
        
        start_time = time.time()
        running_time = 0
        it = 0
        print('ADAM: It, Loss, Time(s), Running Time(h), Learning Rate')
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
            
            self.sess.run([self.train_op_adam], tf_dict)
            
            if it % n_iter_print == 0:
                elapsed = time.time() - start_time
                running_time += elapsed/3600.0
                [loss_value,
                 learning_rate_value] = self.sess.run([self.loss,
                                                       self.learning_rate], tf_dict)
                print('ADAM: %d,%.3e, %.2f, %.2f, %.1e'
                      %(it, loss_value, elapsed, running_time, learning_rate_value))
                sys.stdout.flush()
                start_time = time.time()
            it += 1
    
    # --- NEW: L-BFGS Training Method ---
    def train_lbfgs(self, lambda_physics=0.5):
        print("Starting L-BFGS training...")
        
        # L-BFGS must run on all data, not batches
        tf_dict = {self.t_data_tf: self.all_t_data,
                   self.x_data_tf: self.all_x_data,
                   self.y_data_tf: self.all_y_data,
                   self.z_data_tf: self.all_z_data,
                   self.c_data_tf: self.all_c_data,
                   self.u_data_tf: self.all_u_data,
                   self.v_data_tf: self.all_v_data,
                   self.w_data_tf: self.all_w_data,
                   self.p_data_tf: self.all_p_data,
                   self.t_eqns_tf: self.all_t_eqns,
                   self.x_eqns_tf: self.all_x_eqns,
                   self.y_eqns_tf: self.all_y_eqns,
                   self.z_eqns_tf: self.all_z_eqns,
                   self.lambda_physics_tf: lambda_physics}
        
        self.optimizer_lbfgs.minimize(self.sess,
                                      feed_dict=tf_dict,
                                      fetches=[self.loss],
                                      loss_callback=self.lbfgs_callback)
        print("L-BFGS training finished.")

    # --- NEW: L-BFGS Callback Function ---
    def lbfgs_callback(self, loss):
        self.lbfgs_step += 1
        if self.lbfgs_step % 10 == 0:
            print('L-BFGS: %d, %.3e' % (self.lbfgs_step, loss))
    
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
    
    print(f"\n--- Starting Experiment: {config['name']} ---")
    
    results_dir = config['results_dir']
    vtk_dir = os.path.join(results_dir, "VTK_files")
    pathlib.Path(vtk_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {results_dir}")
    
    data = scipy.io.loadmat(config['data_path'])
    
    t_star = data['t_star']
    x_star, y_star, z_star = data['x_star'], data['y_star'], data['z_star']
    T, N = t_star.shape[0], x_star.shape[0]
    U_star, V_star, W_star = data['U_star'], data['V_star'], data['W_star']
    P_star, C_star = data['P_star'], data['C_star']
    
    T_star = np.tile(t_star, (1,N)).T
    X_star, Y_star, Z_star = np.tile(x_star, (1,T)), np.tile(y_star, (1,T)), np.tile(z_star, (1,T))
    
    T_data, N_data = T, N
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
    
    T_eqns, N_eqns = T, N
    idx_t_eqns = np.concatenate([np.array([0]), np.random.choice(T-2, T_eqns-2, replace=False)+1, np.array([T-1])] )
    idx_x_eqns = np.random.choice(N, N_eqns, replace=False)
    t_eqns = T_star[:, idx_t_eqns][idx_x_eqns,:].flatten()[:,None]
    x_eqns = X_star[:, idx_t_eqns][idx_x_eqns,:].flatten()[:,None]
    y_eqns = Y_star[:, idx_t_eqns][idx_x_eqns,:].flatten()[:,None]
    z_eqns = Z_star[:, idx_t_eqns][idx_x_eqns,:].flatten()[:,None]
        
    model = HFM(t_data, x_data, y_data, z_data, c_data,
                u_data, v_data, w_data, p_data,
                t_eqns, x_eqns, y_eqns, z_eqns,
                config['layers'], config['batch_size'],
                Pec = 1.0/0.0101822, Rey = 1.0/0.0101822,
                activation=config['activation'])
    
    # --- MODIFIED: Two-stage training ---
    # 1. Train with Adam
    model.train_adam(total_time = config['adam_time'], 
                     learning_rate = config['learning_rate'], 
                     lambda_physics = config['lambda_physics'],
                     n_iter_print = 500)
    
    # 2. Optionally train with L-BFGS
    if config['use_lbfgs']:
        model.train_lbfgs(lambda_physics = config['lambda_physics'])
    
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
    
    c_pred, u_pred, v_pred, w_pred, p_pred = model.predict(t_test, x_test, y_test, z_test)
    
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
        # (VTK saving loop, unchanged)
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
        grid.save(os.path.join(vtk_dir, f"allresult_{snap}.vtp"))
    
    mat_path = os.path.join(results_dir, "predictions.mat")
    scipy.io.savemat(mat_path,
                     {'c_pred':c_pred, 'u_pred':u_pred, 'v_pred':v_pred, 'w_pred':w_pred, 'p_pred':p_pred,
                      'error_c': error_c, 'error_u': error_u, 'error_v': error_v, 'error_w': error_w, 'error_p': error_p})
    
    return (error_c, error_u, error_v, error_w, error_p)

# --- Main execution block for Member A ---
if __name__ == "__main__": 
    
    BASE_CONFIG = {
        'learning_rate': 1e-3, 
        'lambda_physics': 0.5, 
        'batch_size': 10000,
        'data_path': '../Data/sortedfifty_data.mat',
        'activation': tf.nn.tanh,
        'layers': [4] + 10*[5*50] + [5] # Use baseline arch
    }
    
    log_file_path = "../Results/Member_A_Optimizer_results.txt"
    pathlib.Path("../Results/").mkdir(exist_ok=True)
    with open(log_file_path, 'w') as f:
        f.write("MEMBER A OPTIMIZER EXPERIMENT LOG\n")
        f.write("="*30 + "\n")
        f.write("Experiment, Error_c, Error_u, Error_v, Error_w, Error_p\n")

    # --- Experiment 6: Optimizer Test ---
    print("\n" + "="*50)
    print("MEMBER A: STARTING EXPERIMENT 6 (OPTIMIZER TEST)")
    print("="*50)
    
    optimizer_tests = {
        "Adam_Only_1.0h": {
            'adam_time': 1.0,
            'use_lbfgs': False
        },
        "Adam_0.8h_LBFGS": {
            'adam_time': 0.8,
            'use_lbfgs': True
        }
    }

    for name, settings in optimizer_tests.items():
        config = BASE_CONFIG.copy()
        config['name'] = f"Exp6_Opt_{name}"
        config['adam_time'] = settings['adam_time']
        config['use_lbfgs'] = settings['use_lbfgs']
        config['results_dir'] = f"../Results/Member_A/{config['name']}"
        
        # Reset graph for each run
        tf.reset_default_graph() 
        errors = run_experiment(config)
        
        with open(log_file_path, 'a') as f:
            f.write(f"{config['name']}, {errors[0]}, {errors[1]}, {errors[2]}, {errors[3]}, {errors[4]}\n")

    print("\n" + "="*50)
    print("MEMBER A: OPTIMIZER EXPERIMENTS COMPLETE.")
    print(f"Results summary saved to {log_file_path}")
    print("="*50)