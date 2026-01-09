import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label
from scipy.optimize import minimize
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ndt_fem_solver import FEMSolver

class DefectLocalizer:
    
    def __init__(self, x, y, t, U_measured, threshold_percentile=92):

        self.x = x
        self.y = y
        self.t = t
        self.U_measured = U_measured # measured temperature field
        self.threshold_percentile = threshold_percentile # grad difference threshold
        
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
        print(f"DefectLocalizer initialized: {len(x)}x{len(y)} map")
    
    def compute_gradient_anomaly(self, U_baseline):

        print("\ncomputing grad diff...")
        
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        
        # time average grad
        grad_mag_measured = np.zeros_like(self.X)
        grad_mag_baseline = np.zeros_like(self.X)
        
        for t_idx in range(len(self.t)):
            # measured gradient
            grad_x_m, grad_y_m = np.gradient(self.U_measured[t_idx], dx, dy)
            grad_mag_measured += np.sqrt(grad_x_m**2 + grad_y_m**2)
            
            # standard baseline gradient
            grad_x_b, grad_y_b = np.gradient(U_baseline[t_idx], dx, dy)
            grad_mag_baseline += np.sqrt(grad_x_b**2 + grad_y_b**2)
        
        grad_mag_measured /= len(self.t)
        grad_mag_baseline /= len(self.t)
        
        # gradient anomaly
        grad_anomaly = np.abs(grad_mag_measured - grad_mag_baseline)
        grad_anomaly = grad_anomaly / (np.max(grad_anomaly) + 1e-10)
        
        print(f"MAX grad difference: {np.max(grad_anomaly):.4f}")
        
        return grad_anomaly
    
    def compute_residual_anomaly(self, U_baseline):

        print("computing u diff...")

        residual = self.U_measured - U_baseline
        residual_mean = np.mean(np.abs(residual), axis=0)

        residual_anomaly = residual_mean / (np.max(residual_mean) + 1e-10)
        
        print(f"MAX u difference: {np.max(residual_anomaly):.4f}")
        
        return residual_anomaly
    
    def locate_defects(self, U_baseline, use_gradient=True, use_residual=True):

        print("\n" + "="*70)
        print("Locate defects")
        print("="*70)
        
        anomaly_maps = []
        
        if use_gradient:
            grad_anomaly = self.compute_gradient_anomaly(U_baseline)
            anomaly_maps.append(grad_anomaly)
        
        if use_residual:
            residual_anomaly = self.compute_residual_anomaly(U_baseline)
            anomaly_maps.append(residual_anomaly)
        
        # Composite abnormality map (weighted average)
        if len(anomaly_maps) == 0:
            raise ValueError("At least one anomaly detection method is required.")
        
        anomaly_map = np.mean(anomaly_maps, axis=0)
        
        # gaussian smooth
        anomaly_smooth = gaussian_filter(anomaly_map, sigma=2.0)
        
        # Divided into a percentage system, convenient for threshold determination
        threshold = np.percentile(anomaly_smooth, self.threshold_percentile)
        defect_mask = anomaly_smooth > threshold
        
        # moving small position
        labeled, num_features = label(defect_mask)
        
        print(f"\nFinding {num_features} numbers of defect points")
        
        # If too small, ignore
        min_size = 10  # smallest size
        filtered_mask = np.zeros_like(defect_mask)
        
        for i in range(1, num_features + 1):
            region = (labeled == i)
            if np.sum(region) >= min_size:
                filtered_mask |= region
                
                # computer center
                y_indices, x_indices = np.where(region)
                center_x = self.x[int(np.mean(x_indices))]
                center_y = self.y[int(np.mean(y_indices))]
                print(f"  defect {i}: center ({center_x:.3f}, {center_y:.3f}), size {np.sum(region)} points")
        
        print(f"\nsaved {np.sum(filtered_mask > 0)} numbers of defect points")
        
        return filtered_mask, anomaly_smooth

class DefectQuantifier:
    
    def __init__(self, x, y, t, U_measured, defect_mask):

        self.x = x
        self.y = y
        self.t = t
        self.U_measured = U_measured
        self.defect_mask = defect_mask
        
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
        # init FEMSolver
        dt = t[1] - t[0] if len(t) > 1 else 0.001
        self.solver = FEMSolver(x, y, dt=dt)
        
        print(f"\nDefectQuantifier initialized:")
        print(f"  defect point num: {np.sum(defect_mask)}")
    
    def optimize_local_k(self, k_init=0.5, bounds=(0.1, 1.0), method='L-BFGS-B'):

        print("\n" + "="*70)
        print("Defect quantification (Local K-value optimization)")
        print("="*70)
        
        # Init K
        K_field = np.ones_like(self.X)
        K_field[self.defect_mask] = k_init
        
        # defect area average K value init
        defect_indices = np.where(self.defect_mask)
        n_defect = len(defect_indices[0])
        
        print(f"\nOptimize parameters:")
        print(f"defect point num: {n_defect}")
        print(f"Init K value: {k_init}")
        print(f"K bounds: {bounds}")
        
        # optimize objective
        def objective(k_defect):
            K_test = np.ones_like(self.X)
            K_test[self.defect_mask] = k_defect[0]
            
            # FEM solve
            U_sim = self.solver.solve_transient(K_test, self.t, verbose=False)
            
            # Computational error (only in critical areas)
            # Focus on the temperature differences near the defect area
            weight = gaussian_filter(self.defect_mask.astype(float), sigma=3.0)
            weight = weight / (np.max(weight) + 1e-10)
            
            error = 0.0
            for t_idx in range(len(self.t)):
                diff = (U_sim[t_idx] - self.U_measured[t_idx]) * weight
                error += np.sum(diff**2)
            
            error /= len(self.t)
            
            return error

        print("\nbegin optimization...")
        result = minimize(
            objective,
            x0=[k_init],
            method=method,
            bounds=[bounds],
            options={'maxiter': 100, 'disp': True}
        )
        
        print("\nOptimization complete!")
        print(f"K value: {result.x[0]:.4f}")
        print(f"error: {result.fun:.6f}")
        print(f"iters: {result.nit}")

        # construct K
        K_optimal = np.ones_like(self.X)
        K_optimal[self.defect_mask] = result.x[0]

        # smoothing to fit ground truth
        # K_optimal = gaussian_filter(K_optimal, sigma=1.5)

        return K_optimal, result

def full_pipeline(data_path, save_dir='NDT_Data_V2', visualize=True):

    print("\n" + "="*70)
    print("NDT full pipeline")
    print("="*70)
    
    # load data
    print(f"\nloading data: {data_path}")
    data = np.load(data_path)
    
    x = data['x']
    y = data['y']
    t = data['t']
    U_measured = data['U_measured']
    K_true = data['K_true']
    
    print(f"data shape: U={U_measured.shape}, K={K_true.shape}")
    
    # FEM solve
    print("\nFirst, solve baseline")
    dt = t[1] - t[0]
    solver = FEMSolver(x, y, dt=dt)
    K_baseline = np.ones_like(solver.X)
    U_baseline = solver.solve_transient(K_baseline, t, verbose=True)

    # defect localization
    localizer = DefectLocalizer(x, y, t, U_measured, threshold_percentile=92) # maybe 90 will be better?
    defect_mask, anomaly_map = localizer.locate_defects(U_baseline)
    
    # defect quantification
    quantifier = DefectQuantifier(x, y, t, U_measured, defect_mask)
    K_optimal, opt_result = quantifier.optimize_local_k(k_init=0.25, bounds=(0.15, 0.8))
    
    # verification
    print("\n" + "="*70)
    print("verification")
    print("="*70)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # true defect area
    true_defect = (((X - 0.35)**2 + (Y - 0.35)**2) < 0.15**2)

    # IoU
    intersection = np.sum(defect_mask & true_defect)
    union = np.sum(defect_mask | true_defect)
    iou = intersection / union if union > 0 else 0
    
    print(f"\nlocalization:")
    print(f"IoU: {iou:.3f}")
    print(f"defect point found: {np.sum(defect_mask)}")
    print(f"true defect point num: {np.sum(true_defect)}")
    
    # K error
    k_true_defect = K_true[true_defect].mean()
    k_pred_defect = K_optimal[defect_mask].mean()
    k_error = np.abs(k_true_defect - k_pred_defect)
    
    print(f"\nK:")
    print(f"true K: {k_true_defect:.4f}")
    print(f"measured K: {k_pred_defect:.4f}")
    print(f"absolute error: {k_error:.4f}")
    print(f"relative error: {k_error/k_true_defect*100:.2f}%")
    
    # saving result
    np.savez(
        os.path.join(save_dir, 'ndt_result_v2.npz'),
        K_optimal=K_optimal,
        defect_mask=defect_mask,
        anomaly_map=anomaly_map,
        opt_result=opt_result.x,
        iou=iou,
        k_error=k_error
    )
    print(f"\nresult saved: {save_dir}/ndt_result_v2.npz")
    
    # visualize
    if visualize:
        visualize_results(x, y, K_true, K_optimal, defect_mask, true_defect, 
                         anomaly_map, save_dir)
    
    return K_optimal, defect_mask

def visualize_results(x, y, K_true, K_pred, defect_mask, true_defect, 
                     anomaly_map, save_dir):
    print("\nGenerate visualization...")
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # first row: K comparison
    im0 = axes[0, 0].imshow(K_true.T, origin='lower', extent=[0,1,0,1], 
                           cmap='RdYlBu_r', vmin=0.2, vmax=1.0)
    axes[0, 0].set_title('True K')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0], label='K')

    from matplotlib.patches import Circle
    circle = Circle((0.35, 0.35), 0.15, color='black', fill=False, linewidth=2, linestyle='--')
    axes[0, 0].add_patch(circle)
    
    im1 = axes[0, 1].imshow(K_pred.T, origin='lower', extent=[0,1,0,1], 
                           cmap='RdYlBu_r', vmin=0.2, vmax=1.0)
    axes[0, 1].set_title('Predicted K')
    axes[0, 1].set_xlabel('x')
    plt.colorbar(im1, ax=axes[0, 1], label='K')

    circle2 = Circle((0.35, 0.35), 0.15, color='cyan', fill=False, linewidth=2, linestyle='--')
    axes[0, 1].add_patch(circle2)
    
    error = np.abs(K_true - K_pred)
    im2 = axes[0, 2].imshow(error.T, origin='lower', extent=[0,1,0,1], 
                           cmap='hot')
    axes[0, 2].set_title('Absolute Error')
    axes[0, 2].set_xlabel('x')
    plt.colorbar(im2, ax=axes[0, 2], label='|K_true - K_pred|')
    
    # second row: defect localization
    axes[1, 0].imshow(anomaly_map.T, origin='lower', extent=[0,1,0,1], 
                     cmap='hot')
    axes[1, 0].set_title('Anomaly Map')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].contour(X, Y, true_defect, colors='cyan', linewidths=2, levels=[0.5])
    
    axes[1, 1].imshow(defect_mask.T, origin='lower', extent=[0,1,0,1], 
                     cmap='binary')
    axes[1, 1].set_title('Detected Defect')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].contour(X, Y, true_defect, colors='red', linewidths=2, levels=[0.5])
    
    # localization quality
    tp = defect_mask & true_defect
    fp = defect_mask & ~true_defect
    fn = ~defect_mask & true_defect
    
    vis_map = np.zeros_like(defect_mask, dtype=int)
    vis_map[tp] = 3  # True Positive
    vis_map[fp] = 2  # False Positive
    vis_map[fn] = 1  # False Negative
    
    cmap_custom = plt.matplotlib.colors.ListedColormap(['white', 'blue', 'red', 'green'])
    axes[1, 2].imshow(vis_map.T, origin='lower', extent=[0,1,0,1], cmap=cmap_custom)
    axes[1, 2].set_title('Localization Quality')
    axes[1, 2].set_xlabel('x')
    
    # figure legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='True Positive'),
        Patch(facecolor='red', label='False Positive'),
        Patch(facecolor='blue', label='False Negative')
    ]
    axes[1, 2].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ndt_results_v2.png'), dpi=150)
    plt.close()
    
    print(f"visualization saved: {save_dir}/ndt_results_v2.png")

if __name__ == "__main__":
    data_path = 'NDT_Data_V2/ndt_data_full.npz'
    
    if os.path.exists(data_path):
        K_optimal, defect_mask = full_pipeline(data_path)
    else:
        print(f"Error: no data found: {data_path}")
        print("please first run: python ndt_data_v2.py")
