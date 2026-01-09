import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

class FEMSolver:
    
    def __init__(self, x, y, dt=0.001):

        self.x = x
        self.y = y
        self.nx = len(x)
        self.ny = len(y)
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.dt = dt

        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
        print(f"FEM Solver initialized: {self.nx}x{self.ny} grid, dt={dt}")
    
    def laser_source(self, t, X, Y):
        sigma = 0.05
        cx = 0.5 + 0.3 * np.cos(2 * np.pi * t)
        cy = 0.5 + 0.3 * np.sin(2 * np.pi * t)
        source = 50.0 * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        return source
    
    def build_fem_matrices(self, K_field):

        nx, ny = self.nx, self.ny
        n_total = nx * ny
        
        # The finite difference approximation using the 5-point template
        # -∇·(K∇u) ≈ -(K_{i+1/2}(u_{i+1}-u_i)/dx^2 + ...)
        
        # Used for constructing sparse matrices
        row_idx = []
        col_idx = []
        data = []
        
        inv_dx2 = 1.0 / (self.dx**2)
        inv_dy2 = 1.0 / (self.dy**2)
        
        def idx(i, j):
            # change 2D index to 1D index
            return i * ny + j
        
        for i in range(nx):
            for j in range(ny):
                n = idx(i, j)
                
                # Dirichlet
                if i == 0 or i == nx-1 or j == 0 or j == ny-1:
                    row_idx.append(n)
                    col_idx.append(n)
                    data.append(1.0)
                else:
                    # internal node, five-point template
                    K_c = K_field[i, j]
                    
                    # Interface thermal conductivity (harmonic average)
                    K_ip = 2 * K_field[i, j] * K_field[i+1, j] / (K_field[i, j] + K_field[i+1, j] + 1e-10)
                    K_im = 2 * K_field[i, j] * K_field[i-1, j] / (K_field[i, j] + K_field[i-1, j] + 1e-10)
                    K_jp = 2 * K_field[i, j] * K_field[i, j+1] / (K_field[i, j] + K_field[i, j+1] + 1e-10)
                    K_jm = 2 * K_field[i, j] * K_field[i, j-1] / (K_field[i, j] + K_field[i, j-1] + 1e-10)
                    
                    # center node
                    diag = -(K_ip + K_im) * inv_dx2 - (K_jp + K_jm) * inv_dy2
                    row_idx.append(n)
                    col_idx.append(n)
                    data.append(diag)
                    
                    # Neighbor in the x direction
                    row_idx.append(n)
                    col_idx.append(idx(i+1, j))
                    data.append(K_ip * inv_dx2)
                    
                    row_idx.append(n)
                    col_idx.append(idx(i-1, j))
                    data.append(K_im * inv_dx2)
                    
                    # Neighbor in the y direction
                    row_idx.append(n)
                    col_idx.append(idx(i, j+1))
                    data.append(K_jp * inv_dy2)
                    
                    row_idx.append(n)
                    col_idx.append(idx(i, j-1))
                    data.append(K_jm * inv_dy2)
        
        A = sp.csr_matrix((data, (row_idx, col_idx)), shape=(n_total, n_total))
        
        # mass matrix
        M = sp.identity(n_total, format='csr')
        
        return A, M
    
    def solve_transient(self, K_field, t_array, u0=None, verbose=True):

        nt = len(t_array)
        nx, ny = self.nx, self.ny
        
        # Init
        if u0 is None:
            u = np.zeros((nx, ny))
        else:
            u = u0.copy()
        
        U_history = np.zeros((nt, nx, ny))
        U_history[0] = u
        
        # construct FEM matrices
        if verbose:
            print("construct FEM matrices...")
        A, M = self.build_fem_matrices(K_field)
        
        # Time integration matrix (M - dt*A)
        LHS = M - self.dt * A
        
        # LU Decomposition
        if verbose:
            print("LU decomposition...")
        lu = spla.splu(LHS.tocsc())
        
        # time steps
        if verbose:
            print(f"time stepping: {nt} steps...")
        
        for n in range(1, nt):
            t = t_array[n]

            f = self.laser_source(t, self.X, self.Y).flatten()
            
            # M*u + dt*f
            rhs = M @ u.flatten() + self.dt * f

            u_new = lu.solve(rhs).reshape(nx, ny)

            u_new[0, :] = 0
            u_new[-1, :] = 0
            u_new[:, 0] = 0
            u_new[:, -1] = 0
            
            U_history[n] = u_new
            u = u_new
            
            if verbose and n % (nt // 10) == 0:
                print(f"t={t:.3f}, max(u)={np.max(u):.2f}")
        
        if verbose:
            print("solved successfully")
        
        return U_history
    
    def compute_gradient_magnitude(self, U):
        if U.ndim == 3:
            # time average of gradients
            grad_mag_sum = np.zeros((self.nx, self.ny))
            for t_idx in range(U.shape[0]):
                grad_x, grad_y = np.gradient(U[t_idx], self.dx, self.dy)
                grad_mag_sum += np.sqrt(grad_x**2 + grad_y**2)
            return grad_mag_sum / U.shape[0]
        else:
            grad_x, grad_y = np.gradient(U, self.dx, self.dy)
            return np.sqrt(grad_x**2 + grad_y**2)
    
    def interpolate_to_grid(self, U_measured, x_measured, y_measured, t_measured):
        # need to interpolate grid?
        if (len(x_measured) == self.nx and len(y_measured) == self.ny and
            np.allclose(x_measured, self.x) and np.allclose(y_measured, self.y)):
            return U_measured
        
        print("interpolating...")
        nt = len(t_measured)
        U_interp = np.zeros((nt, self.nx, self.ny))
        
        for t_idx in range(nt):
            interp_func = RegularGridInterpolator(
                (x_measured, y_measured),
                U_measured[t_idx],
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )

            points = np.stack([self.X.ravel(), self.Y.ravel()], axis=-1)
            U_interp[t_idx] = interp_func(points).reshape(self.nx, self.ny)
        
        return U_interp

def compare_baseline_vs_measured(measured_data, save_dir='NDT_Data_V2'):

    print("\n" + "="*70)
    print("compare (K=1) situation with measured data")
    print("="*70)
    
    # load measured data
    x = measured_data['x']
    y = measured_data['y']
    t = measured_data['t']
    U_measured = measured_data['U_measured']
    
    # init FEMSolver
    dt = t[1] - t[0] if len(t) > 1 else 0.001
    solver = FEMSolver(x, y, dt=dt)
    
    # Solve the base case where k = 1
    print("\nSolve the base case where k = 1...")
    K_baseline = np.ones_like(solver.X)
    U_baseline = solver.solve_transient(K_baseline, t, verbose=True)
    
    # residual calculation
    print("\nresidual calculation...")
    residual = U_measured - U_baseline
    
    # Calculate the gradient difference
    print("Calculate the gradient difference...")
    grad_measured = solver.compute_gradient_magnitude(U_measured)
    grad_baseline = solver.compute_gradient_magnitude(U_baseline)
    grad_diff = np.abs(grad_measured - grad_baseline)
    
    # Time average residual
    residual_mean = np.mean(np.abs(residual), axis=0)
    
    # statistics
    print(f"\nresidual statistics:")
    print(f"MEAN: {np.mean(residual_mean):.4f}")
    print(f"MAX: {np.max(residual_mean):.4f}")
    print(f"STD: {np.std(residual_mean):.4f}")
    
    print(f"\nGradient difference statistics:")
    print(f"MEAN: {np.mean(grad_diff):.4f}")
    print(f"MAX: {np.max(grad_diff):.4f}")
    
    # visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # first row: baseline vs measured on residual
    im0 = axes[0, 0].imshow(U_baseline[-1].T, origin='lower', extent=[0,1,0,1], cmap='hot')
    axes[0, 0].set_title('Baseline (K=1)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(U_measured[-1].T, origin='lower', extent=[0,1,0,1], cmap='hot')
    axes[0, 1].set_title('Measured')
    axes[0, 1].set_xlabel('x')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(residual_mean.T, origin='lower', extent=[0,1,0,1], cmap='RdBu_r')
    axes[0, 2].set_title('Time-Avg Residual')
    axes[0, 2].set_xlabel('x')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # second row: baseline vs measured on gradient
    im3 = axes[1, 0].imshow(grad_baseline.T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    axes[1, 0].set_title('Gradient (Baseline)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(grad_measured.T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    axes[1, 1].set_title('Gradient (Measured)')
    axes[1, 1].set_xlabel('x')
    plt.colorbar(im4, ax=axes[1, 1])
    
    im5 = axes[1, 2].imshow(grad_diff.T, origin='lower', extent=[0,1,0,1], cmap='hot')
    axes[1, 2].set_title('Gradient Difference')
    axes[1, 2].set_xlabel('x')
    plt.colorbar(im5, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/baseline_comparison.png', dpi=150)
    plt.close()
    
    print(f"\nconverted to image: {save_dir}/baseline_comparison.png")
    
    return residual, grad_diff, U_baseline

if __name__ == "__main__":

    import os
    
    data_file = 'NDT_Data_V2/ndt_data_full.npz'
    if os.path.exists(data_file):
        data = np.load(data_file)
        residual, grad_diff, U_baseline = compare_baseline_vs_measured(data)
    else:
        print(f"please run ndt_data_v2.py first!")
