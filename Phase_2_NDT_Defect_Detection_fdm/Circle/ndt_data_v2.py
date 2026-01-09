import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def k_distribution_ground_truth(X, Y):
    K_val = np.full_like(X, 1.0)
    # circle
    mask_circle = ((X - 0.35)**2 + (Y - 0.35)**2) < 0.15**2
    K_val[mask_circle] = 0.2
    return K_val

def laser_source_known(t, X, Y):
    sigma = 0.05
    cx = 0.5 + 0.3 * np.cos(2 * np.pi * t)
    cy = 0.5 + 0.3 * np.sin(2 * np.pi * t)
    source = 50.0 * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    return source

def solve_fdm_full(nx=80, ny=80, nt_save=100):
    print("="*70)
    print("NDT Data Generation V2 - FDM Solver")
    print("="*70)

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # k matrix
    X, Y = np.meshgrid(x, y, indexing='ij')
    K_raw = k_distribution_ground_truth(X, Y)
    K = gaussian_filter(K_raw, sigma=1.5)  # adding smooth
    
    # CFL condition (When we use FEM we can choose more flexible time step)
    dt_stable = 0.08 * (dx**2) / np.max(K)
    t_total = 1.0
    nt_total = int(t_total / dt_stable) + 1
    
    # time idx
    save_indices = np.linspace(0, nt_total-1, nt_save, dtype=int)
    
    print(f"Spatial grid: {nx} x {ny}")
    print(f"Total time steps: {nt_total}")
    print(f"Saved snapshots: {nt_save}")
    print(f"Time step dt: {dt_stable:.6f}")
    print(f"Grid spacing: dx={dx:.4f}, dy={dy:.4f}")

    t_all = np.linspace(0, t_total, nt_total)
    dt = t_all[1] - t_all[0]
    
    # init
    U_saved = np.zeros((nt_save, nx, ny))
    t_saved = t_all[save_indices]

    u_current = np.zeros((nx, ny))

    inv_dx2 = 1.0 / (dx**2)
    inv_dy2 = 1.0 / (dy**2)
    
    print("\nStart FDM solution...")
    save_counter = 0
    
    for n in range(nt_total):
        u_xx = (np.roll(u_current, -1, axis=0) - 2*u_current + np.roll(u_current, 1, axis=0)) * inv_dx2
        u_yy = (np.roll(u_current, -1, axis=1) - 2*u_current + np.roll(u_current, 1, axis=1)) * inv_dy2

        k_x, k_y = np.gradient(K, dx, dy)
        u_x, u_y = np.gradient(u_current, dx, dy)
        
        # Convection + Diffusion: ∂u/∂t = ∇·(K∇u) + f
        rhs = K * (u_xx + u_yy) + k_x * u_x + k_y * u_y + laser_source_known(t_all[n], X, Y)

        u_current = u_current + dt * rhs
        
        # Dirichlet BC
        u_current[0, :] = 0
        u_current[-1, :] = 0
        u_current[:, 0] = 0
        u_current[:, -1] = 0

        if n in save_indices:
            U_saved[save_counter] = u_current.copy()
            save_counter += 1
            if save_counter % 10 == 0:
                print(f"  Saved snapshot {save_counter}/{nt_save}, t={t_all[n]:.3f}, max(u)={np.max(u_current):.2f}")
    
    print(f"\nModeling completed!")
    print(f"Temperature field statistics:")
    print(f"MAX: {np.max(U_saved):.2f}")
    print(f"MIN: {np.min(U_saved):.2f}")
    print(f"MEAN: {np.mean(U_saved):.2f}")
    
    return x, y, t_saved, U_saved, K, K_raw

def compute_laser_trajectory(t_array):
    cx = 0.5 + 0.3 * np.cos(2 * np.pi * t_array)
    cy = 0.5 + 0.3 * np.sin(2 * np.pi * t_array)
    return cx, cy

def visualize_data(x, y, t, U, K, K_raw, save_dir):
    print("\nvisualization...")
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # k distribution -- first figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im0 = axes[0].imshow(K_raw.T, origin='lower', extent=[0,1,0,1], cmap='RdYlBu_r')
    axes[0].set_title('True K (Raw)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0], label='K')
    
    im1 = axes[1].imshow(K.T, origin='lower', extent=[0,1,0,1], cmap='RdYlBu_r')
    axes[1].set_title('True K (Smoothed)')
    axes[1].set_xlabel('x')
    plt.colorbar(im1, ax=axes[1], label='K')
    
    # Defect location
    circle = plt.Circle((0.35, 0.35), 0.15, color='red', fill=False, linewidth=2)
    axes[2].add_patch(circle)
    axes[2].imshow(K.T, origin='lower', extent=[0,1,0,1], cmap='RdYlBu_r', alpha=0.7)
    axes[2].set_title('Defect Location')
    axes[2].set_xlabel('x')
    axes[2].plot([0.35], [0.35], 'r*', markersize=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'K_distribution.png'), dpi=150)
    plt.close()
    
    # Evolution of the temperature field (four time snapshots) -- second figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    cx, cy = compute_laser_trajectory(t)
    
    time_indices = [0, len(t)//3, 2*len(t)//3, -1]
    for idx, t_idx in enumerate(time_indices):
        im = axes[idx].imshow(U[t_idx].T, origin='lower', extent=[0,1,0,1], cmap='hot')
        axes[idx].set_title(f't = {t[t_idx]:.3f}')
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('y')
        
        # laser position
        axes[idx].plot(cx[t_idx], cy[t_idx], 'c*', markersize=15, label='Laser')
        
        # defect position
        circle = plt.Circle((0.35, 0.35), 0.15, color='cyan', fill=False, linewidth=2, linestyle='--')
        axes[idx].add_patch(circle)
        
        plt.colorbar(im, ax=axes[idx], label='Temperature')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'temperature_evolution.png'), dpi=150)
    plt.close()
    
    # Laser path, similar to the diagram in GSP -- third figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(cx, cy, 'b-', linewidth=2, label='Laser trajectory')
    ax.plot(cx[0], cy[0], 'go', markersize=10, label='Start')
    ax.plot(cx[-1], cy[-1], 'ro', markersize=10, label='End')

    circle = plt.Circle((0.35, 0.35), 0.15, color='red', fill=False, linewidth=2, label='Defect')
    ax.add_patch(circle)
    ax.plot([0.35], [0.35], 'r*', markersize=15)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Laser Scanning Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, 'laser_trajectory.png'), dpi=150)
    plt.close()
    
    print(f"visualization completed, saved to {save_dir}")

def main():
    save_dir = "NDT_Data_V2"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    x, y, t, U, K, K_raw = solve_fdm_full(nx=80, ny=80, nt_save=100)

    if np.isnan(U).any():
        print("\nfinding NaN!")
        return
    
    # add noise
    noise_level = 0.02
    U_noisy = U + noise_level * np.std(U) * np.random.randn(*U.shape)
    
    # save data
    print(f"\nSaving data to {save_dir}...")
    np.savez(
        os.path.join(save_dir, 'ndt_data_full.npz'),
        x=x,
        y=y,
        t=t,
        U_true=U,           # no noise
        U_measured=U_noisy,  # noise
        K_true=K,           # smooth
        K_raw=K_raw,        # no smooth
        laser_cx=compute_laser_trajectory(t)[0],
        laser_cy=compute_laser_trajectory(t)[1]
    )
    
    # visualize
    visualize_data(x, y, t, U_noisy, K, K_raw, save_dir)
    
    print("\n" + "="*70)
    print("data OK!")
    print(f"data saved to: {save_dir}/ndt_data_full.npz")
    print(f"matrix sizes: U={U.shape}, K={K.shape}")
    print("="*70)

if __name__ == "__main__":
    main()
