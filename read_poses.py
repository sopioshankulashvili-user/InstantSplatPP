import numpy as np

#read np file ours_1000/pose_optimized.npy and ours_1000/pose_interpolated.npy
pose_optimized = np.load("ours_1000/pose_optimized.npy")
pose_interpolated = np.load("ours_1000/pose_interpolated.npy")


# #print first 5 rows of pose_optimized and pose_interpolated
# print("pose_optimized:")
# print(pose_optimized[3:6])
# print(len(pose_optimized))
# print("pose_interpolated:")
# print(pose_interpolated[3:6])
# print(len(pose_interpolated))

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pandas as pd

def analyze_poses(poses, label):
    # 1. Extract Translation (last column, first 3 rows)
    translations = poses[:, :3, 3]
    
    # 2. Extract Rotation and convert to Euler angles (Degrees)
    rotations = poses[:, :3, :3]
    euler_angles = R.from_matrix(rotations).as_euler('xyz', degrees=True)
    
    # 3. Calculate Step Sizes (Smoothness)
    # Distance between consecutive camera centers
    step_sizes = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    
    # 4. Compile Statistics
    stats = {
        'Metric': ['PosX', 'PosY', 'PosZ', 'Roll', 'Pitch', 'Yaw', 'StepSize'],
        'Mean': [
            *np.mean(translations, axis=0), 
            *np.mean(euler_angles, axis=0), 
            np.mean(step_sizes)
        ],
        'StdDev': [
            *np.std(translations, axis=0), 
            *np.std(euler_angles, axis=0), 
            np.std(step_sizes)
        ]
    }
    
    return translations, euler_angles, step_sizes, pd.DataFrame(stats)

# Load your data
pose_opt = np.load("ours_1000/pose_optimized.npy")
pose_int = np.load("ours_1000/pose_interpolated.npy")

# Process
t_opt, e_opt, s_opt, df_opt = analyze_poses(pose_opt, "Optimized")
t_int, e_int, s_int, df_int = analyze_poses(pose_int, "Interpolated")

# --- Visualization ---

fig = plt.figure(figsize=(15, 10))

# Plot 1: 3D Trajectory
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot(t_int[:, 0], t_int[:, 1], t_int[:, 2], label='Interpolated Path', alpha=0.6, color='blue')
ax1.scatter(t_opt[:, 0], t_opt[:, 1], t_opt[:, 2], label='Optimized Points', color='red', s=50)
ax1.set_title("3D Camera Trajectory")
ax1.legend()

# Plot 2: Translation over "Time"
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(t_int[:, 0], label='X', alpha=0.8)
ax2.plot(t_int[:, 1], label='Y', alpha=0.8)
ax2.plot(t_int[:, 2], label='Z', alpha=0.8)
ax2.set_title("Translation Components (Interpolated)")
ax2.set_xlabel("Frame Index")
ax2.legend()

# Plot 3: Rotation (Euler Angles)
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(e_int[:, 0], label='Roll', linestyle='--')
ax3.plot(e_int[:, 1], label='Pitch', linestyle='--')
ax3.plot(e_int[:, 2], label='Yaw', linestyle='--')
ax3.set_title("Orientation (Degrees)")
ax3.set_xlabel("Frame Index")
ax3.legend()

# Plot 4: Step Size (Smoothness)
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(s_int, label='Interpolated Step', color='blue')
ax4.set_title("Motion Smoothness (Dist between frames)")
ax4.set_ylabel("Euclidean Distance")
ax4.set_xlabel("Frame Index")

plt.tight_layout()
plt.show()

# Display Statistics
print("\n--- Optimized Poses Stats (N=25) ---")
print(df_opt.to_string(index=False))
print("\n--- Interpolated Poses Stats (N=289) ---")
print(df_int.to_string(index=False))