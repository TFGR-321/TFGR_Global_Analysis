import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Phase 21 — TFGR Scaling Relations
#   Using 4 galaxies: NGC 2403, 3198, 2903, 5055
# ==========================================

# -------------------------------
# 1. 手入力したフィット結果
#    （Phase 19/20 で得られたもの）
# -------------------------------
names   = np.array(["NGC 2403", "NGC 3198", "NGC 2903", "NGC 5055"])

# V0 [km/s]
V0      = np.array([108.33, 119.47, 174.73, 152.34])
V0_err  = np.array([  0.43,   0.73,   1.18,   0.72])

# rc [kpc]
rc      = np.array([ 7.16, 13.64,  1.85, 11.80])
rc_err  = np.array([ 0.06,  0.26,  0.09,  0.43])

# n (shape exponent)
n       = np.array([ 2.12,  3.51,  2.58,  1.82])
n_err   = np.array([ 0.04,  0.19,  0.33,  0.11])

# disk scaling factor f_disk
# 2403, 3198 は 1.0 固定（実際のフィットではスケール自由度無し）
f_disk      = np.array([1.00, 1.00, 0.449, 0.669])
f_disk_err  = np.array([0.00, 0.00, 0.011, 0.007])

# 色を決めておく（任意）
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


# -------------------------------
# 2. コンソール用テーブル表示
# -------------------------------
print("\n===== TFGR Parameter Summary (4 Galaxies) =====")
print("Galaxy   |   V0 [km/s]   rc [kpc]   n    f_disk")
print("-----------------------------------------------")
for i in range(len(names)):
    print(f"{names[i]:8s}| "
          f"{V0[i]:8.2f}   {rc[i]:7.2f}   {n[i]:4.2f}   {f_disk[i]:6.3f}")
print("===============================================\n")


# -------------------------------
# 3. rc vs V0
# -------------------------------
plt.figure(figsize=(7,5))
for i in range(len(names)):
    plt.errorbar(rc[i], V0[i],
                 xerr=rc_err[i], yerr=V0_err[i],
                 fmt="o", color=colors[i], capsize=3)
    plt.text(rc[i]*1.02, V0[i]*1.01, names[i], fontsize=9)

plt.xlabel("TFGR scale radius rc [kpc]")
plt.ylabel("TFGR velocity scale V0 [km/s]")
plt.title("TFGR Scaling: V0 vs rc")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# -------------------------------
# 4. rc vs n
# -------------------------------
plt.figure(figsize=(7,5))
for i in range(len(names)):
    plt.errorbar(rc[i], n[i],
                 xerr=rc_err[i], yerr=n_err[i],
                 fmt="o", color=colors[i], capsize=3)
    plt.text(rc[i]*1.02, n[i]*1.01, names[i], fontsize=9)

plt.xlabel("TFGR scale radius rc [kpc]")
plt.ylabel("Shape index n")
plt.title("TFGR Scaling: n vs rc")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# -------------------------------
# 5. rc vs f_disk
# -------------------------------
plt.figure(figsize=(7,5))
for i in range(len(names)):
    plt.errorbar(rc[i], f_disk[i],
                 xerr=rc_err[i],
                 yerr=f_disk_err[i] if f_disk_err[i] > 0 else None,
                 fmt="o", color=colors[i], capsize=3)
    # f_disk が固定値のものは凡例でわかるようにマーク
    label = names[i] + (" (fixed)" if f_disk_err[i] == 0 else "")
    plt.text(rc[i]*1.02, f_disk[i]*1.01, label, fontsize=9)

plt.xlabel("TFGR scale radius rc [kpc]")
plt.ylabel("Disk scaling factor f_disk")
plt.ylim(0.3, 1.1)
plt.title("TFGR Scaling: f_disk vs rc")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
