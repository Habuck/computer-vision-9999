import os
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results/part_b", exist_ok=True)
os.makedirs("report_latex/figures", exist_ok=True)

DATA_DIR = "data/teddy"
# Middlebury 2003 quarter-size ground truth stores disparity * 4 (0 = unknown).
GT_SCALE = 4.0
# Teddy is distributed without calibration; these values only fix the metric scale of the point cloud.
FOCAL_PX = 450.0
BASELINE_M = 0.16
VMAX = 64


def save_fig(name, dpi=180):
    plt.savefig(f"results/part_b/{name}", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"report_latex/figures/{name}", dpi=dpi, bbox_inches="tight")
    plt.close()


def load_teddy():
    imgL = cv2.imread(f"{DATA_DIR}/im2.png")
    imgR = cv2.imread(f"{DATA_DIR}/im6.png")
    gtL = cv2.imread(f"{DATA_DIR}/disp2.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / GT_SCALE
    gtR = cv2.imread(f"{DATA_DIR}/disp6.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / GT_SCALE
    return imgL, imgR, gtL, gtR


def nonoccluded_mask(gtL, gtR, tol=1.0):
    h, w = gtL.shape
    xx = np.tile(np.arange(w), (h, 1))
    xr = np.round(xx - gtL).astype(int)
    inside = (gtL > 0) & (xr >= 0) & (xr < w)
    xr_c = np.clip(xr, 0, w - 1)
    dR = np.take_along_axis(gtR, xr_c, axis=1)
    return inside & (np.abs(dR - gtL) <= tol)


def sampson_distance(F, p1, p2):
    x1 = np.hstack([p1, np.ones((len(p1), 1))])
    x2 = np.hstack([p2, np.ones((len(p2), 1))])
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    num = np.sum(x2 * Fx1, axis=1) ** 2
    den = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2
    return np.sqrt(num / den)


def draw_line(img, line, color):
    w = img.shape[1]
    a, b, c = line
    p0 = (0, int(round(-c / b)))
    p1 = (w, int(round(-(c + a * w) / b)))
    cv2.line(img, p0, p1, color, 1, cv2.LINE_AA)


def compute_epipolar_geometry(imgL, imgR):
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(grayL, None)
    kp2, des2 = sift.detectAndCompute(grayR, None)

    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.999)
    inl = mask.ravel() == 1
    in1, in2 = pts1[inl], pts2[inl]

    lines1 = cv2.computeCorrespondEpilines(in2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(in1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    d1 = np.abs(np.sum(lines1[:, :2] * in1, axis=1) + lines1[:, 2]) / np.hypot(lines1[:, 0], lines1[:, 1])
    d2 = np.abs(np.sum(lines2[:, :2] * in2, axis=1) + lines2[:, 2]) / np.hypot(lines2[:, 0], lines2[:, 1])
    epi_err = float(np.mean((d1 + d2) / 2.0))
    sampson_err = float(np.mean(sampson_distance(F.astype(np.float64), in1.astype(np.float64), in2.astype(np.float64))))
    vertical_offset = float(np.mean(np.abs(in1[:, 1] - in2[:, 1])))

    visL, visR = imgL.copy(), imgR.copy()
    rng = np.random.default_rng(12)
    order = np.argsort(in1[:, 1])
    for idx in order[np.linspace(0, len(order) - 1, min(20, len(order)), dtype=int)]:
        color = tuple(int(c) for c in rng.integers(40, 255, 3))
        draw_line(visL, lines1[idx], color)
        draw_line(visR, lines2[idx], color)
        cv2.circle(visL, tuple(np.int32(in1[idx])), 4, color, -1)
        cv2.circle(visR, tuple(np.int32(in2[idx])), 4, color, -1)

    gap = np.full((imgL.shape[0], 10, 3), 255, np.uint8)
    epi_pair = np.hstack([visL, gap, visR])
    cv2.imwrite("results/part_b/epipolar_lines.png", epi_pair)
    cv2.imwrite("report_latex/figures/epipolar_lines.png", epi_pair)

    stats = {
        "SIFT Keypoints (L)": len(kp1),
        "SIFT Keypoints (R)": len(kp2),
        "Ratio-Test Matches": len(good),
        "RANSAC Inliers": int(inl.sum()),
        "Inlier Ratio (%)": round(100.0 * inl.sum() / len(good), 2),
        "Mean Epipolar Error (px)": round(epi_err, 4),
        "Mean Sampson Error (px)": round(sampson_err, 4),
        "Mean |yL - yR| of Inliers (px)": round(vertical_offset, 4),
    }
    return F, stats


def build_matcher(algo, bsize, ndisp):
    if algo == "BM":
        m = cv2.StereoBM_create(numDisparities=ndisp, blockSize=bsize)
        m.setUniquenessRatio(10)
        m.setSpeckleWindowSize(100)
        m.setSpeckleRange(2)
        return m
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=ndisp,
        blockSize=bsize,
        P1=8 * 3 * bsize**2,
        P2=32 * 3 * bsize**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_disparity(algo, bsize, ndisp, imgL, imgR, repeats=5):
    if algo == "BM":
        srcL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        srcR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    else:
        srcL, srcR = imgL, imgR
    matcher = build_matcher(algo, bsize, ndisp)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        raw = matcher.compute(srcL, srcR)
        times.append((time.perf_counter() - t0) * 1000.0)
    return raw.astype(np.float32) / 16.0, float(np.median(times))


def write_ply(filename, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3).astype(np.uint8)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(verts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(filename, "w") as f:
        f.write(header)
        for v, c in zip(verts, colors):
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]} {c[1]} {c[2]}\n")


def reproject(disp, imgL, min_disp=10.0):
    h, w = disp.shape
    cx, cy = 0.5 * w, 0.5 * h
    Q = np.float32([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, FOCAL_PX],
        [0, 0, 1.0 / BASELINE_M, 0],
    ])
    pts = cv2.reprojectImageTo3D(disp, Q)
    mask = disp >= min_disp
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    return pts[mask], colors[mask]


def plot_cloud(ax, pts, cols, elev, azim, title, limits, max_pts=30000):
    keep = np.all([(pts[:, i] >= lo) & (pts[:, i] <= hi) for i, (lo, hi) in enumerate(limits)], axis=0)
    pts, cols = pts[keep], cols[keep]
    step = max(1, len(pts) // max_pts)
    p, c = pts[::step], cols[::step] / 255.0
    ax.scatter(p[:, 0], p[:, 2], -p[:, 1], c=c, s=1.5, alpha=0.9, linewidths=0)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[2])
    ax.set_zlim(-limits[1][1], -limits[1][0])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Depth Z (m)")
    if elev > 60:
        ax.set_zticks([])
    else:
        ax.set_zlabel("Y (m)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontweight="bold", fontsize=10)


def run_part_b():
    imgL, imgR, gtL, gtR = load_teddy()
    h, w = gtL.shape
    nonocc = nonoccluded_mask(gtL, gtR)
    print(f"Teddy {w}x{h}: GT disparity range {gtL[gtL > 0].min():.2f}-{gtL.max():.2f} px, "
          f"non-occluded pixels {100.0 * nonocc.mean():.2f}%")

    F, epi_stats = compute_epipolar_geometry(imgL, imgR)
    print("Fundamental matrix F:\n", np.array2string(F, precision=6))
    for k, v in epi_stats.items():
        print(f"  {k}: {v}")
    pd.DataFrame([epi_stats]).to_csv("results/part_b/epipolar_part_b.csv", index=False)

    configs = [
        ("BM", 9, 64),
        ("BM", 15, 64),
        ("BM", 21, 64),
        ("SGBM", 3, 64),
        ("SGBM", 5, 64),
        ("SGBM", 9, 64),
        ("SGBM", 5, 96),
    ]

    records, disps = [], {}
    for algo, bsize, ndisp in configs:
        disp, runtime = compute_disparity(algo, bsize, ndisp, imgL, imgR)
        valid = disp > 0
        eval_mask = valid & nonocc
        err = np.abs(disp[eval_mask] - gtL[eval_mask])
        records.append({
            "Method": algo,
            "blockSize": bsize,
            "numDisparities": ndisp,
            "Valid Disparity (%)": round(100.0 * valid.mean(), 2),
            "Valid in Non-occ. (%)": round(100.0 * (valid & nonocc).sum() / nonocc.sum(), 2),
            "RMSE (px)": round(float(np.sqrt(np.mean(err**2))), 3),
            "Bad-Pixel >1px (%)": round(float(np.mean(err > 1.0) * 100.0), 2),
            "Bad-Pixel >2px (%)": round(float(np.mean(err > 2.0) * 100.0), 2),
            "Runtime (ms)": round(runtime, 2),
        })
        disps[(algo, bsize, ndisp)] = disp

    df = pd.DataFrame(records)
    df.to_csv("results/part_b/quantitative_part_b.csv", index=False)
    print(df.to_string(index=False))

    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("black")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.6))
    axes = axes.ravel()
    axes[0].imshow(np.ma.masked_where(gtL <= 0, gtL), cmap=cmap, vmin=0, vmax=VMAX)
    axes[0].set_title("Ground Truth (disp2 / 4)", fontweight="bold")
    for ax, rec in zip(axes[1:], records):
        d = disps[(rec["Method"], rec["blockSize"], rec["numDisparities"])]
        ax.imshow(np.ma.masked_where(d <= 0, d), cmap=cmap, vmin=0, vmax=VMAX)
        ax.set_title(f"{rec['Method']} (b={rec['blockSize']}, d={rec['numDisparities']})\n"
                     f"Valid {rec['Valid Disparity (%)']:.1f}% | Bad>2px {rec['Bad-Pixel >2px (%)']:.1f}%",
                     fontsize=10)
    for ax in axes:
        ax.axis("off")
    plt.suptitle("Part B: Teddy Disparity Maps - BM vs. SGBM (black = invalid)", fontsize=14, fontweight="bold")
    plt.tight_layout(h_pad=2.5)
    save_fig("disparity_comparison.png")

    best_bm = df[df.Method == "BM"].sort_values("Bad-Pixel >2px (%)").iloc[0]
    best_sgbm = df[df.Method == "SGBM"].sort_values("Bad-Pixel >2px (%)").iloc[0]
    key_bm = (best_bm.Method, int(best_bm.blockSize), int(best_bm.numDisparities))
    key_sgbm = (best_sgbm.Method, int(best_sgbm.blockSize), int(best_sgbm.numDisparities))
    print(f"Best BM: {key_bm}, best SGBM: {key_sgbm}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].imshow(nonocc, cmap="gray")
    axes[0].set_title("Non-occluded mask (L-R GT check)", fontweight="bold")
    for ax, key in zip(axes[1:], [key_bm, key_sgbm]):
        d = disps[key]
        vis = np.zeros((h, w, 3), np.float32)
        vis[nonocc] = 0.85
        vis[(d <= 0) & nonocc] = [0.2, 0.4, 1.0]
        bad = (d > 0) & nonocc & (np.abs(d - gtL) > 2.0)
        vis[bad] = [1.0, 0.1, 0.1]
        ax.imshow(vis)
        ax.set_title(f"{key[0]} (b={key[1]}, d={key[2]}): red = error >2px, blue = invalid", fontsize=10)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    save_fig("disparity_error_maps.png")

    best_disp = disps[key_sgbm]
    pts, cols = reproject(best_disp, imgL)
    write_ply("results/part_b/reconstructed_cloud.ply", pts, cols)
    gt_pts, gt_cols = reproject(gtL, imgL)
    print(f"Point cloud: {len(pts)} points written to results/part_b/reconstructed_cloud.ply")

    limits = [(float(gt_pts[:, i].min()), float(gt_pts[:, i].max())) for i in range(3)]
    fig = plt.figure(figsize=(18, 5.8))
    plot_cloud(fig.add_subplot(1, 3, 1, projection="3d"), pts, cols, 15, -70,
               f"SGBM (b={key_sgbm[1]}) - Oblique View", limits)
    plot_cloud(fig.add_subplot(1, 3, 2, projection="3d"), pts, cols, 89, -90,
               f"SGBM (b={key_sgbm[1]}) - Top-Down Depth Profile", limits)
    plot_cloud(fig.add_subplot(1, 3, 3, projection="3d"), gt_pts, gt_cols, 15, -70,
               "Ground-Truth Disparity - Oblique View", limits)
    plt.tight_layout()
    save_fig("point_cloud_views.png")


if __name__ == "__main__":
    run_part_b()
