import os
import numpy as np
import cv2
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astropy import units as u
from math import sin, cos, tan

# ---------------- CONFIG ----------------
OUT_DIR = "dataset_real"
IMG_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

MAG_CUTOFF = 7
NUM_IMAGES = 200
IMG_SIZE = 128
FOV_DEG = 15.0
MIN_STARS = 8

# ---------------------------------------
print("Querying Gaia catalog...")

query = f"""
SELECT TOP 20000 ra, dec, phot_g_mean_mag
FROM gaiadr3.gaia_source
WHERE phot_g_mean_mag < {MAG_CUTOFF}
"""

job = Gaia.launch_job(query)
table = job.get_results()

print("Total catalog rows:", len(table))

ra = np.array(table["ra"])
dec = np.array(table["dec"])
mag = np.array(table["phot_g_mean_mag"])

print("Stars loaded:", len(ra))
# ---------------------------------------
coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
star_vecs = coords.cartesian.xyz.value.T
star_mags = mag

# ---------------------------------------
def random_quaternion():
    u1, u2, u3 = np.random.rand(3)
    qx = np.sqrt(1-u1)*np.sin(2*np.pi*u2)
    qy = np.sqrt(1-u1)*np.cos(2*np.pi*u2)
    qz = np.sqrt(u1)*np.sin(2*np.pi*u3)
    qw = np.sqrt(u1)*np.cos(2*np.pi*u3)
    return np.array([qw, qx, qy, qz])

def quat_to_rot(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])

def project(points):
    f = (IMG_SIZE/2) / np.tan(np.deg2rad(FOV_DEG/2))
    xs = f * (points[:,0]/points[:,2]) + IMG_SIZE/2
    ys = f * (points[:,1]/points[:,2]) + IMG_SIZE/2
    return xs, ys

def mag_to_intensity(m):
    m = np.clip(m, -1, MAG_CUTOFF)
    norm = (MAG_CUTOFF - m) / (MAG_CUTOFF + 1)
    return int(150 + 105*norm)

labels = []
count = 0

while count < NUM_IMAGES:
    q = random_quaternion()
    R = quat_to_rot(q)

    cam_vecs = (R @ star_vecs.T).T

    mask = cam_vecs[:,2] > 0
    cam_vecs = cam_vecs[mask]
    mags = star_mags[mask]

    xs, ys = project(cam_vecs)

    valid = (xs>=0)&(xs<IMG_SIZE)&(ys>=0)&(ys<IMG_SIZE)
    xs = xs[valid]
    ys = ys[valid]
    mags = mags[valid]

    if len(xs) < MIN_STARS:
        continue

    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    for x,y,m in zip(xs,ys,mags):
        ix, iy = int(x), int(y)
        cv2.circle(img, (ix,iy), 1, mag_to_intensity(m), -1)

    noise = np.random.normal(0, 1.0, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(IMG_DIR, f"img_{count:04d}.png"), img)
    labels.append(q)

    count += 1
    if count % 25 == 0:
        print("Generated", count)

labels = np.array(labels)
np.savetxt(os.path.join(OUT_DIR, "labels.csv"), labels, delimiter=",")

print("DONE.")
