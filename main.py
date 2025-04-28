import streamlit as st
import io
import base64
import itertools
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# ────────────────────────────────────────────────────────────────────
# Setup paths and imports
# ────────────────────────────────────────────────────────────────────
from SNN import SiameseNetwork

# ────────────────────────────────────────────────────────────────────
# Device and model loading
# ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path = "checkpoints/SNN_best.pth"
ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt.get("model_state", ckpt)
siamese = SiameseNetwork().to(device)
siamese.load_state_dict(state_dict)
siamese.eval()

yolo = YOLO("checkpoints/YOLO_best.pt")

MIN_SIZE = 4

# ────────────────────────────────────────────────────────────────────
# Preprocessing
# ────────────────────────────────────────────────────────────────────

class ResizeAndCenterPad:
    """
    1) Resize so the longest edge == max_size (preserve aspect ratio).
    2) Paste into the center of a new max_size×max_size canvas (fill with fill).
    """
    def __init__(self, max_size: int, fill: int = 255, interpolation=Image.BILINEAR):
        self.max_size = max_size
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        # 1) Resize longest side to max_size
        w, h = img.size
        if w >= h:
            new_w = self.max_size
            new_h = round(h * self.max_size / w)
        else:
            new_h = self.max_size
            new_w = round(w * self.max_size / h)
        img_resized = img.resize((new_w, new_h), self.interpolation)

        # 2) Center‐pad onto blank canvas
        canvas = Image.new(img.mode, (self.max_size, self.max_size), color=self.fill)
        left = (self.max_size - new_w) // 2
        top  = (self.max_size - new_h) // 2
        canvas.paste(img_resized, (left, top))
        return canvas

prep = transforms.Compose([
    transforms.Grayscale(),          # → still a PIL Image in “L” mode

    ResizeAndCenterPad(64, fill=255),  # white background
    transforms.ToTensor(),           # → [1×64×64], values in [0,1]
])

# ────────────────────────────────────────────────────────────────────
# Helper: PIL → base64 URI
# ────────────────────────────────────────────────────────────────────
def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# ────────────────────────────────────────────────────────────────────
# Inference: estimate probability
# ────────────────────────────────────────────────────────────────────
def calculate_probability(x: float) -> float:
    """
    Calculate the probability of a given value.
    :param x: The input value.
    :return: The calculated probability.
    """
    B_1 = -0.56652371  # From logistic regression results
    B_0 = 3.9185283
    return 1.0/(1.0+np.exp( -(B_0 + B_1 * x)))


# ────────────────────────────────────────────────────────────────────
# Inference: group by digit class, compare every combination
# ────────────────────────────────────────────────────────────────────
def infer_by_label(query_img: Image.Image, reference_img: Image.Image):
    det_q = yolo(query_img, device=device.type)[0]
    det_r = yolo(reference_img, device=device.type)[0]
    ann_q, ann_r = det_q.plot(), det_r.plot()

    boxes_q = det_q.boxes.xyxy.cpu().numpy()
    cls_q   = det_q.boxes.cls.cpu().numpy().astype(int)
    boxes_r = det_r.boxes.xyxy.cpu().numpy()
    cls_r   = det_r.boxes.cls.cpu().numpy().astype(int)

    def group(img, boxes, classes):
        out = defaultdict(list)
        for (x1, y1, x2, y2), c in zip(boxes, classes):
            crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
            if crop.width >= MIN_SIZE and crop.height >= MIN_SIZE:
                out[c].append((x1, crop))
        return out

    gq = group(query_img, boxes_q, cls_q)
    gr = group(reference_img, boxes_r, cls_r)

    comps = []  # (digit, dist, sim_pct, q_crop, r_crop)
    for digit in sorted(gq.keys() & gr.keys()):
        q_crops = [crop for (_, crop) in gq[digit]]
        r_crops = [crop for (_, crop) in gr[digit]]
        for q_crop, r_crop in itertools.product(q_crops, r_crops):
            tq = prep(q_crop).unsqueeze(0).to(device)
            tr = prep(r_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                eq, er = siamese(tq, tr)
            dist = F.pairwise_distance(eq, er, p=1).item()
            sim_pct = 100.0 * calculate_probability(dist) #* (1.0 / (1.0 + dist))
            comps.append((digit, dist, sim_pct, q_crop, r_crop))

    return ann_q, ann_r, comps

# ────────────────────────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Digit Comparison Table", layout="wide")
st.title("WriteMatch: Demo")

st.markdown("[Download sample images](https://github.com/jamesli124/WriteMatchDeploy/tree/main/sample_test_data)")

q_file = st.file_uploader("Upload First Image", type=["png","jpg","jpeg"])
r_file = st.file_uploader("Upload Second Image", type=["png","jpg","jpeg"])

if st.button("Compare Digits"):
    if not (q_file and r_file):
        st.warning("Please upload both images.")
    else:
        q_img = Image.open(q_file).convert("RGB")
        r_img = Image.open(r_file).convert("RGB")
        ann_q, ann_r, comps = infer_by_label(q_img, r_img)

        st.subheader("Annotated Images")
        c1, c2 = st.columns(2)
        with c1:
            st.image(ann_q, caption="First", use_container_width=True)
        with c2:
            st.image(ann_r, caption="Second", use_container_width=True)

        if not comps:
            st.info("No matching digits found.")
        else:
            # 1) compute overall average distance
            distances = [dist for (_, dist, _, _, _) in comps]
            avg_dist  = sum(distances) / len(distances)

            sim_prob = calculate_probability(avg_dist)

            # 2) decide match vs. non-match
            threshold = 0.69
            is_match  = sim_prob > threshold

            # 3) display the summary
            st.write(f"**Average distance:** {avg_dist:.2f}")

            if is_match:
                st.success(f"✅ This is more likely than not a match ({(sim_prob * 100):.1f}% match)")
            else:
                st.error(f"❌ This is likely NOT a match ({(sim_prob * 100):.1f}% match)")

            # 4) now build & show the per-digit table        
            md  = "| Digit | First Crop | Second Crop | Distance | Similarity % |\n"
            md += "|:-----:|:----------:|:-----------:|:--------:|:-------------:|\n"
            for digit, dist, sim, q_crop, r_crop in comps:
                q_uri = pil_to_base64(q_crop)
                r_uri = pil_to_base64(r_crop)
                md += (
                    f"| {digit} "
                    f"| <img src='{q_uri}' width='32'/> "
                    f"| <img src='{r_uri}' width='32'/> "
                    f"| {dist:.2f} "
                    f"| {sim:.1f}% |\n"
                )
            st.subheader("Comparison Table")
            st.markdown(md, unsafe_allow_html=True)
