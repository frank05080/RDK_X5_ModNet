import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from hbm_runtime import HB_HBMRuntime

# -----------------------------
# HB Runtime 模型路径 & 背景
# -----------------------------
MODEL_PATH = "modnet_output.bin"
REF_SIZE = 512

hb_dtype_map = {
    "U8": np.uint8,
    "S8": np.int8,
    "F32": np.float32,
    "F16": np.float16,
    "U16": np.uint16,
    "S16": np.int16,
    "S32": np.int32,
    "U32": np.uint32,
    "BOOL8": np.bool_,
}

# -----------------------------
# 初始化 HB 模型
# -----------------------------
model = HB_HBMRuntime(MODEL_PATH)
model_name = model.model_names[0]
input_names = model.input_names[model_name]
input_shapes = model.input_shapes[model_name]
input_dtypes = model.input_dtypes[model_name]

# -----------------------------
# resize + pad
# -----------------------------
def resize_with_padding(im, target_size=REF_SIZE):
    orig_h, orig_w, _ = im.shape
    scale = target_size / max(orig_h, orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    im_resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = target_size - new_w
    pad_h = target_size - new_h

    pad_left = pad_w // 2
    pad_top = pad_h // 2

    im_padded = cv2.copyMakeBorder(
        im_resized,
        pad_top,
        pad_h - pad_top,
        pad_left,
        pad_w - pad_left,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    return im_padded, scale, pad_left, pad_top, orig_w, orig_h, new_w, new_h

# -----------------------------
# MODNet inference
# -----------------------------
def modnet_infer(image_path):
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_norm = (im - 127.5) / 127.5

    im_pad, scale, pad_x, pad_y, orig_w, orig_h, new_w, new_h = resize_with_padding(im_norm, REF_SIZE)
    inp = np.transpose(im_pad, (2,0,1))[None].astype(np.float32)

    input_tensors = {}
    for name in input_names:
        np_dtype = hb_dtype_map.get(input_dtypes[name].name, np.float32)
        input_tensors[name] = inp.astype(np_dtype)

    # 设置调度参数
    priority = {model_name: 5}
    bpu_cores = {model_name: [0]}
    model.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)

    # 推理
    results = model.run(input_tensors)
    matte = np.squeeze(results[model_name]['output'])
    matte = np.clip(matte, 0, 1)

    # 去 padding
    matte_unpad = matte[pad_y:pad_y+new_h, pad_x:pad_x+new_w]
    # resize 回原图大小
    matte_final = cv2.resize(matte_unpad, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return matte_final

# -----------------------------
# 合成
# -----------------------------
def combine_foreground_bg(image_path, bg_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    bg = Image.open(bg_path).convert("RGB")
    bg = bg.resize((image_np.shape[1], image_np.shape[0]))
    bg_np = np.array(bg)

    # HB 推理得到 alpha matte
    matte_final = modnet_infer(image_path)
    matte_np = matte_final[:, :, None]
    matte_np = np.repeat(matte_np, 3, axis=2)

    foreground = image_np * matte_np
    background = bg_np * (1 - matte_np)
    output = (foreground + background).astype(np.uint8)
    return image, bg, Image.fromarray(output)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("🎄 Christmas MODNet Demo (HB Runtime)")
        self.root.geometry("1000x520")
        self.root.configure(bg="#f2f2f2")

        self.image_path = ""
        self.bg_path = ""
        self.original_img = None
        self.bg_img = None
        self.combined_img = None

        self.default_dir = os.path.dirname(os.path.abspath(__file__))

        # -----------------------------
        # Title
        # -----------------------------
        title = tk.Label(
            root,
            text="MODNet 人像抠图 & 背景替换 Demo",
            font=("Arial", 18, "bold"),
            bg="#f2f2f2"
        )
        title.pack(pady=10)

        # -----------------------------
        # Image Area
        # -----------------------------
        img_frame = tk.Frame(root, bg="#f2f2f2")
        img_frame.pack(pady=10)

        self.canvas = tk.Canvas(
            img_frame,
            width=900,
            height=300,
            bg="white",
            highlightthickness=1,
            highlightbackground="#cccccc"
        )
        self.canvas.pack()

        # -----------------------------
        # Button Area
        # -----------------------------
        btn_frame = tk.Frame(root, bg="#f2f2f2")
        btn_frame.pack(pady=15)

        self.btn_image = tk.Button(
            btn_frame, text="📷 选择原图",
            width=15, command=self.select_image
        )
        self.btn_image.grid(row=0, column=0, padx=10)

        self.btn_bg = tk.Button(
            btn_frame, text="🖼️ 选择背景",
            width=15, command=self.select_bg
        )
        self.btn_bg.grid(row=0, column=1, padx=10)

        self.btn_combine = tk.Button(
            btn_frame, text="✨ 开始合成",
            width=15, command=self.combine
        )
        self.btn_combine.grid(row=0, column=2, padx=10)

        self.btn_save = tk.Button(
            btn_frame, text="💾 保存结果",
            width=15, command=self.save
        )
        self.btn_save.grid(row=0, column=3, padx=10)

    # -----------------------------
    # 即时预览函数
    # -----------------------------
    def update_preview(self):
        self.canvas.delete("all")
        w, h = 300, 300

        if self.original_img:
            orig_tk = ImageTk.PhotoImage(self.original_img.resize((w, h)))
            self.canvas.orig_tk = orig_tk
            self.canvas.create_image(0, 0, anchor="nw", image=orig_tk)

        if self.bg_img:
            bg_tk = ImageTk.PhotoImage(self.bg_img.resize((w, h)))
            self.canvas.bg_tk = bg_tk
            self.canvas.create_image(300, 0, anchor="nw", image=bg_tk)

        if self.combined_img:
            comb_tk = ImageTk.PhotoImage(self.combined_img.resize((w, h)))
            self.canvas.comb_tk = comb_tk
            self.canvas.create_image(600, 0, anchor="nw", image=comb_tk)

    # -----------------------------
    # Select image → 立刻显示
    # -----------------------------
    def select_image(self):
        path = filedialog.askopenfilename(
            initialdir=self.default_dir,
            title="选择原图",
            filetypes=[("Image Files", "*.jpg *.png")]
        )
        if path:
            self.image_path = path
            self.original_img = Image.open(path).convert("RGB")
            self.update_preview()

    def select_bg(self):
        path = filedialog.askopenfilename(
            initialdir=self.default_dir,
            title="选择背景",
            filetypes=[("Image Files", "*.jpg *.png")]
        )
        if path:
            self.bg_path = path
            self.bg_img = Image.open(path).convert("RGB")
            self.update_preview()

    # -----------------------------
    # 合成
    # -----------------------------
    def combine(self):
        if not self.image_path or not self.bg_path:
            messagebox.showerror("错误", "请先选择原图和背景")
            return

        self.original_img, self.bg_img, self.combined_img = combine_foreground_bg(
            self.image_path, self.bg_path
        )
        self.update_preview()

    # -----------------------------
    # 保存
    # -----------------------------
    def save(self):
        if self.combined_img is None:
            messagebox.showerror("错误", "请先合成图片")
            return

        save_path = filedialog.asksaveasfilename(
            initialdir=self.default_dir,
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png")]
        )
        if save_path:
            self.combined_img.save(save_path)
            messagebox.showinfo("成功", "图片已保存")

# -----------------------------
# 启动 GUI
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

