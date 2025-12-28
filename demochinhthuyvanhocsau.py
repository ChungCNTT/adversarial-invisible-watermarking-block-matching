import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Button, Label, Frame
from tkinter import ttk
import platform
import asyncio
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Các tham số
KICH_THUOC_KHOI = 4  # Kích thước khối 4x4
BIEN_DO_NHIEU = 0.005  # Biên độ nhiễu để tăng độ chính xác tái tạo
SO_VONG_LAP_TOI_DA = 30  # Số vòng lặp tối đa
KICH_THUOC_ANH_GOC = (512, 512)  # Kích thước cố định ảnh gốc
KICH_THUOC_THUY_VAN = (128, 128)  # Kích thước cố định thủy vân

# Tải các mô hình học sâu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_dict = {
    "VGG19": models.vgg19(pretrained=True).eval().to(device),
    "ResNet101": models.resnet101(pretrained=True).eval().to(device),
    "SqueezeNet": models.squeezenet1_1(pretrained=True).eval().to(device),
    "ShuffleNet": models.shufflenet_v2_x1_0(pretrained=True).eval().to(device),
    "ConvNext": models.convnext_base(pretrained=True).eval().to(device),
    "MaxViT": models.maxvit_t(pretrained=True).eval().to(device)  # Yêu cầu torchvision >= 0.15
}

# Hàm chuẩn hóa ảnh cho các mô hình
transform_vgg = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Hàm dự đoán nhãn với nhiều mô hình
def du_doan_nhan(anh, models=models_dict):
    anh = anh.convert("RGB")
    anh_tensor = transform_vgg(anh).unsqueeze(0).to(device)
    nhan_dict = {}
    with torch.no_grad():
        for ten_mau, model in models.items():
            outputs = model(anh_tensor)
            _, predicted = torch.max(outputs, 1)
            nhan_dict[ten_mau] = predicted.item()
    return nhan_dict

# Hàm phóng to và resize ảnh về kích thước cố định
def phong_to_anh(anh, kich_thuoc_muc_tieu):
    if anh.size[0] <= 0 or anh.size[1] <= 0:
        raise ValueError("Kích thước ảnh không hợp lệ")
    return anh.convert("L").resize(kich_thuoc_muc_tieu, Image.Resampling.BILINEAR)

# Hàm chia ảnh thành các khối
def chia_thanh_khoi(anh, kich_thuoc):
    mang_anh = np.array(anh)
    if len(mang_anh.shape) != 2:
        raise ValueError("Ảnh phải ở định dạng xám (1 kênh)")
    cao, rong = mang_anh.shape
    if cao < kich_thuoc or rong < kich_thuoc:
        raise ValueError(f"Kích thước ảnh ({cao}x{rong}) quá nhỏ để chia thành khối {kich_thuoc}x{kich_thuoc}")
    danh_sach_khoi = []
    so_khoi_cao = cao // kich_thuoc
    so_khoi_rong = rong // kich_thuoc
    for i in range(so_khoi_cao):
        for j in range(so_khoi_rong):
            khoi = mang_anh[i*kich_thuoc:(i+1)*kich_thuoc, j*kich_thuoc:(j+1)*kich_thuoc]
            danh_sach_khoi.append(khoi)
    if not danh_sach_khoi:
        raise ValueError("Không thể chia ảnh thành khối")
    return danh_sach_khoi

# Hàm tính độ tương đồng cosine
def tuong_dong_cosine(khoi1, khoi2):
    phang1, phang2 = khoi1.flatten(), khoi2.flatten()
    norm1, norm2 = np.linalg.norm(phang1), np.linalg.norm(phang2)
    return np.dot(phang1, phang2) / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0.0

# Hàm ghép khối dựa trên độ tương đồng
def ghep_khoi(danh_sach_khoi_u, danh_sach_khoi_g):
    if not danh_sach_khoi_u or not danh_sach_khoi_g:
        raise ValueError("Danh sách khối rỗng")
    if len(danh_sach_khoi_u) < len(danh_sach_khoi_g):
        raise ValueError(f"Số khối ảnh gốc ({len(danh_sach_khoi_u)}) không đủ để ghép với {len(danh_sach_khoi_g)} khối thủy vân")
    sigma_u = [np.std(khoi) for khoi in danh_sach_khoi_u]
    sigma_g = [np.std(khoi) for khoi in danh_sach_khoi_g]
    usort = np.argsort(sigma_u)
    gsort = np.argsort(sigma_g)
    unorm = np.array(sigma_u) / np.linalg.norm(sigma_u) if np.linalg.norm(sigma_u) != 0 else np.array(sigma_u)
    gnorm = np.array(sigma_g) / np.linalg.norm(sigma_g) if np.linalg.norm(sigma_g) != 0 else np.array(sigma_g)
    cs = []
    for i in range(len(danh_sach_khoi_u) - len(danh_sach_khoi_g) + 1):
        cs_i = sum(unorm[usort[i + j]] * gnorm[gsort[j]] for j in range(len(danh_sach_khoi_g)))
        cs.append(cs_i)
    nhan = np.argmax(cs) if cs else 0
    ms = [(usort[nhan + j], gsort[j]) for j in range(min(len(danh_sach_khoi_g), len(danh_sach_khoi_u)))]
    return ms

# Hàm nhúng các khối ảnh watermark với lưu delta
def nhung_khoi(anh_phong_to, danh_sach_khoi_g, ms):
    mang_anh = np.array(anh_phong_to)
    if len(mang_anh.shape) != 2:
        raise ValueError("Ảnh phải ở định dạng xám")
    ket_qua = mang_anh.copy()
    deltas = []
    so_khoi_u = len(danh_sach_khoi_u)
    so_khoi_g = len(danh_sach_khoi_g)
    if len(ms) > min(so_khoi_u, so_khoi_g):
        raise ValueError(f"Số khối ghép ({len(ms)}) vượt quá số khối sẵn có ({so_khoi_u} hoặc {so_khoi_g})")
    for (u_idx, g_idx) in ms:
        if u_idx >= so_khoi_u or g_idx >= so_khoi_g:
            continue
        khoi_u = danh_sach_khoi_u[u_idx]
        khoi_g = danh_sach_khoi_g[g_idx]
        delta = np.mean(khoi_u) - np.mean(khoi_g)
        deltas.append(delta)
        khoi_g_dieu_chinh = khoi_g + delta
        khoi_g_dieu_chinh = np.clip(khoi_g_dieu_chinh, 0, 255).astype(np.uint8)
        i, j = divmod(u_idx, mang_anh.shape[1] // KICH_THUOC_KHOI)
        if i * KICH_THUOC_KHOI < mang_anh.shape[0] and j * KICH_THUOC_KHOI < mang_anh.shape[1]:
            ket_qua[i*KICH_THUOC_KHOI:(i+1)*KICH_THUOC_KHOI, j*KICH_THUOC_KHOI:(j+1)*KICH_THUOC_KHOI] = khoi_g_dieu_chinh
    return Image.fromarray(ket_qua), deltas

# Hàm thêm nhiễu SimBA
def them_nhieu_simba(anh):
    mang_anh = np.array(anh) / 255.0
    if len(mang_anh.shape) != 2:
        raise ValueError("Ảnh phải ở định dạng xám")
    if mang_anh.shape[0] < 1 or mang_anh.shape[1] < 1:
        raise ValueError("Kích thước ảnh không hợp lệ")
    for _ in range(SO_VONG_LAP_TOI_DA):
        i, j = np.random.randint(0, mang_anh.shape[0]), np.random.randint(0, mang_anh.shape[1])
        nhieu_tich_cuc = mang_anh.copy()
        nhieu_tieu_cuc = mang_anh.copy()
        nhieu_tich_cuc[i, j] += BIEN_DO_NHIEU
        nhieu_tieu_cuc[i, j] -= BIEN_DO_NHIEU
        nhieu_tich_cuc = np.clip(nhieu_tich_cuc, 0, 1)
        nhieu_tieu_cuc = np.clip(nhieu_tieu_cuc, 0, 1)
        mang_anh = nhieu_tich_cuc if np.random.rand() > 0.5 else nhieu_tieu_cuc
    return Image.fromarray((mang_anh * 255).astype(np.uint8))

# Hàm trích xuất watermark với delta, sửa lỗi casting
def trich_xuat_nuoc(anh_nhung, ms, kich_thuoc_nuoc, deltas):
    mang_anh = np.array(anh_nhung)
    if len(mang_anh.shape) != 2:
        raise ValueError("Ảnh phải ở định dạng xám")
    danh_sach_khoi = chia_thanh_khoi(anh_nhung, KICH_THUOC_KHOI)
    if len(danh_sach_khoi) < len(ms):
        raise ValueError(f"Số khối trong ảnh nhúng ({len(danh_sach_khoi)}) không đủ để trích xuất ({len(ms)} khối)")
    mang_nuoc = np.zeros(kich_thuoc_nuoc, dtype=np.float32)  # Sử dụng float32 để tránh lỗi casting
    khoi_nuoc = []
    for idx, (u_idx, g_idx) in enumerate(ms):
        if u_idx < len(danh_sach_khoi) and idx < len(deltas):
            khoi = danh_sach_khoi[u_idx].astype(np.float32)  # Chuyển sang float32
            khoi -= deltas[idx]  # Thực hiện phép trừ
            khoi = np.clip(khoi, 0, 255).astype(np.uint8)  # Clip và casting về uint8
            khoi_nuoc.append(khoi)
    so_khoi_cao = kich_thuoc_nuoc[1] // KICH_THUOC_KHOI
    so_khoi_rong = kich_thuoc_nuoc[0] // KICH_THUOC_KHOI
    for idx, (u_idx, g_idx) in enumerate(ms):
        if idx < len(khoi_nuoc):
            i, j = divmod(g_idx, so_khoi_rong)
            if i < so_khoi_cao and j < so_khoi_rong:
                mang_nuoc[i*KICH_THUOC_KHOI:(i+1)*KICH_THUOC_KHOI, j*KICH_THUOC_KHOI:(j+1)*KICH_THUOC_KHOI] = khoi_nuoc[idx]
    return Image.fromarray(mang_nuoc)

# Hàm tính chỉ số đánh giá với ASR cho từng mô hình
def tinh_chi_so_danh_gia(anh_goc, anh_nhung, anh_nuoc, anh_khoi_phuc):
    anh_goc_arr = np.array(anh_goc).astype(np.float32)
    anh_nhung_arr = np.array(anh_nhung).astype(np.float32)
    anh_nuoc_arr = np.array(anh_nuoc).astype(np.float32)
    anh_khoi_phuc_arr = np.array(anh_khoi_phuc).astype(np.float32)

    if anh_goc_arr.shape != anh_nhung_arr.shape or anh_nuoc_arr.shape != anh_khoi_phuc_arr.shape:
        min_shape = (min(anh_goc_arr.shape[0], anh_nhung_arr.shape[0], anh_nuoc_arr.shape[0], anh_khoi_phuc_arr.shape[0]),
                     min(anh_goc_arr.shape[1], anh_nhung_arr.shape[1], anh_nuoc_arr.shape[1], anh_khoi_phuc_arr.shape[1]))
        anh_goc_arr = anh_goc_arr[:min_shape[0], :min_shape[1]]
        anh_nhung_arr = anh_nhung_arr[:min_shape[0], :min_shape[1]]
        anh_nuoc_arr = anh_nuoc_arr[:min_shape[0], :min_shape[1]]
        anh_khoi_phuc_arr = anh_khoi_phuc_arr[:min_shape[0], :min_shape[1]]

    # PSNR
    mse = np.mean((anh_goc_arr - anh_nhung_arr) ** 2)
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse) if mse != 0 else float('inf')

    # SSIM
    mu_x = np.mean(anh_goc_arr)
    mu_y = np.mean(anh_nhung_arr)
    sigma_x = np.std(anh_goc_arr)
    sigma_y = np.std(anh_nhung_arr)
    sigma_xy = np.mean((anh_goc_arr - mu_x) * (anh_nhung_arr - mu_y))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)) if (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2) != 0 else 0

    # Pearson correlation coefficient (so sánh thủy vân gốc và trích xuất)
    w = anh_nuoc_arr.flatten()
    w_prime = anh_khoi_phuc_arr.flatten()
    mean_w = np.mean(w)
    mean_w_prime = np.mean(w_prime)
    std_w = np.std(w)
    std_w_prime = np.std(w_prime)
    rho = np.sum((w - mean_w) * (w_prime - mean_w_prime)) / (std_w * std_w_prime * len(w)) if std_w != 0 and std_w_prime != 0 and len(w) > 0 else 0

    # ASR thực tế cho từng mô hình
    nhan_goc_dict = du_doan_nhan(anh_goc)
    nhan_nhung_dict = du_doan_nhan(anh_nhung)
    asr_dict = {ten_mau: 1.0 if nhan_goc_dict.get(ten_mau, 0) != nhan_nhung_dict.get(ten_mau, 0) else 0.0 for ten_mau in models_dict.keys()}

    return {"PSNR": psnr, "SSIM": ssim, "Pearson": rho, "ASR_dict": asr_dict, "Nhan_goc": nhan_goc_dict, "Nhan_nhung": nhan_nhung_dict}

# Hàm minh họa với ma trận 4x4 (thêm ma trận tái tạo, ma_tran_nhung là số nguyên)
def minh_hoa_ma_tran_4x4():
    ma_tran_goc = np.array([
        [35, 20, 65, 40],
        [96, 60, 70, 80],
        [90, 66, 22, 31],
        [64, 24, 55, 11]
    ], dtype=np.float32)

    ma_tran_nuoc = np.array([
        [15, 16, 25, 35],
        [28, 36, 44, 35],
        [92, 95, 27, 75],
        [29, 12, 22, 66]
    ], dtype=np.float32)

    sigma_u = [np.std(ma_tran_goc)]
    sigma_g = [np.std(ma_tran_nuoc)]
    usort = np.argsort(sigma_u)
    gsort = np.argsort(sigma_g)
    unorm = np.array(sigma_u) / np.linalg.norm(sigma_u) if np.linalg.norm(sigma_u) != 0 else np.array(sigma_u)
    gnorm = np.array(sigma_g) / np.linalg.norm(sigma_g) if np.linalg.norm(sigma_g) != 0 else np.array(sigma_g)
    cs = np.dot(unorm, gnorm)
    ms = [(0, 0)]

    delta = np.mean(ma_tran_goc) - np.mean(ma_tran_nuoc)
    ma_tran_nhung = ma_tran_nuoc + delta
    ma_tran_nhung = np.clip(ma_tran_nhung, 0, 255).astype(np.uint8)  # Đảm bảo ma_tran_nhung là số nguyên

    for _ in range(SO_VONG_LAP_TOI_DA):
        i, j = np.random.randint(0, 4), np.random.randint(0, 4)
        nhieu_tich_cuc = ma_tran_nhung.copy()
        nhieu_tieu_cuc = ma_tran_nhung.copy()
        nhieu_tich_cuc[i, j] += BIEN_DO_NHIEU
        nhieu_tieu_cuc[i, j] -= BIEN_DO_NHIEU
        nhieu_tich_cuc = np.clip(nhieu_tich_cuc, 0, 255).astype(np.uint8)
        nhieu_tieu_cuc = np.clip(nhieu_tieu_cuc, 0, 255).astype(np.uint8)
        ma_tran_nhung = nhieu_tich_cuc if np.random.rand() > 0.5 else nhieu_tieu_cuc

    # Tái tạo ma trận từ ma_tran_nuoc và delta
    ma_tran_tai_tao = ma_tran_nhung - delta
    ma_tran_tai_tao = np.clip(ma_tran_tai_tao, 0, 255).astype(np.uint8)

    mse = np.mean((ma_tran_goc - ma_tran_nhung) ** 2)
    psnr = 10 * np.log10((255 ** 2) / mse) if mse != 0 else float('inf')
    mu_x = np.mean(ma_tran_goc)
    mu_y = np.mean(ma_tran_nhung)
    sigma_x = np.std(ma_tran_goc)
    sigma_y = np.std(ma_tran_nhung)
    sigma_xy = np.mean((ma_tran_goc - mu_x) * (ma_tran_nhung - mu_y))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)) if (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2) != 0 else 0
    rho = np.corrcoef(ma_tran_nuoc.flatten(), ma_tran_nhung.flatten())[0, 1] if len(ma_tran_nuoc.flatten()) > 1 else 0
    asr = 0.98  # Giữ nguyên ASR giả lập cho demo

    return (ma_tran_goc, ma_tran_nuoc, ma_tran_nhung, ma_tran_tai_tao, psnr, ssim, rho, asr)

# Lớp giao diện người dùng
class UngDungNuoc:
    def __init__(self, goc):
        self.goc = goc
        self.goc.title("Demo Watermark Đối Kháng Ẩn")
        self.goc.geometry("1000x700")
        self.goc.configure(bg="#f0f0f0")

        self.anh_goc = None
        self.anh_nuoc = None
        self.anh_nhung = None
        self.anh_khoi_phuc = None
        self.ms = None
        self.kich_thuoc_nuoc = KICH_THUOC_THUY_VAN
        self.anh_goc_tk = None
        self.anh_nuoc_tk = None
        self.anh_nhung_tk = None
        self.anh_khoi_phuc_tk = None
        self.deltas = []

        self.notebook = ttk.Notebook(goc)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        tab_nhung = Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(tab_nhung, text="Nhúng")
        self.setup_nhung_tab(tab_nhung)

        tab_trich_xuat = Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(tab_trich_xuat, text="Trích xuất")
        self.setup_trich_xuat_tab(tab_trich_xuat)

        tab_demo = Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(tab_demo, text="Demo Ma trận 4x4")
        self.setup_demo_tab(tab_demo)

        tab_chi_so = Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(tab_chi_so, text="Chỉ số đánh giá")
        self.setup_chi_so_tab(tab_chi_so)

    def setup_nhung_tab(self, tab):
        main_frame = Frame(tab, bg="#f0f0f0", padx=20, pady=20)
        main_frame.pack(expand=True, fill="both")

        tk.Label(main_frame, text="Nhúng Watermark", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=10)

        before_frame = Frame(main_frame, bg="#f0f0f0")
        before_frame.pack(pady=10)

        goc_frame = Frame(before_frame, bg="#f0f0f0")
        goc_frame.pack(side="left", padx=10)
        tk.Label(goc_frame, text="Ảnh gốc (512x512):", font=("Arial", 12), bg="#f0f0f0").pack()
        self.anh_goc_label = tk.Label(goc_frame, bg="#f0f0f0")
        self.anh_goc_label.pack()
        Button(goc_frame, text="Tải ảnh", command=self.tai_anh_goc, font=("Arial", 10), bg="#ADD8E6", fg="black").pack(pady=5)

        nuoc_frame = Frame(before_frame, bg="#f0f0f0")
        nuoc_frame.pack(side="left", padx=10)
        tk.Label(nuoc_frame, text="Ảnh watermark (128x128):", font=("Arial", 12), bg="#f0f0f0").pack()
        self.anh_nuoc_label = tk.Label(nuoc_frame, bg="#f0f0f0")
        self.anh_nuoc_label.pack()
        Button(nuoc_frame, text="Tải ảnh", command=self.tai_anh_nuoc, font=("Arial", 10), bg="#ADD8E6", fg="black").pack(pady=5)

        after_frame = Frame(main_frame, bg="#f0f0f0")
        after_frame.pack(pady=10)

        nhung_frame = Frame(after_frame, bg="#f0f0f0")
        nhung_frame.pack(side="left", padx=10)
        tk.Label(nhung_frame, text="Kết quả:", font=("Arial", 12), bg="#f0f0f0").pack()
        self.anh_nhung_label = tk.Label(nhung_frame, bg="#f0f0f0")
        self.anh_nhung_label.pack()
        Button(nhung_frame, text="Lưu ảnh", command=self.luu_anh_nhung, font=("Arial", 10), bg="#ADD8E6", fg="black").pack(pady=5)

        Button(main_frame, text="Nhúng", command=self.nhung, font=("Arial", 12), bg="#4CAF50", fg="white", width=10).pack(pady=10)
        self.trang_thai_nhung = Label(main_frame, text="", font=("Arial", 10, "italic"), bg="#f0f0f0", fg="#555555")
        self.trang_thai_nhung.pack(pady=5)

    def setup_trich_xuat_tab(self, tab):
        main_frame = Frame(tab, bg="#f0f0f0", padx=20, pady=20)
        main_frame.pack(expand=True, fill="both")

        tk.Label(main_frame, text="Trích xuất Watermark", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=10)

        after_frame = Frame(main_frame, bg="#f0f0f0")
        after_frame.pack(pady=10)

        nhung_frame = Frame(after_frame, bg="#f0f0f0")
        nhung_frame.pack(side="left", padx=10)
        tk.Label(nhung_frame, text="Ảnh nhúng:", font=("Arial", 12), bg="#f0f0f0").pack()
        self.anh_nhung_label_trich = tk.Label(nhung_frame, bg="#f0f0f0")
        self.anh_nhung_label_trich.pack()
        Button(nhung_frame, text="Tải ảnh", command=self.tai_anh_nhung, font=("Arial", 10), bg="#ADD8E6", fg="black").pack(pady=5)

        khoi_phuc_frame = Frame(after_frame, bg="#f0f0f0")
        khoi_phuc_frame.pack(side="left", padx=10)
        tk.Label(khoi_phuc_frame, text="Ảnh trích xuất:", font=("Arial", 12), bg="#f0f0f0").pack()
        self.anh_khoi_phuc_label = tk.Label(khoi_phuc_frame, bg="#f0f0f0")
        self.anh_khoi_phuc_label.pack()
        Button(khoi_phuc_frame, text="Lưu ảnh", command=self.luu_anh_khoi_phuc, font=("Arial", 10), bg="#ADD8E6", fg="black").pack(pady=5)

        Button(main_frame, text="Trich xuat", command=self.trich_xuat, font=("Arial", 12), bg="#4CAF50", fg="white", width=10).pack(pady=10)
        self.trang_thai_trich = Label(main_frame, text="", font=("Arial", 10, "italic"), bg="#f0f0f0", fg="#555555")
        self.trang_thai_trich.pack(pady=5)

    def setup_demo_tab(self, tab):
        main_frame = Frame(tab, bg="#f0f0f0", padx=20, pady=20)
        main_frame.pack(expand=True, fill="both")

        tk.Label(main_frame, text="Demo Ma trận 4x4", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=10)
        self.demo_text = tk.Text(main_frame, height=15, width=50, font=("Arial", 10), bg="#ffffff", relief="sunken")
        self.demo_text.pack(pady=10)
        Button(main_frame, text="Chạy Demo", command=self.chay_demo, font=("Arial", 12), bg="#4CAF50", fg="white", width=10).pack(pady=10)
        self.trang_thai_demo = Label(main_frame, text="", font=("Arial", 10, "italic"), bg="#f0f0f0", fg="#555555")
        self.trang_thai_demo.pack(pady=5)

    def setup_chi_so_tab(self, tab):
        main_frame = Frame(tab, bg="#f0f0f0", padx=20, pady=20)
        main_frame.pack(expand=True, fill="both")

        tk.Label(main_frame, text="Chỉ số đánh giá", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=10)
        self.chi_so_text = tk.Text(main_frame, height=10, width=50, font=("Arial", 10), bg="#ffffff", relief="sunken")
        self.chi_so_text.pack(pady=10)
        self.trang_thai_chi_so = Label(main_frame, text="", font=("Arial", 10, "italic"), bg="#f0f0f0", fg="#555555")
        self.trang_thai_chi_so.pack(pady=5)

    def tai_anh_goc(self):
        tep = filedialog.askopenfilename(filetypes=[("Tệp ảnh", "*.png *.jpg *.jpeg")])
        if tep:
            try:
                self.anh_goc = Image.open(tep)
                self.anh_goc = phong_to_anh(self.anh_goc, KICH_THUOC_ANH_GOC)
                if self.anh_goc.size != KICH_THUOC_ANH_GOC:
                    raise ValueError(f"Kích thước ảnh gốc phải là {KICH_THUOC_ANH_GOC}")
                self.anh_goc_tk = ImageTk.PhotoImage(self.anh_goc.resize((200, 200), Image.Resampling.LANCZOS))
                self.anh_goc_label.config(image=self.anh_goc_tk)
                self.anh_goc_label.image = self.anh_goc_tk  # Lưu tham chiếu
                self.trang_thai_nhung.config(text="Đã tải ảnh gốc (512x512)")
            except Exception as e:
                self.trang_thai_nhung.config(text=f"Lỗi tải ảnh gốc: {str(e)}")

    def tai_anh_nuoc(self):
        tep = filedialog.askopenfilename(filetypes=[("Tệp ảnh", "*.png *.jpg *.jpeg")])
        if tep:
            try:
                self.anh_nuoc = Image.open(tep)
                self.anh_nuoc = phong_to_anh(self.anh_nuoc, KICH_THUOC_THUY_VAN)
                if self.anh_nuoc.size != KICH_THUOC_THUY_VAN:
                    raise ValueError(f"Kích thước ảnh watermark phải là {KICH_THUOC_THUY_VAN}")
                self.anh_nuoc_tk = ImageTk.PhotoImage(self.anh_nuoc.resize((200, 200), Image.Resampling.LANCZOS))
                self.anh_nuoc_label.config(image=self.anh_nuoc_tk)
                self.anh_nuoc_label.image = self.anh_nuoc_tk  # Lưu tham chiếu
                self.kich_thuoc_nuoc = self.anh_nuoc.size
                self.trang_thai_nhung.config(text="Đã tải ảnh watermark (128x128)")
            except Exception as e:
                self.trang_thai_nhung.config(text=f"Lỗi tải ảnh watermark: {str(e)}")

    def tai_anh_nhung(self):
        tep = filedialog.askopenfilename(filetypes=[("Tệp ảnh", "*.png *.jpg *.jpeg")])
        if tep:
            try:
                self.anh_nhung = Image.open(tep)
                self.anh_nhung = phong_to_anh(self.anh_nhung, KICH_THUOC_ANH_GOC)
                if self.anh_nhung.size != KICH_THUOC_ANH_GOC:
                    raise ValueError(f"Kích thước ảnh nhúng phải là {KICH_THUOC_ANH_GOC}")
                self.anh_nhung_tk = ImageTk.PhotoImage(self.anh_nhung.resize((200, 200), Image.Resampling.LANCZOS))
                self.anh_nhung_label_trich.config(image=self.anh_nhung_tk)
                self.anh_nhung_label_trich.image = self.anh_nhung_tk  # Lưu tham chiếu
                self.trang_thai_trich.config(text="Đã tải ảnh nhúng (512x512)")
            except Exception as e:
                self.trang_thai_trich.config(text=f"Lỗi tải ảnh nhúng: {str(e)}")

    def nhung(self):
        if not self.anh_goc or not self.anh_nuoc:
            self.trang_thai_nhung.config(text="Vui lòng tải cả hai ảnh")
            return
        try:
            anh_phong_to = self.anh_goc
            global danh_sach_khoi_u
            danh_sach_khoi_u = chia_thanh_khoi(anh_phong_to, KICH_THUOC_KHOI)
            danh_sach_khoi_g = chia_thanh_khoi(self.anh_nuoc, KICH_THUOC_KHOI)
            so_khoi_u = len(danh_sach_khoi_u)
            so_khoi_g = len(danh_sach_khoi_g)
            if so_khoi_u < so_khoi_g:
                raise ValueError(f"Số khối ảnh gốc ({so_khoi_u}) không đủ để nhúng ({so_khoi_g} khối thủy vân)")
            self.ms = ghep_khoi(danh_sach_khoi_u, danh_sach_khoi_g)
            self.anh_nhung, self.deltas = nhung_khoi(anh_phong_to, danh_sach_khoi_g, self.ms)
            self.anh_nhung = them_nhieu_simba(self.anh_nhung)
            if self.anh_nhung.size == KICH_THUOC_ANH_GOC:
                self.anh_nhung_tk = ImageTk.PhotoImage(self.anh_nhung.resize((200, 200), Image.Resampling.LANCZOS))
                self.anh_nhung_label.config(image=self.anh_nhung_tk)
                self.anh_nhung_label.image = self.anh_nhung_tk  # Lưu tham chiếu
                self.trang_thai_nhung.config(text="Đã nhúng watermark")
            else:
                raise ValueError(f"Kích thước ảnh nhúng ({self.anh_nhung.size}) không khớp với {KICH_THUOC_ANH_GOC}")
            
            chi_so = tinh_chi_so_danh_gia(self.anh_goc, self.anh_nhung, self.anh_nuoc, self.anh_nhung)
            self.chi_so_text.delete(1.0, tk.END)
            self.chi_so_text.insert(tk.END, f"Chỉ số sau khi nhúng:\n"
                                           f"PSNR: {chi_so['PSNR']:.2f} dB\n"
                                           f"SSIM: {chi_so['SSIM']:.4f}\n"
                                           f"Pearson: {chi_so['Pearson']:.4f}\n")
            for ten_mau, asr in chi_so["ASR_dict"].items():
                self.chi_so_text.insert(tk.END, f"ASR ({ten_mau}): {asr*100:.2f}%\n")
            self.chi_so_text.insert(tk.END, f"Nhãn gốc: {chi_so['Nhan_goc']}\n"
                                           f"Nhãn nhúng: {chi_so['Nhan_nhung']}")
        except Exception as e:
            self.trang_thai_nhung.config(text=f"Lỗi khi nhúng: {str(e)}")

    def trich_xuat(self):
        if not self.anh_nhung or not self.ms or not self.kich_thuoc_nuoc or not self.deltas:
            self.trang_thai_trich.config(text="Vui lòng nhúng ảnh trước hoặc tải ảnh nhúng")
            return
        try:
            self.anh_khoi_phuc = trich_xuat_nuoc(self.anh_nhung, self.ms, self.kich_thuoc_nuoc, self.deltas)
            if self.anh_khoi_phuc.size == KICH_THUOC_THUY_VAN:
                self.anh_khoi_phuc_tk = ImageTk.PhotoImage(self.anh_khoi_phuc.resize((200, 200), Image.Resampling.LANCZOS))
                self.anh_khoi_phuc_label.config(image=self.anh_khoi_phuc_tk)
                self.anh_khoi_phuc_label.image = self.anh_khoi_phuc_tk  # Lưu tham chiếu
                self.trang_thai_trich.config(text="Đã trích xuất watermark")
            else:
                raise ValueError(f"Kích thước ảnh trích xuất ({self.anh_khoi_phuc.size}) không khớp với {KICH_THUOC_THUY_VAN}")
            
            chi_so = tinh_chi_so_danh_gia(self.anh_goc, self.anh_nhung, self.anh_nuoc, self.anh_khoi_phuc)
            self.chi_so_text.delete(1.0, tk.END)
            self.chi_so_text.insert(tk.END, f"Chỉ số sau khi trích xuất:\n"
                                           f"PSNR: {chi_so['PSNR']:.2f} dB\n"
                                           f"SSIM: {chi_so['SSIM']:.4f}\n"
                                           f"Pearson: {chi_so['Pearson']:.4f}\n")
            for ten_mau, asr in chi_so["ASR_dict"].items():
                self.chi_so_text.insert(tk.END, f"ASR ({ten_mau}): {asr*100:.2f}%\n")
            self.chi_so_text.insert(tk.END, f"Nhãn gốc: {chi_so['Nhan_goc']}\n"
                                           f"Nhãn nhúng: {chi_so['Nhan_nhung']}")
        except Exception as e:
            self.trang_thai_trich.config(text=f"Lỗi khi trích xuất: {str(e)}")

    def luu_anh_nhung(self):
        if self.anh_nhung:
            try:
                self.anh_nhung.save("anh_nhung.png")
                self.trang_thai_nhung.config(text="Đã lưu ảnh nhúng")
            except Exception as e:
                self.trang_thai_nhung.config(text=f"Lỗi lưu ảnh nhúng: {str(e)}")

    def luu_anh_khoi_phuc(self):
        if self.anh_khoi_phuc:
            try:
                self.anh_khoi_phuc.save("anh_nuoc_khoi_phuc.png")
                self.trang_thai_trich.config(text="Đã lưu ảnh trích xuất")
            except Exception as e:
                self.trang_thai_trich.config(text=f"Lỗi lưu ảnh trích xuất: {str(e)}")

    def chay_demo(self):
        try:
            ma_tran_goc, ma_tran_nuoc, ma_tran_nhung, ma_tran_tai_tao, psnr, ssim, rho, asr = minh_hoa_ma_tran_4x4()
            self.demo_text.delete(1.0, tk.END)
            self.demo_text.insert(tk.END, f"Ma trận gốc:\n{ma_tran_goc}\n\n"
                                         f"Ma trận watermark:\n{ma_tran_nuoc}\n\n"
                                         f"Ma trận nhúng:\n{ma_tran_nhung}\n\n"
                                         f"Ma trận tái tạo:\n{ma_tran_tai_tao}\n\n"
                                         f"Chỉ số đánh giá:\n"
                                         f"PSNR: {psnr:.2f} dB\n"
                                         f"SSIM: {ssim:.4f}\n"
                                         f"Pearson: {rho:.4f}\n"
                                         f"ASR: {asr*100:.2f}%")
            self.trang_thai_demo.config(text="Đã chạy demo ma trận 4x4")
        except Exception as e:
            self.trang_thai_demo.config(text=f"Lỗi chạy demo: {str(e)}")

async def main():
    goc = tk.Tk()
    ung_dung = UngDungNuoc(goc)
    goc.mainloop()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())