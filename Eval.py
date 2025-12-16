import os
<<<<<<< HEAD
import re
=======
>>>>>>> b0f38e7fe77292f9bcb01f9e4a7df925126ff6ed
import glob
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from model import Teacher, Student, Student_x
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode
<<<<<<< HEAD
import numpy as np
import cv2
import importlib
import sys
import traceback
from pathlib import Path

# Try to load local CLIP once (best-effort). If missing, set clip_mod=None and skip later.
clip_mod = None
clip_model_global = None
try:
    # repo_root should be the parent directory that contains the CoA package
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    # ensure parent is on sys.path so both 'CoA.CLIP.clip' and 'CLIP.clip' can be found
    parent = os.path.abspath(os.path.join(repo_root, '..'))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    tried = []
    for modname in ('CoA.CLIP.clip', 'CLIP.clip', 'CoA.CLIP.clip as clip'):
        try:
            clip_mod = importlib.import_module(modname.split(' as ')[0])
            break
        except Exception as _e:
            tried.append((modname, str(_e)))
            clip_mod = None

    # load model to CPU if possible; tolerate any failure and continue without CLIP
    if clip_mod is not None:
        try:
            device_try = torch.device('cpu')
            download_root = os.path.join(repo_root, 'clip_model') if os.path.isdir(os.path.join(repo_root, 'clip_model')) else os.path.join(parent, 'CoA', 'clip_model')
            clip_model_global, _ = clip_mod.load('ViT-B/32', device=device_try, download_root=download_root)
            if clip_model_global is not None:
                clip_model_global.to('cpu')
                clip_model_global.eval()
        except Exception:
            clip_model_global = None
        # write a small CLIP init status snapshot for debugging
        try:
            out_dir = os.path.join(repo_root, 'outputs', 'clip_dehaze')
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, 'clip_init_status.txt'), 'a') as stf:
                stf.write(f"clip_mod_exists={clip_mod is not None}, clip_model_loaded={clip_model_global is not None}\n")
        except Exception:
            pass
except Exception:
    clip_mod = None
    clip_model_global = None
    try:
        out_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'clip_dehaze')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'clip_init_status.txt'), 'a') as stf:
            stf.write("clip_import_failed\n")
    except Exception:
        pass

# common image transform used by dehaze(); placed at module top so dehaze() can use it
transform = Compose([
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])
=======
>>>>>>> b0f38e7fe77292f9bcb01f9e4a7df925126ff6ed

# MODEL_PATH = './model/Teacher_model/Teacher.pth'
# OUTPUT_FOLDER = './outputs/Teacher'

# MODEL_PATH = './model/Student_model/Student.pth'
# OUTPUT_FOLDER = './outputs/Student'

MODEL_PATH = './model/EMA_model/EMA_r.pth'
<<<<<<< HEAD
OUTPUT_FOLDER = './outputs/clip_dehaze'

print(f"模型文件是否存在: {os.path.exists(MODEL_PATH)}")

# Ensure `model` exists in module globals regardless of how the file is executed.
# Some execution paths (importing as module or running via stdin) may not run the
# original __main__ block that initializes `model`, which causes NameError below.
try:
    model
except NameError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = Student_x().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("模型加载成功（fallback）！")
    except Exception as _e:
        # If loading fails, keep a model instance if possible, but warn the user.
        try:
            model = Student_x().to(device)
        except Exception:
            model = None
        print(f"模型加载失败（fallback）: {_e}")


def dehaze(model, image_path, folder):
    pil = Image.open(image_path).convert('RGB')
    # prepare output path and skip if it already exists to avoid duplicate processing
    out_path = os.path.join(folder, os.path.basename(image_path))
    os.makedirs(folder, exist_ok=True)
    if os.path.exists(out_path):
        # If output exists, skip heavy work and avoid duplicate CLIP writes
        print(f"Skipping dehaze for {os.path.basename(image_path)} (output exists)")
        return

    haze = transform(pil).unsqueeze(0).to(device)
    H, W = haze.shape[2], haze.shape[3]
    haze_rs = Resize((H // 16 * 16, W // 16 * 16), interpolation=InterpolationMode.BICUBIC, antialias=True)(haze)
    out = model(haze_rs)[0].squeeze(0)
    out = Resize((H, W), interpolation=InterpolationMode.BICUBIC, antialias=True)(out)
    torchvision.utils.save_image(out, out_path)

    # --- optional: score dehazed image with CLIP and write concise top descriptions ---
    # CLIP scoring: use global clip_mod / clip_model_global if available
    candidates = ["clear photo", "hazy photo", "foggy", "low light", "bright", "noisy", "sharp", "blurry", "indoor", "outdoor"]
    # always write the candidate prompts used once for traceability (before scoring)
    try:
        with open(os.path.join(folder, 'clip_prompts_used.txt'), 'a') as f2:
            f2.write(f"{os.path.basename(image_path)}: {candidates}\n")
    except Exception:
        pass

    if clip_mod is None or clip_model_global is None:
        # CLIP not available, skip scoring but keep prompts recorded
        return
    try:
        # tokenization may rely on repo-local tokenize implementation
        tokenized = clip_mod.tokenize(candidates).to('cpu')
        with torch.no_grad():
            # preprocess the original PIL image for CLIP image encoder
            img_pre = Compose([Resize((224, 224)), ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
            img_t = img_pre(pil).unsqueeze(0).to('cpu')
            img_feat = clip_model_global.encode_image(img_t)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            text_feat = clip_model_global.encode_text(tokenized.to('cpu'))
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            logits = (img_feat @ text_feat.t()) * clip_model_global.logit_scale.exp()
            # probs are softmax probabilities over the candidate prompts
            probs_t = logits.softmax(dim=-1).squeeze(0).cpu()

            # select top-K unique labels (deduplicate by label and keep highest score)
            K = 3
            # probs_t is a 1D tensor of softmax probabilities on CPU
            topk_vals, topk_inds = probs_t.topk(min(K, probs_t.numel()), dim=-1)
            # build mapping label->score, keep max score if duplicates
            label_scores = {}
            # flatten topk results to Python lists first to avoid multi-element tensor->scalar conversions
            try:
                topk_inds_list = topk_inds.view(-1).tolist()
            except Exception:
                topk_inds_list = list(topk_inds)
            try:
                topk_vals_list = topk_vals.view(-1).tolist()
            except Exception:
                topk_vals_list = list(topk_vals)

            for idx, val in zip(topk_inds_list, topk_vals_list):
                try:
                    idx_i = int(idx)
                except Exception:
                    # fallback: if idx is a list/tuple pick first element
                    if hasattr(idx, '__iter__'):
                        idx_i = int(idx[0])
                    else:
                        continue
                try:
                    val_f = float(val)
                except Exception:
                    # fallback: if val is array-like, take first element
                    if hasattr(val, '__iter__'):
                        val_f = float(val[0])
                    else:
                        continue
                lbl = candidates[idx_i] if 0 <= idx_i < len(candidates) else None
                if lbl is None:
                    continue
                # prefer the larger score if label repeated
                prev = label_scores.get(lbl)
                if prev is None or val_f > prev:
                    label_scores[lbl] = val_f

            # produce sorted list of (label, score) descending and keep up to K items
            sorted_desc = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)[:K]

            # print a concise log line and append to file in outputs
            # NOTE: score values are softmax probabilities (sum over candidates ~= 1)
            # They represent relative likelihood of the image matching each candidate
            # prompt under the CLIP model (not free-form captions). Higher is better.
            out_line = f"{os.path.basename(image_path)} -> {sorted_desc}"
            # write a short human-readable explanation file once per folder
            try:
                info_path = os.path.join(folder, 'clip_scores_info.txt')
                if not os.path.exists(info_path):
                    with open(info_path, 'w', encoding='utf-8') as inf:
                        inf.write('Note: scores are CLIP softmax probabilities over the fixed candidate prompts.\\n')
                        inf.write('Candidates are chosen by the script; these are not generated captions.\\n')
                        inf.write('Higher score => CLIP thinks the image matches that prompt more.\\n')
            except Exception:
                pass
            print(f"[CLIP_DESC] {out_line}")
            with open(os.path.join(folder, 'clip_descriptions.txt'), 'a') as f:
                f.write(out_line + '\n')
            # (no second write of prompts) -- prompts already recorded above
    except Exception as e:
        # Log exception per-image for debugging and continue
        try:
            err_path = os.path.join(folder, 'clip_err.log')
            with open(err_path, 'a') as ef:
                ef.write(f"Image: {os.path.basename(image_path)}\n")
                ef.write(traceback.format_exc())
                ef.write('\n')
        except Exception:
            pass
        print(f"[CLIP_ERROR] {os.path.basename(image_path)} -> {str(e)}")
        return

def calculate_metrics(dehazed_path, clear_path):
    """计算去雾图像与清晰图像的SSIM和PSNR"""
    # 读取图像（转换为灰度计算SSIM）
    dehazed = np.array(Image.open(dehazed_path).convert('L'))  # 转为灰度
    clear = np.array(Image.open(clear_path).convert('L'))
    assert dehazed.shape == clear.shape, f"尺寸不匹配: {dehazed.shape} vs {clear.shape}" 
    # 计算SSIM（范围0-1，越高越好）
    ssim_value = ssim(dehazed, clear, data_range=255)
    
    # 计算PSNR（单位dB，越高越好）
    mse = np.mean((dehazed - clear) ** 2)
    psnr_value = 10 * np.log10(255**2 / mse) if mse != 0 else float('inf')
    
    return ssim_value, psnr_value

if __name__ == '__main__':

    # transform = Compose([
    #     ToTensor(),
    #     Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
=======
OUTPUT_FOLDER = './outputs/EMA'


def dehaze(model, image_path, folder):
    haze = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    h, w = haze.shape[2], haze.shape[3]
    haze = Resize((h // 16 * 16, w // 16 * 16), interpolation=InterpolationMode.BICUBIC, antialias=True)(haze)
    out = model(haze)[0].squeeze(0)
    out = Resize((h, w), interpolation=InterpolationMode.BICUBIC, antialias=True)(out)
    torchvision.utils.save_image(out, os.path.join(folder, os.path.basename(image_path)))


if __name__ == '__main__':

    transform = Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
>>>>>>> b0f38e7fe77292f9bcb01f9e4a7df925126ff6ed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = Teacher().to(device)
    # model = Student().to(device)
    model = Student_x().to(device)

<<<<<<< HEAD
    #源代码
    # model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    # model.eval()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# 在 __main__ 中添加：
EVAL_MODE =False  # 设为True启用评估模式

if EVAL_MODE:
    # 路径设置
    hazy_dir = 'dataset/Haze4K/test/hazy_school'
    clear_dir = 'dataset/Haze4K/test/clear_school'  # 真正的清晰图像目录
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 获取配对图像（确保同名）
    hazy_images = sorted(glob.glob(os.path.join(hazy_dir, '*.jpg')))
    clear_images = []
    for h_path in hazy_images:
        # 提取数字部分（如从 "1000_0.73_1.8.png" 提取 "1000"）
        base = os.path.basename(h_path)
        num = re.match(r'^(\d+)', base).group(1)  # 匹配开头的数字
        clear_path = os.path.join(clear_dir, f"{num}.jpg")  # 拼接为 1000.jpg
        clear_images.append(clear_path)

    total_ssim, total_psnr = 0, 0
    for hazy_path, true_clear_path in zip(hazy_images, clear_images):
        # 去雾处理
        dehazed_path = os.path.join(OUTPUT_FOLDER, os.path.basename(hazy_path))
        
        try:
            # 检查去雾结果文件
            if not os.path.exists(dehazed_path):
                print(f"警告: 去雾文件未生成，跳过 {dehazed_path}")
                continue
            
        # 读取图像
            dehazed_img = Image.open(dehazed_path)
            true_clear_img = Image.open(true_clear_path)
            
            # 尺寸对齐
            if true_clear_img.size != dehazed_img.size:
                true_clear_img = true_clear_img.resize(dehazed_img.size, Image.BICUBIC)
            # 4. 计算指标（去雾结果 vs 真实清晰图）
            ssim_val, psnr_val = calculate_metrics(dehazed_path, true_clear_path)
            total_ssim += ssim_val
            total_psnr += psnr_val
            
            print(f"图像: {os.path.basename(hazy_path)} | SSIM: {ssim_val:.4f} | PSNR: {psnr_val:.2f} dB")
        except Exception as e:
            print(f"处理失败 {os.path.basename(hazy_path)}: {str(e)}")
            continue
    # 打印平均指标
    avg_ssim = total_ssim / len(hazy_images)
    avg_psnr = total_psnr / len(hazy_images)
    print(f"\n平均指标 | SSIM: {avg_ssim:.4f} | PSNR: {avg_psnr:.2f} dB")



    

else:
    # 仅去雾模式
    # 修改为当前路径下的路径
    INPUT_FOLDER = 'dataset/Haze4K/test/hazy_school'
=======
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    INPUT_FOLDER = './test'
>>>>>>> b0f38e7fe77292f9bcb01f9e4a7df925126ff6ed

    images = glob.glob(os.path.join(INPUT_FOLDER, '*jpg')) + glob.glob(os.path.join(INPUT_FOLDER, '*png')) + glob.glob(os.path.join(INPUT_FOLDER, '*jpeg'))

    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} | Elapsed: {elapsed} | Rate: {rate_fmt} items/sec"
    with torch.no_grad():
<<<<<<< HEAD
        print(f"Total images found: {len(images)}. Processing...")
        print(f"First few images: {images[:5]}")
        for image in tqdm(images, bar_format=bar_format, desc="Processing images:"):
            dehaze(model, image, OUTPUT_FOLDER)
        print(f"去雾结果已保存到: {OUTPUT_FOLDER}")
    print("处理完成！")
=======
        for image in tqdm(images, bar_format=bar_format, desc="Models are struggling to get out of the fog 😊 :"):
            dehaze(model, image, OUTPUT_FOLDER)
>>>>>>> b0f38e7fe77292f9bcb01f9e4a7df925126ff6ed
