# =============================================================================
# NADER - Neural Architecture Design via Multi-Agent Collaboration
# Google Colab Execution Script
# =============================================================================
# 이 파일을 Google Colab에서 실행하세요.
# 
# 사용법:
# 1. Google Drive에 NADER 프로젝트를 업로드
# 2. 이 노트북을 Colab에서 실행
# 3. GPU 런타임 선택 (런타임 > 런타임 유형 변경 > GPU)
# =============================================================================

# %% [markdown]
# # NADER - Google Colab Setup and Execution

# %% Cell 1: Google Drive 마운트
print("=" * 60)
print("Step 1: Mounting Google Drive...")
print("=" * 60)

from google.colab import drive
drive.mount('/content/drive')

# %% Cell 2: 프로젝트 경로 설정
print("\n" + "=" * 60)
print("Step 2: Setting up project paths...")
print("=" * 60)

import os
import sys

# ============================================
# 여기서 프로젝트 경로를 수정하세요!
# ============================================
PROJECT_PATH = "/content/drive/MyDrive/NADER"  # Google Drive 내 NADER 폴더 경로
# ============================================

# 경로 확인 및 이동
if not os.path.exists(PROJECT_PATH):
    raise FileNotFoundError(f"프로젝트 경로를 찾을 수 없습니다: {PROJECT_PATH}")

os.chdir(PROJECT_PATH)
print(f"Working directory: {os.getcwd()}")

# Python path 추가
sys.path.insert(0, PROJECT_PATH)
sys.path.insert(0, os.path.join(PROJECT_PATH, 'nader'))
sys.path.insert(0, os.path.join(PROJECT_PATH, 'zero-cost-nas'))

print(f"Python path updated!")

# %% Cell 3: 의존성 설치
print("\n" + "=" * 60)
print("Step 3: Installing dependencies...")
print("=" * 60)

# requirements.txt로 전체 의존성 설치
# !pip install -r requirements.txt

# 위 명령이 안 되면 아래 개별 설치 사용:
# !pip install -q numpy<2.0.0 pandas scipy matplotlib seaborn tqdm
# !pip install -q yacs easydict termcolor prettytable typing_extensions Pillow
# !pip install -q timm thop tensorboard graphviz
# !pip install -q nas-bench-201 openai>=1.33.0 langchain>=0.2.1 langchain-community chromadb tiktoken
# !pip install -q mmengine mmcv-lite>=2.0.0rc4

# %% Cell 4: GPU 확인 및 환경 설정
print("\n" + "=" * 60)
print("Step 4: Checking GPU and configuring environment...")
print("=" * 60)

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: GPU not available! Please enable GPU in Runtime > Change runtime type")

# %% Cell 5: Config 설정 (GPU 활성화)
print("\n" + "=" * 60)
print("Step 5: Configuring NADER for GPU...")
print("=" * 60)

# Config 파일 직접 수정 (import 대신 파일 수정으로 langchain 의존성 회피)
config_path = os.path.join(PROJECT_PATH, 'nader/train_utils/config.py')
print(f"Config path: {config_path}")

# config.py 파일 읽기
with open(config_path, 'r') as f:
    config_content = f.read()

# USE_GPU와 NUM_WORKERS 설정 변경
# USE_GPU = True (GPU 사용)
if '_C.USE_GPU = False' in config_content:
    config_content = config_content.replace('_C.USE_GPU = False', '_C.USE_GPU = True')
    print("✓ USE_GPU: False -> True")
else:
    print("✓ USE_GPU: already True")

# NUM_WORKERS = 4 (Colab용)
if '_C.DATA.NUM_WORKERS = 0' in config_content:
    config_content = config_content.replace('_C.DATA.NUM_WORKERS = 0', '_C.DATA.NUM_WORKERS = 4')
    print("✓ NUM_WORKERS: 0 -> 4")
else:
    print("✓ NUM_WORKERS: already configured")

# 파일 저장
with open(config_path, 'w') as f:
    f.write(config_content)

print("\n✓ Config 설정 완료!")
print("  - USE_GPU = True")
print("  - NUM_WORKERS = 4")

# %% Cell 6: OpenAI API Key 설정
print("\n" + "=" * 60)
print("Step 6: Setting OpenAI API Key...")
print("=" * 60)

import os

# ============================================
# 여기에 OpenAI API Key를 입력하세요!
# ============================================
OPENAI_API_KEY = "your_openai_api_key_here"  # <-- 수정 필요!
LLM_MODEL_NAME = "gpt-4"  # 또는 "gpt-4-turbo", "gpt-3.5-turbo"
# ============================================

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LLM_MODEL_NAME"] = LLM_MODEL_NAME

if OPENAI_API_KEY == "your_openai_api_key_here":
    print("⚠️  WARNING: OpenAI API Key가 설정되지 않았습니다!")
    print("   위의 OPENAI_API_KEY를 실제 키로 변경하세요.")
else:
    print(f"✓ OpenAI API Key 설정 완료")
    print(f"✓ LLM Model: {LLM_MODEL_NAME}")

# %% Cell 7: NADER 실행 파라미터 설정
print("\n" + "=" * 60)
print("Step 7: NADER Execution Parameters...")
print("=" * 60)

# ============================================
# 실행 파라미터 설정
# ============================================
PARAMS = {
    "max_iter": 2,          # Iteration 횟수
    "dataset": "cifar10",   # cifar10, cifar100, imagenet16-120
    "gen_num": 50,          # 생성할 아키텍처 수
    "proxy": "synflow",     # synflow, snip, grasp, fisher, grad_norm, jacob_cov
    "top_k": 5,             # Top-K 선정
    "epochs": 200,          # 학습 에포크 수
    "width": 5,             # DFS width
    "seed": 777,            # Random seed
    "cluster": "colab",     # 클러스터 설정 (colab, local, l40, a800, 4090d)
    
    # ============================================
    # Google Drive 자동 백업 (각 이터레이션 끝마다 logs+output 저장)
    # GPU 런타임이 끊겨도 데이터가 보존됩니다!
    # ============================================
    "gdrive_backup_dir": "/content/drive/MyDrive/NADER_Backup",  # None으로 설정시 백업 비활성화
    
    # ============================================
    # Resume 옵션 (중단 후 재개시 사용)
    # ============================================
    "resume": False,        # True로 변경하면 이전 상태에서 재개
    "resume_log_dir": None, # 특정 폴더에서 재개시 경로 지정 (None이면 자동)
}

print("실행 파라미터:")
for k, v in PARAMS.items():
    print(f"  {k}: {v}")

# %% Cell 7.5: Resume 상태 파일 생성 (중단 후 재개시 사용)
# ============================================
# 중단된 실험을 재개하려면 이 셀을 실행하세요!
# ============================================
GENERATE_RESUME_STATE = False  # True로 변경하고 아래 설정 후 실행

if GENERATE_RESUME_STATE:
    import json
    import glob
    
    # 1. 로그 디렉토리 자동 찾기 또는 직접 지정
    log_base = f"logs/nas-bench-201/{PARAMS['dataset']}"
    log_dirs = glob.glob(f"{log_base}/*seed{PARAMS['seed']}*")
    
    if log_dirs:
        # 가장 최근 폴더 선택
        LOG_DIR = sorted(log_dirs)[-1]
        print(f"✓ 로그 디렉토리 발견: {LOG_DIR}")
    else:
        LOG_DIR = ""  # 직접 입력 필요
        print("⚠️ 로그 디렉토리를 수동으로 지정하세요!")
    
    # 2. block_txt에서 generated_archs 생성
    block_txt_dir = os.path.join(LOG_DIR, 'block_txt')
    if os.path.exists(block_txt_dir):
        generated_archs = []
        for filename in os.listdir(block_txt_dir):
            if filename.endswith('.txt') and 'resnet_basic.txt' != filename:
                with open(os.path.join(block_txt_dir, filename), 'r') as f:
                    arch_content = f.read()
                model_name = filename.replace('.txt', '')
                generated_archs.append({
                    "arch": arch_content,
                    "model_name": model_name
                })
        
        # generated_archs_iter1.json 저장
        archs_path = os.path.join(LOG_DIR, 'generated_archs_iter1.json')
        with open(archs_path, 'w') as f:
            json.dump(generated_archs, f, indent=2)
        print(f"✓ generated_archs_iter1.json 생성 완료! ({len(generated_archs)} architectures)")
    
    # 3. resume_state.json 생성
    RESUME_ITER = 1   # 멈춘 iteration 번호
    RESUME_STEP = 2   # 재개할 step (2=Proxy 평가부터, 3=학습부터)
    
    resume_state = {
        "current_iter": RESUME_ITER,
        "current_step": RESUME_STEP,
        "overall_best_test_acc": -1,
        "overall_best_val_acc": -1,
        "overall_best_model": None
    }
    
    state_path = os.path.join(LOG_DIR, 'resume_state.json')
    with open(state_path, 'w') as f:
        json.dump(resume_state, f, indent=2)
    print(f"✓ resume_state.json 생성 완료! (iter={RESUME_ITER}, step={RESUME_STEP})")
    
    # PARAMS 업데이트
    PARAMS["resume"] = True
    PARAMS["resume_log_dir"] = LOG_DIR
    print(f"\n✓ Resume 준비 완료! PARAMS['resume'] = True로 설정됨")
    print(f"  Step 8을 실행하면 iter {RESUME_ITER}의 step {RESUME_STEP}부터 시작합니다.")

# %% Cell 8: NADER 실행 (Command Line)
print("\n" + "=" * 60)
print("Step 8: Running NADER...")
print("=" * 60)

# Resume 옵션 추가
resume_args = ""
if PARAMS.get("resume", False):
    resume_args = " --resume"
    if PARAMS.get("resume_log_dir"):
        resume_args += f" --resume-log-dir \"{PARAMS['resume_log_dir']}\""
    print(f"[Resume Mode] 이전 상태에서 재개합니다.")

# Google Drive 자동 백업 옵션 추가
gdrive_arg = ""
if PARAMS.get("gdrive_backup_dir"):
    gdrive_arg = f' --gdrive-dir "{PARAMS["gdrive_backup_dir"]}"'
    print(f"[Auto Backup] 각 이터레이션 종료 후 {PARAMS['gdrive_backup_dir']}에 자동 백업됩니다.")

# 명령어 생성
cmd = f"""python nader/nader-nas-bench-201-full.py \
    --max-iter {PARAMS['max_iter']} \
    --dataset {PARAMS['dataset']} \
    --gen-num {PARAMS['gen_num']} \
    --proxy {PARAMS['proxy']} \
    --top-k {PARAMS['top_k']} \
    --epochs {PARAMS['epochs']} \
    --width {PARAMS['width']} \
    --seed {PARAMS['seed']} \
    --cluster {PARAMS['cluster']}{resume_args}{gdrive_arg}"""

print(f"실행 명령:\n{cmd}\n")

# 실행
# !{cmd}

# %% Cell 9: 결과 확인
print("\n" + "=" * 60)
print("Step 9: Checking Results...")
print("=" * 60)

import os
import json

# 결과 경로
result_base = f"logs/nas-bench-201/{PARAMS['dataset']}"

# iteration_results.jsonl 파일 찾기
for root, dirs, files in os.walk(result_base):
    for f in files:
        if f == 'iteration_results.jsonl':
            result_path = os.path.join(root, f)
            print(f"\n결과 파일 발견: {result_path}")
            
            with open(result_path, 'r') as file:
                for line in file:
                    result = json.loads(line)
                    print(f"\n--- Iteration {result['iteration']} ---")
                    print(f"  Best Model: {result['best_model_this_iter']['model_name']}")
                    print(f"  Test Acc: {result['best_model_this_iter']['test_acc']:.2f}%")
                    print(f"  Total Time: {result['total_time_sec']:.2f}s")

print("\n" + "=" * 60)
print("NADER 실행 완료!")
print("=" * 60)
