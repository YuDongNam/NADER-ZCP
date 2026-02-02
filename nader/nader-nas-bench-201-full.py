import os
import sys
import warnings
import subprocess
import time
import shutil
import argparse
import math
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.utils import *
from nader import Nader


warnings.filterwarnings("ignore")

ZERO_COST_REPO_PATH = './zero-cost-nas'
if ZERO_COST_REPO_PATH not in sys.path:
    sys.path.append(ZERO_COST_REPO_PATH)

# 라이브러리 임포트 시도
try:
    from foresight.pruners import predictive
    from ModelFactory.register import Registers, import_all_modules_for_register2
except ImportError:
    pass 

def print_and_save(text, file):
    print(text)
    with open(file, 'a') as f:
        f.write(text + '\n')
        f.flush()

# =============================================================================
# Resume State Management
# =============================================================================
import json as json_module

def save_iteration_state(log_dir, state):
    """Save iteration state for resume functionality"""
    state_path = os.path.join(log_dir, 'resume_state.json')
    with open(state_path, 'w') as f:
        json_module.dump(state, f, indent=2)
    print(f"[Resume] State saved: iter={state['current_iter']}, step={state['current_step']}")

def load_iteration_state(log_dir):
    """Load iteration state for resume"""
    state_path = os.path.join(log_dir, 'resume_state.json')
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            state = json_module.load(f)
        print(f"[Resume] State loaded: iter={state['current_iter']}, step={state['current_step']}")
        return state
    return None

def save_generated_archs(log_dir, current_iter, archs):
    """Save generated architectures for resume"""
    archs_path = os.path.join(log_dir, f'generated_archs_iter{current_iter}.json')
    with open(archs_path, 'w') as f:
        json_module.dump(archs, f, indent=2)

def load_generated_archs(log_dir, current_iter):
    """Load previously generated architectures"""
    archs_path = os.path.join(log_dir, f'generated_archs_iter{current_iter}.json')
    if os.path.exists(archs_path):
        with open(archs_path, 'r') as f:
            return json_module.load(f)
    return None

def save_top_k_candidates(log_dir, current_iter, candidates):
    """Save top-k candidates for resume"""
    path = os.path.join(log_dir, f'top_k_candidates_iter{current_iter}.json')
    # Convert to serializable format
    serializable = [(float(score), item) for score, item in candidates]
    with open(path, 'w') as f:
        json_module.dump(serializable, f, indent=2)

def load_top_k_candidates(log_dir, current_iter):
    """Load top-k candidates for resume"""
    path = os.path.join(log_dir, f'top_k_candidates_iter{current_iter}.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json_module.load(f)
        return [(score, item) for score, item in data]
    return None


blockmap={
    "resnet_basic":{777:"resnet_basic",888:"resnet_basic",999:"resnet_basic"},
    "nasbench_random":{
        777:"nasbench201_seed777",
        888:"resnet_basic",
        999:"nasbench201_seed777"
    }
}

def get_model_instance(model_name, num_classes=10):
    try:
        # 동적으로 생성된 모델 파일들을 다시 로드하여 레지스트리 업데이트
        import_all_modules_for_register2()
        if model_name in Registers.model:
            model_class = Registers.model[model_name]
            # 모델 생성 (보통 num_classes를 인자로 받음)
            model = model_class(num_classes=num_classes)
            return model
        else:
            print(f"[Model Error] Model {model_name} not found in Registers.model")
            return None
    except Exception as e:
        print(f"[Model Error] Failed to instantiate model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

# 프록시 평가 코드
def evaluate_proxy(model_name, proxy_type='synflow', batch_size=32, seed=888):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # 1. 모델 생성
    model = get_model_instance(model_name)
    if model is None: return -1.0
    model = model.to(device)
    model.train()

    # 2. 데이터 로드 (1 배치)
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        # inputs, targets 로딩 제거 (find_measures 내부에서 처리함)
    except Exception as e:
        print(f"[Proxy Eval Error] Data loading failed: {e}")
        return -1.0

    # 3. 점수 계산
    try:
        # predictive.find_measures 인자 수정
        # net_orig: 모델
        # dataload_info: ('random', 배치수, 클래스수)
        # loss_fn: 손실함수
        measures = predictive.find_measures(
            net_orig=model, 
            dataloader=loader, 
            dataload_info=('random', 1, 10),
            measure_names=[proxy_type], 
            loss_fn=nn.CrossEntropyLoss(), 
            device=device
        )
        score = measures[proxy_type]
        del model
        torch.cuda.empty_cache()
        return score
    except Exception as e:
        print(f"[Proxy Eval Error] Score calculation failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return -1.0

def select_best_trained_model(nader, trained_models, runs_log_path):
    """
    학습된 모델들 중 test_acc 기준으로 가장 성능이 좋은 모델 선정
    
    Args:
        nader: Nader instance
        trained_models: list of model names that were trained
        runs_log_path: path to log file
    
    Returns:
        (best_model_name, best_test_acc, best_val_acc)
    """
    best_model_name = None
    best_test_acc = -1
    best_val_acc = -1
    
    print_and_save(f"\n>>> Selecting Best Model from {len(trained_models)} trained models...", runs_log_path)
    
    for model_name in trained_models:
        test_acc_file = os.path.join(nader.train_log_dir, model_name, '1', 'test_acc.txt')
        val_acc_file = os.path.join(nader.train_log_dir, model_name, '1', 'val_acc.txt')
        
        # Try reading test_acc.txt first
        has_test_acc = False
        if os.path.exists(test_acc_file):
            try:
                with open(test_acc_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        data_lines = [l.strip() for l in lines[1:] if l.strip()]
                        if data_lines:
                            last_line = data_lines[-1]
                            parts = last_line.split(',')
                            test_acc = float(parts[1])
                            has_test_acc = True
            except Exception as e:
                print_and_save(f"  Warning: Error reading test_acc.txt for {model_name}: {e}", runs_log_path)

        # Read val_acc.txt
        val_acc = -1
        if os.path.exists(val_acc_file):
            try:
                with open(val_acc_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        data_lines = [l.strip() for l in lines[1:] if l.strip()]
                        if data_lines:
                            last_line = data_lines[-1]
                            parts = last_line.split(',')
                            val_acc = float(parts[1])
            except Exception as e:
                print_and_save(f"  Warning: Error reading val_acc.txt for {model_name}: {e}", runs_log_path)
        
        # Fallback Logic
        if not has_test_acc:
            if val_acc != -1:
                print_and_save(f"  Warning: {model_name} - test_acc.txt not found, using val_acc as proxy.", runs_log_path)
                test_acc = val_acc
            else:
                 print_and_save(f"  Warning: {model_name} - neither test_acc.txt nor val_acc.txt found/valid", runs_log_path)
                 continue

        print_and_save(f"  {model_name}: test_acc={test_acc:.2f}%, val_acc={val_acc:.2f}%", runs_log_path)
        
        # Update best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_val_acc = val_acc
            best_model_name = model_name
    
    if best_model_name:
        print_and_save(f"\n>>> Best Model Selected: {best_model_name}", runs_log_path)
        print_and_save(f"    Test Acc: {best_test_acc:.2f}%, Val Acc: {best_val_acc:.2f}%", runs_log_path)
    else:
        print_and_save(f"\n>>> ERROR: No valid model found!", runs_log_path)
    
    return best_model_name, best_test_acc, best_val_acc

def update_base_model_for_next_iter(nader, best_model_name, runs_log_path):
    """
    다음 iteration을 위해 base model 업데이트
    
    Args:
        nader: Nader instance
        best_model_name: best performing model name
        runs_log_path: path to log file
    """
    print_and_save(f"\n>>> Updating base model to: {best_model_name}", runs_log_path)
    
    # Update research team's base block
    nader.research.base_block = best_model_name
    
    print_and_save(f"    Base model updated successfully!", runs_log_path)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--max-iter',type=int,required=True)
    parser.add_argument('-d','--dataset',type=str,choices=['cifar10','cifar100','imagenet16-120'],required=True)
    parser.add_argument('-b','--base-block',type=str,default='resnet_basic',choices=['resnet_basic','convnext','nasbench_random'])
    parser.add_argument('-r','--research-team-name',type=str,default='nader',choices=['nader','nader_wor','nader_hc'])
    parser.add_argument('-c','--cluster',type=str,default='colab',choices=['l40','a800','4090d','colab','local'])
    parser.add_argument('--seed',type=int,default=777,choices=[777,888,999])
    parser.add_argument('--width',default=5,type=int)
    parser.add_argument('--inspiration-retriever-mode',default='random',type=str,choices=['random','reflection'])
    parser.add_argument('--research-experience',default=None,type=str,choices=[None,'research_fail_240807_experience'])
    parser.add_argument('-p','--proposer-mode',type=str,default='llm')
    parser.add_argument('--log-dir',type=str,default='logs/nas-bench-201')
    parser.add_argument('--train-log-dir',type=str,default='output/nas-bench-201')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs per model')
    parser.add_argument('--gen-num', type=int, default=50, help='number of archs to generate')
    parser.add_argument('--proxy', type=str, required=True, choices=['synflow','snip','grasp','fisher','grad_norm','jacob_cov','heuristic3','heuristic4'], help='proxy type')
    parser.add_argument('--top-k', type=int, default=5, help='top k selection')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume from last saved state')
    parser.add_argument('--resume-log-dir', type=str, default=None, help='Specific log directory to resume from')
    parser.add_argument('--model-name', type=str, default='gpt-5-nano', help='LLM model name')
    parser.add_argument('--gdrive-dir', type=str, default=None, help='Google Drive path for backup')
    
    args = parser.parse_args()
    cluster = args.cluster
    seed = args.seed
    max_iter = args.max_iter
    width = args.width
    dataset = args.dataset
    base_block=args.base_block
    research_team_name=args.research_team_name
    research_experience = args.research_experience
    inspiration_retriever_mode=args.inspiration_retriever_mode
    epochs = args.epochs
    gen_num = args.gen_num
    proxy = args.proxy
    top_k = args.top_k
    
    # Prioritize environment variable, then argument
    model_name = os.environ.get('LLM_MODEL_NAME', args.model_name)
    os.environ['LLM_MODEL_NAME'] = model_name
    print(f"Using Model: {model_name}")

    tag_prefix_base=f'{research_team_name}_{args.proposer_mode}_{base_block}_seed{seed}'
    log_dir = f'{args.log_dir}/{dataset}/{research_team_name}_{args.proposer_mode}_{base_block}_{inspiration_retriever_mode}_dfs-width{args.width}_run{args.max_iter}_seed{seed}'
    runs_log_path = os.path.join(log_dir, 'log_runs.log')
    set_seed(seed)
    nader = Nader(
        base_block=blockmap[base_block][seed],
        dataset=f"nas-bench-201-{dataset}",
        database_dir='data/nas-bench-201',
        inspiration_retriever_mode=inspiration_retriever_mode,
        research_team_name=research_team_name,
        develop_team_name='try-fb',
        research_use_experience=research_experience,
        develop_use_experiecne='develop_allfailed_240709_experience',
        log_dir=log_dir,
        train_log_dir=f'{args.train_log_dir}/{dataset}',
        tag_prefix_base=tag_prefix_base,
        research_mode='greedy', # dfs-one -> greedy to pick best global block even if graph is broken
        proposer_mode=args.proposer_mode,
        model_name=model_name
    )

    # -----------------------------------------------------
    # Iteration Loop - Multi-Iteration Evolutionary Search
    # -----------------------------------------------------
    overall_best_test_acc = -1
    overall_best_val_acc = -1
    overall_best_model = None
    
    # Create detailed iteration log file
    iter_log_path = os.path.join(log_dir, 'iteration_results.jsonl')
    
    # Resume handling
    resume_iter = 1
    resume_step = 1
    if args.resume:
        resume_log = args.resume_log_dir if args.resume_log_dir else log_dir
        resume_state = load_iteration_state(resume_log)
        if resume_state:
            resume_iter = resume_state['current_iter']
            resume_step = resume_state['current_step']
            overall_best_test_acc = resume_state.get('overall_best_test_acc', -1)
            overall_best_val_acc = resume_state.get('overall_best_val_acc', -1)
            overall_best_model = resume_state.get('overall_best_model', None)
            # ★ Resume시 base_block 복원 (이전 iteration의 best model)
            resumed_base_block = resume_state.get('next_base_block', None)
            if resumed_base_block:
                nader.research.base_block = resumed_base_block
                print_and_save(f"\n[Resume] Base model restored to: {resumed_base_block}", runs_log_path)
            print_and_save(f"\n[Resume] Resuming from Iteration {resume_iter}, Step {resume_step}", runs_log_path)
        else:
            print_and_save(f"\n[Resume] No saved state found. Starting from scratch.", runs_log_path)
    
    for current_iter in range(1, max_iter + 1):
        # Skip completed iterations when resuming
        if args.resume and current_iter < resume_iter:
            print_and_save(f"\n[Resume] Skipping completed iteration {current_iter}", runs_log_path)
            continue
            
        iter_start_time = time.time()
        
        print_and_save(f"\n{'='*70}", runs_log_path)
        print_and_save(f"ITERATION {current_iter}/{max_iter}", runs_log_path)
        print_and_save(f"Current Base Model: {nader.research.base_block}", runs_log_path)
        print_and_save(f"{'='*70}\n", runs_log_path)
        
        # Initialize timing variables to avoid NameError if steps are skipped during resume
        gen_elapsed = 0.0
        eval_elapsed = 0.0
        train_elapsed = 0.0
        
        # -----------------------------------------------------
        # [Step 1] Generate Architectures
        # -----------------------------------------------------
        # Resume: Skip Step 1 if already completed
        if args.resume and current_iter == resume_iter and resume_step > 1:
            generated_archs = load_generated_archs(log_dir, current_iter)
            if generated_archs:
                print_and_save(f"\n>>> STEP 1: [Resume] Loading {len(generated_archs)} previously generated architectures", runs_log_path)
            else:
                print_and_save(f"\n>>> STEP 1: [Resume] No saved architectures found. Regenerating...", runs_log_path)
                gen_start_time = time.time()
                generated_archs = nader.propose_batch_architectures(gen_num)
                gen_elapsed = time.time() - gen_start_time
                print_and_save(f"Total Generated Architectures: {len(generated_archs)} (Time: {gen_elapsed:.2f}s)", runs_log_path)
                save_generated_archs(log_dir, current_iter, generated_archs)
        else:
            print_and_save(f"\n>>> STEP 1: Generating {gen_num} Architectures...", runs_log_path)
            gen_start_time = time.time()
            generated_archs = nader.propose_batch_architectures(gen_num)
            gen_elapsed = time.time() - gen_start_time
            print_and_save(f"Total Generated Architectures: {len(generated_archs)} (Time: {gen_elapsed:.2f}s)", runs_log_path)
            # Save generated architectures for potential resume
            save_generated_archs(log_dir, current_iter, generated_archs)
        
        # Save state after Step 1
        save_iteration_state(log_dir, {
            'current_iter': current_iter,
            'current_step': 2,
            'overall_best_test_acc': overall_best_test_acc,
            'overall_best_val_acc': overall_best_val_acc,
            'overall_best_model': overall_best_model,
            'next_base_block': nader.research.base_block  # ★ 현재 base_block 저장
        })

        # -----------------------------------------------------
        # [Step 2] Proxy Evaluation & Selection
        # -----------------------------------------------------
        # Resume: Skip Step 2 if already completed
        if args.resume and current_iter == resume_iter and resume_step > 2:
            top_k_candidates = load_top_k_candidates(log_dir, current_iter)
            if top_k_candidates:
                print_and_save(f"\n>>> STEP 2: [Resume] Loading {len(top_k_candidates)} previously selected candidates", runs_log_path)
            else:
                # Fall through to normal Step 2 processing
                top_k_candidates = None
        else:
            top_k_candidates = None
        
        if top_k_candidates is None:
            print_and_save(f"\n>>> STEP 2: Evaluating with Proxy [{args.proxy}]...", runs_log_path)
            eval_start_time = time.time()
            scored_candidates = []
            
            for item in tqdm(generated_archs, desc=f"Iter {current_iter} - Evaluating"):
                model_name = item['model_name']
                score = evaluate_proxy(model_name, args.proxy)
                
                # NaN/Inf 점수 필터링 (유효한 점수만 추가)
                if score is not None and not math.isnan(score) and not math.isinf(score) and score > 0:
                    scored_candidates.append((score, item))
                else:
                    print_and_save(f"  [Filtered] {model_name}: Invalid score ({score})", runs_log_path)
            
            print_and_save(f"  Valid models after filtering: {len(scored_candidates)}/{len(generated_archs)}", runs_log_path)
            
            # 점수 높은 순 정렬 (Synflow 등은 클수록 좋음)
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            top_k_candidates = scored_candidates[:args.top_k]
            eval_elapsed = time.time() - eval_start_time
            
            if len(top_k_candidates) == 0:
                print_and_save(f"  [WARNING] No valid models found in this iteration! Skipping training.", runs_log_path)
                continue
            
            # Save top-k candidates for resume
            save_top_k_candidates(log_dir, current_iter, top_k_candidates)
        
        print_and_save(f"\n>>> Top {min(args.top_k, len(top_k_candidates))} Selected:", runs_log_path)
        top_k_info = []
        for i, (score, item) in enumerate(top_k_candidates):
            print_and_save(f"Rank {i+1} (Proxy Score: {score:.4f}): {item['model_name']}", runs_log_path)
            top_k_info.append({
                'rank': i+1,
                'model_name': item['model_name'],
                'proxy_score': float(score)
            })
        
        # Save state after Step 2
        save_iteration_state(log_dir, {
            'current_iter': current_iter,
            'current_step': 3,
            'overall_best_test_acc': overall_best_test_acc,
            'overall_best_val_acc': overall_best_val_acc,
            'overall_best_model': overall_best_model,
            'next_base_block': nader.research.base_block  # ★ 현재 base_block 저장
        })

        # ---------------------------------------------------------
        # [Step 3] Training Top-K Architectures
        # ---------------------------------------------------------
        print_and_save(f"\n>>> STEP 3: Training Top {args.top_k} Architectures", runs_log_path)
        train_start_time = time.time()
        
        trained_models = []
        training_details = []

        for i, (score, item) in enumerate(top_k_candidates):
            rank_num = i + 1
            block_name = item['model_name']
            arch = item['arch']
            
            model_train_start = time.time()
            print_and_save(f"\n--- Training Rank {rank_num}/{args.top_k} : {block_name} ---", runs_log_path)

            # NADER가 이 아키텍처를 '새로운 것'으로 인식하게 하려면 파일에 써줘야 합니다.
            new_anno = {
                "iter": current_iter,
                "block_name": block_name,
                "raw_block_name": item.get('parent'),             # Parent block (e.g., p4834)
                "inspiration_id": item.get('inspiration_id'),    # ID for inspiration
                "block": arch,
            }
            
            # annos.json 파일에 append (Injection)
            try:
                with open(nader.blocks_path, 'a') as f:
                    f.write(json.dumps(new_anno) + '\n')
            except Exception as e:
                print_and_save(f"[Error] Failed to write to annos file: {e}", runs_log_path)
                continue

            # 쉘 스크립트 생성
            path_sh = os.path.join(log_dir, f"train_iter{current_iter}_rank{rank_num}_{block_name}.sh")
            models = nader.generate_train_script(cluster=cluster, batch_path=path_sh, target_models=[block_name])
            print_and_save(f"Generated Script: {path_sh} for models: {models}", runs_log_path)
            
            if not models or block_name not in models:
                print_and_save("-> Warning: Model not found in generated script (Maybe already trained?)", runs_log_path)
            
            log_dirs = [os.path.join(nader.train_log_dir, model, '1') for model in models]

            # Check if training is already completed
            train_status_path = os.path.join(nader.train_log_dir, block_name, '1', 'train_status.txt')
            val_acc_path = os.path.join(nader.train_log_dir, block_name, '1', 'val_acc.txt')
            
            already_trained = False
            # Condition 1: status file exists
            if os.path.exists(train_status_path):
                 already_trained = True
            # Condition 2: val_acc has enough epochs (fallback if status file missing)
            elif os.path.exists(val_acc_path):
                 try:
                     with open(val_acc_path, 'r') as f:
                         # Skip header, count data lines
                         lines = [l for l in f.readlines() if l.strip()]
                         if len(lines) > epochs: # > because header is 1 line. So 200 data + 1 header = 201.
                             print_and_save(f"-> Model {block_name} has {len(lines)-1} epochs in log. Considered finished.", runs_log_path)
                             already_trained = True
                 except:
                     pass

            if already_trained:
                 print_and_save(f"-> Model {block_name} seems already trained. Skipping execution.", runs_log_path)

            # 실행
            if not already_trained:
                # [FIX] stdout/stderr를 파일로 리다이렉트 (PIPE 버퍼 overflow 방지!)
                # PIPE 사용 시 버퍼(64KB)가 차면 학습 프로세스가 block되어 hang 발생
                train_stdout_log = os.path.join(log_dir, f"train_stdout_iter{current_iter}_rank{rank_num}.log")
                train_stderr_log = os.path.join(log_dir, f"train_stderr_iter{current_iter}_rank{rank_num}.log")
                
                stdout_file = open(train_stdout_log, 'w')
                stderr_file = open(train_stderr_log, 'w')
                process = subprocess.Popen(f'bash {path_sh}', shell=True, stdout=stdout_file, stderr=stderr_file)
                
                # 모니터링 with epoch progress timeout
                last_epoch_check = None
                last_epoch_time = time.time()
                EPOCH_PROGRESS_TIMEOUT = 600  # 10 minutes without epoch progress = hang
                
                try:
                    while len(log_dirs) > 0:
                        ns = []
                        all_finished = True
                        
                        ret_code = process.poll()
                        is_process_running = (ret_code is None)

                        for d in log_dirs:
                            status_file = os.path.join(d, 'train_status.txt')
                            val_file = os.path.join(d, 'val_acc.txt')
                            
                            curr_epoch = 0
                            if os.path.isfile(val_file):
                                with open(val_file, 'r') as f:
                                     lines = f.readlines()
                                     curr_epoch = max(0, len(lines) - 1)
                            ns.append(f"{curr_epoch}/{epochs}")

                            if not os.path.exists(status_file):
                                all_finished = False
                                if not is_process_running:
                                    print_and_save(f"[Error] Process terminated with code {ret_code} but {status_file} not found.", runs_log_path)
                                    # Read stderr from log file
                                    if os.path.exists(train_stderr_log):
                                        with open(train_stderr_log, 'r') as f:
                                            stderr_content = f.read()[-2000:]  # Last 2000 chars
                                            if stderr_content.strip():
                                                print(f"STDERR (last 2000 chars): {stderr_content}")
                                    all_finished = True
                        
                        # Check for epoch progress timeout (hang detection)
                        current_epoch_str = str(ns)
                        if last_epoch_check != current_epoch_str:
                            last_epoch_check = current_epoch_str
                            last_epoch_time = time.time()
                        else:
                            # No progress - check timeout
                            stall_time = time.time() - last_epoch_time
                            if stall_time > EPOCH_PROGRESS_TIMEOUT:
                                print_and_save(f"[Timeout] Training stuck at {ns} for {stall_time:.0f}s. Killing process...", runs_log_path)
                                process.kill()
                                process.wait()
                                # Create train_status.txt to mark as "failed but done"
                                for d in log_dirs:
                                    status_file = os.path.join(d, 'train_status.txt')
                                    if not os.path.exists(status_file):
                                        with open(status_file, 'w') as f:
                                            f.write('timeout_killed')
                                all_finished = True
                        
                        print_and_save(f"Training Progress: {ns}", runs_log_path)
                        
                        if all_finished: 
                            break
                        
                        time.sleep(10)
                finally:
                    # 파일 핸들 정리
                    stdout_file.close()
                    stderr_file.close()
            else:
                 # If skipped, just simulate success or do nothing
                 print_and_save(f"Rank {rank_num} Training skipped (Already done).", runs_log_path)

            model_train_elapsed = time.time() - model_train_start
            print_and_save(f"Rank {rank_num} Training finished! (Time: {model_train_elapsed:.2f}s)", runs_log_path)
            
            trained_models.append(block_name)
            training_details.append({
                'rank': rank_num,
                'model_name': block_name,
                'training_time_sec': model_train_elapsed
            })

            # ★ 각 모델 학습 완료 직후 즉시 Google Drive에 백업 ★
            # Colab 런타임이 끊기기 전에 학습된 모델을 바로 저장
            if args.gdrive_dir:
                try:
                    # 1. Output 폴더 백업 (학습된 모델)
                    model_src = os.path.join(nader.train_log_dir, block_name)
                    model_dest = os.path.join(args.gdrive_dir, 'output', dataset, block_name)
                    
                    if os.path.exists(model_src):
                        if not os.path.exists(os.path.dirname(model_dest)):
                            os.makedirs(os.path.dirname(model_dest), exist_ok=True)
                        
                        if sys.version_info >= (3, 8):
                            shutil.copytree(model_src, model_dest, dirs_exist_ok=True)
                        else:
                            from distutils.dir_util import copy_tree
                            copy_tree(model_src, model_dest)
                        
                        print_and_save(f"    [Backup] Model {block_name} backed up to: {model_dest}", runs_log_path)
                    
                    # 2. Logs 폴더 백업 (resume 정보, iteration 로그 등)
                    exp_name = os.path.basename(log_dir)
                    logs_dest = os.path.join(args.gdrive_dir, 'logs', os.path.dirname(log_dir).replace(args.log_dir + '/', ''), exp_name)
                    
                    if not os.path.exists(os.path.dirname(logs_dest)):
                        os.makedirs(os.path.dirname(logs_dest), exist_ok=True)
                    
                    if sys.version_info >= (3, 8):
                        shutil.copytree(log_dir, logs_dest, dirs_exist_ok=True)
                    else:
                        from distutils.dir_util import copy_tree
                        copy_tree(log_dir, logs_dest)
                    
                    print_and_save(f"    [Backup] Logs backed up to: {logs_dest}", runs_log_path)
                    
                except Exception as e:
                    print_and_save(f"    [Backup Warning] Failed to backup {block_name}: {e}", runs_log_path)

            # 결과 업데이트 및 mog.json/png 갱신
            try:
                 nader.research.mog_graph_manage.update_train_result(
                     nader.train_log_dir,
                     filename='test_acc',
                     tag_prefix=tag_prefix_base,
                     flush=True
                 )
            except Exception as e:
                 print_and_save(f"[Error] Failed to update mog graph: {e}", runs_log_path)

        train_elapsed = time.time() - train_start_time
        print_and_save(f"\n>>> All Top-K Training Completed (Total Training Time: {train_elapsed:.2f}s)", runs_log_path)

        # ---------------------------------------------------------
        # [Step 4] Select Best Model
        # ---------------------------------------------------------
        print_and_save(f"\n>>> STEP 4: Selecting Best Model from Iteration {current_iter}", runs_log_path)
        best_model_name, best_test_acc, best_val_acc = select_best_trained_model(nader, trained_models, runs_log_path)
        
        # Update overall best
        if best_test_acc > overall_best_test_acc:
            overall_best_test_acc = best_test_acc
            overall_best_val_acc = best_val_acc
            overall_best_model = best_model_name
        
        # ---------------------------------------------------------
        # [Step 5] Save Iteration Results
        # ---------------------------------------------------------
        iter_elapsed = time.time() - iter_start_time
        
        iteration_result = {
            'iteration': current_iter,
            'base_model': nader.research.base_block,
            'generation_time_sec': gen_elapsed,
            'evaluation_time_sec': eval_elapsed,
            'training_time_sec': train_elapsed,
            'total_time_sec': iter_elapsed,
            'num_generated': len(generated_archs),
            'top_k_candidates': top_k_info,
            'training_details': training_details,
            'best_model_this_iter': {
                'model_name': best_model_name,
                'test_acc': best_test_acc,
                'val_acc': best_val_acc
            },
            'overall_best': {
                'model_name': overall_best_model,
                'test_acc': overall_best_test_acc,
                'val_acc': overall_best_val_acc
            }
        }
        
        # Save to JSONL
        with open(iter_log_path, 'a') as f:
            f.write(json.dumps(iteration_result) + '\n')
        
        print_and_save(f"\n{'='*70}", runs_log_path)
        print_and_save(f"ITERATION {current_iter} SUMMARY:", runs_log_path)
        print_and_save(f"  Best Model: {best_model_name}", runs_log_path)
        print_and_save(f"  Test Acc: {best_test_acc:.2f}% | Val Acc: {best_val_acc:.2f}%", runs_log_path)
        print_and_save(f"  Time: Generation={gen_elapsed:.2f}s, Eval={eval_elapsed:.2f}s, Training={train_elapsed:.2f}s, Total={iter_elapsed:.2f}s", runs_log_path)
        print_and_save(f"  Overall Best: {overall_best_model} (Test Acc: {overall_best_test_acc:.2f}%)", runs_log_path)
        print_and_save(f"{'='*70}\n", runs_log_path)
        
        # ---------------------------------------------------------
        # [Step 6] Update Base Model for Next Iteration
        # ---------------------------------------------------------
        if current_iter < max_iter and best_model_name:
            update_base_model_for_next_iter(nader, best_model_name, runs_log_path)
            
            # Save state for next iteration (current iteration completed)
            save_iteration_state(log_dir, {
                'current_iter': current_iter + 1,
                'current_step': 1,
                'overall_best_test_acc': overall_best_test_acc,
                'overall_best_val_acc': overall_best_val_acc,
                'overall_best_model': overall_best_model,
                'next_base_block': nader.research.base_block  # ★ 다음 iteration의 base_block (best_model_name)
            })
        elif current_iter == max_iter:
            print_and_save(f"\n>>> Final Iteration Complete!", runs_log_path)
            print_and_save(f">>> Overall Best Model: {overall_best_model}", runs_log_path)
            print_and_save(f">>> Overall Best Test Acc: {overall_best_test_acc:.2f}%", runs_log_path)
            
            # Save final completed state
            save_iteration_state(log_dir, {
                'current_iter': max_iter,
                'current_step': 'completed',
                'overall_best_test_acc': overall_best_test_acc,
                'overall_best_val_acc': overall_best_val_acc,
                'overall_best_model': overall_best_model,
                'next_base_block': nader.research.base_block  # ★ 최종 base_block 저장
            })

        # ---------------------------------------------------------
        # [Step 7] Google Drive Backup (logs + output 폴더)
        # ---------------------------------------------------------
        if args.gdrive_dir:
            print_and_save(f"\n>>> STEP 7: Backing up to Google Drive: {args.gdrive_dir}", runs_log_path)
            backup_start = time.time()
            
            try:
                # 백업 루트 디렉토리 생성
                if not os.path.exists(args.gdrive_dir):
                    os.makedirs(args.gdrive_dir, exist_ok=True)
                
                # 1. logs 폴더 백업
                exp_name = os.path.basename(log_dir)
                logs_dest = os.path.join(args.gdrive_dir, 'logs', os.path.dirname(log_dir).replace(args.log_dir + '/', ''), exp_name)
                
                if not os.path.exists(os.path.dirname(logs_dest)):
                    os.makedirs(os.path.dirname(logs_dest), exist_ok=True)
                
                if sys.version_info >= (3, 8):
                    shutil.copytree(log_dir, logs_dest, dirs_exist_ok=True)
                else:
                    from distutils.dir_util import copy_tree
                    copy_tree(log_dir, logs_dest)
                print_and_save(f"    [logs] Backup completed: {logs_dest}", runs_log_path)
                
                # 2. output 폴더 백업 (학습 결과)
                output_src = nader.train_log_dir
                output_dest = os.path.join(args.gdrive_dir, 'output', dataset)
                
                if os.path.exists(output_src):
                    if not os.path.exists(output_dest):
                        os.makedirs(output_dest, exist_ok=True)
                    
                    # trained_models만 백업 (현재 이터레이션에서 학습된 모델들)
                    for model_name in trained_models:
                        model_src = os.path.join(output_src, model_name)
                        model_dest = os.path.join(output_dest, model_name)
                        
                        if os.path.exists(model_src):
                            if sys.version_info >= (3, 8):
                                shutil.copytree(model_src, model_dest, dirs_exist_ok=True)
                            else:
                                from distutils.dir_util import copy_tree
                                copy_tree(model_src, model_dest)
                    
                    print_and_save(f"    [output] Backup completed: {output_dest} ({len(trained_models)} models)", runs_log_path)
                
                backup_elapsed = time.time() - backup_start
                print_and_save(f"    Backup finished in {backup_elapsed:.2f}s", runs_log_path)
                
            except Exception as e:
                print_and_save(f"    [Warning] Backup failed: {e}", runs_log_path)
                import traceback
                traceback.print_exc()
        
        # Reset resume flag after first resumed iteration
        if args.resume and current_iter == resume_iter:
            resume_step = 1  # Reset for next iterations


    # Final Summary
    print_and_save(f"\n\n{'#'*70}", runs_log_path)
    print_and_save(f"# EVOLUTIONARY SEARCH COMPLETE", runs_log_path)
    print_and_save(f"{'#'*70}", runs_log_path)
    print_and_save(f"Total Iterations: {max_iter}", runs_log_path)
    print_and_save(f"Final Best Model: {overall_best_model}", runs_log_path)
    print_and_save(f"Final Best Test Acc: {overall_best_test_acc:.2f}%", runs_log_path)
    print_and_save(f"Final Best Val Acc: {overall_best_val_acc:.2f}%", runs_log_path)
    print_and_save(f"Detailed iteration logs saved to: {iter_log_path}", runs_log_path)
    print_and_save(f"{'#'*70}\n", runs_log_path)


    '''
    best_test_acc,best_val_acc=-1,-1
    runs_log_path = os.path.join(log_dir,'log_runs.log')
    for iter in range(1,max_iter+1):
        start = time.time()
        print_and_save(f"Iter: {iter}/{max_iter}",runs_log_path)
        
        # create model
        nader.run_one_iter(iter=iter,width=width)
        path = os.path.join(log_dir,f"train_full_iter{iter}.sh")
        models = nader.generate_train_script(cluster=cluster,batch_path=path)
        print_and_save(f"Create models:{models}",runs_log_path)
        log_dirs = [os.path.join(nader.train_log_dir,model,'1') for model in models]
        e1 = time.time()
        
        process = subprocess.Popen(f'bash {path}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while len(log_dirs)>0:
            ns = []
            status = True
            for d in log_dirs:
                path = os.path.join(d,'val_acc.txt')
                if not os.path.isfile(path):
                    n = 0
                else:
                    with open(path,'r') as f:
                        ds = f.readlines()
                        n = len(ds)-1
                ns.append(f"{n}/{epochs}")
                path = os.path.join(d,'train_status.txt')
                if not os.path.isfile(path):
                    status=False
            print_and_save(f"Process:{ns}",runs_log_path)
            if status:
                break
            time.sleep(60)
        print_and_save(f"Training finised!",runs_log_path)
        e2 = time.time()

        # summary
        block_list,test_acc = nader.research.mog_graph_manage.update_train_result(nader.train_log_dir,filename='test_acc',tag_prefix=tag_prefix_base,iter=iter,flush=True)
        block_list,val_acc = nader.research.mog_graph_manage.update_train_result(nader.train_log_dir,filename='val_acc',tag_prefix=tag_prefix_base,anno_name='mog_val',iter=iter,flush=True)
        best_test_acc = max(test_acc,best_test_acc)
        best_val_acc = max(val_acc,best_val_acc)
        print_and_save(f"Iter {iter}/{max_iter}, Val acc:{val_acc}, Test acc:{test_acc}, Best val acc{best_val_acc}, Best test acc:{best_test_acc}",runs_log_path)
    '''

