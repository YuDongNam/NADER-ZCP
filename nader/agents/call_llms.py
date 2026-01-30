import time
import os
import gc

from openai import OpenAI

# Global client cache to avoid recreating clients repeatedly (memory leak prevention)
_client_cache = {}

def open_proxy():
    os.environ["HTTP_PROXY"] = "http://0.0.0.0:7890"
    os.environ["HTTPS_PROXY"] = "http://0.0.0.0:7890"

def close_proxy():
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)

def _get_client(model_name):
    """Get or create a cached OpenAI client to prevent memory leaks."""
    global _client_cache
    
    if 'deepseek' in model_name:
        cache_key = 'deepseek'
        if cache_key not in _client_cache:
            _client_cache[cache_key] = OpenAI(
                api_key=os.environ['API_KEY_DEEPSEEK'], 
                base_url="https://api.deepseek.com", 
                max_retries=10
            )
        return _client_cache[cache_key]
    elif 'gpt' in model_name:
        cache_key = 'openai'
        if cache_key not in _client_cache:
            _client_cache[cache_key] = OpenAI(max_retries=10)
        return _client_cache[cache_key]
    else:
        raise NotImplementedError(f"Unknown model: {model_name}")


def call_llm(messages,max_try=50,use_proxy=False,*agrs,**kwds):
    model_name = os.environ['LLM_MODEL_NAME']
    if isinstance(messages,str):
        messages = [{"role":"user","content":messages}]
    if use_proxy:
        open_proxy()
    
    # Use cached client instead of creating new one each time
    client = _get_client(model_name)
    
    for i in range(max_try):
        try:
            response = client.chat.completions.create(model=model_name, messages=messages, timeout=60)
            break
        except BaseException as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            time.sleep(3)
            if i!=max_try-1:
                print(str(e)+'retry', flush=True)
                continue
            elif (i+1)%10:
                time.sleep(3)
            raise
    
    if use_proxy:
        close_proxy()
        
    response = {
        "prompt_tokens":response.usage.prompt_tokens,
        "completion_tokens":response.usage.completion_tokens,
        'model':response.model,
        "content":response.choices[0].message.content
    }
    
    # Force garbage collection to prevent memory accumulation
    gc.collect()
    
    return response

if __name__=='__main__':
    for i in range(1):
        message = [{"role": "user", "content": """鲁迅为什么暴打周树人"""}]
        res = call_llm(message)
        print(res)