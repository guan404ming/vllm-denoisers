import json
import os
import pathlib
import re
import sys
import urllib.request

import modal

DEFAULT_MODEL = "inclusionAI/LLaDA2.0-mini"
GPU = os.environ.get("DENOISERS_GPU", "A100-80GB")
REF_TRANSFORMERS = os.environ.get("DENOISERS_REF_TRANSFORMERS", "5.2.0")
EOS = 156892
NIGHTLY_COMMIT = "fce64bc5c2c939a4521bfb9b462c8c74c1c13893"
NIGHTLY_WHEEL = (
    f"https://wheels.vllm.ai/{NIGHTLY_COMMIT}/"
    "vllm-0.32.1rc1.dev121%2Bgfce64bc5c-cp38-abi3-manylinux_2_28_x86_64.whl"
)
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
GSM8K_INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
PLUGIN_DIR = pathlib.Path(__file__).parent.parent
PROMPTS = [
    "What is 17 * 23?",
    "Write a haiku about the ocean.",
    "Explain why the sky is blue in two sentences.",
    "List three prime numbers greater than 100 and explain how you checked each one.",
    (
        "Summarize the plot of Romeo and Juliet for a ten year old reader, keeping it short, "
        "gentle, and focused on the main events of the story."
    ),
    (
        "You are a careful math tutor. A train leaves at 3 pm going 60 miles per hour, and "
        "another leaves at 4 pm going 80 miles per hour on the same track. When does the "
        "second train catch the first? Show each step."
    ),
    (
        "Translate the following sentence into French and then explain any grammar choices "
        "you made: The quick brown fox jumps over the lazy dog near the river bank."
    ),
    (
        "Write a Python function that returns the n-th Fibonacci number using iteration, "
        "then explain its time and memory complexity in plain words for a beginner."
    ),
]

app = modal.App("denoisers-llada2-eval")
hf_cache = modal.Volume.from_name("pagedcanvas-hf-cache", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache}
SECRETS = [modal.Secret.from_name("huggingface")]

ref_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.8.0", f"transformers=={REF_TRANSFORMERS}", "accelerate", "safetensors", "einops"
)
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(NIGHTLY_WHEEL)
    .add_local_dir(str(PLUGIN_DIR), "/root/plugin", copy=True, ignore=[".venv", "**/__pycache__"])
    .run_commands("pip install --no-deps /root/plugin")
    .env({"VLLM_ALLREDUCE_USE_FLASHINFER": "0"})
)


def first_eos(tokens: list[int]) -> list[int]:
    return tokens[: tokens.index(EOS) + 1] if EOS in tokens else tokens


def chat_ids(tokenizer, prompt: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )


@app.function(image=ref_image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=7200)
def hf_generate(model_id: str, prompts: list[str], gen_length: int) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    def greedy(logits, temperature=0.0, top_k=0, top_p=1.0):
        probs = torch.softmax(logits.float(), dim=-1)
        token = probs.argmax(dim=-1)
        return token, probs.gather(-1, token.unsqueeze(-1)).squeeze(-1)

    model._sample_with_temperature_topk_topp = greedy
    module = sys.modules[type(model).__module__]
    if hasattr(module, "create_bidirectional_mask"):
        module.create_bidirectional_mask = lambda attention_mask, **kw: (
            attention_mask.log() if attention_mask.max() > 0 else attention_mask
        )
    out = []
    for prompt in prompts:
        ids = chat_ids(tok, prompt)
        if model.config.model_type == "Fast_dLLM_Qwen":
            gen = model.generate(
                torch.tensor([ids], device="cuda"),
                max_new_tokens=gen_length,
                small_block_size=8,
                threshold=0.9,
                temperature=0,
            )[:, len(ids) :]
        else:
            gen = model.generate(
                inputs=torch.tensor([ids]), gen_length=gen_length, block_length=32, steps=32
            )
        tokens = first_eos(gen[0].tolist())
        out.append({"prompt_ids": ids, "tokens": tokens, "text": tok.decode(tokens)})
    return out


@app.function(image=vllm_image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=3600)
def vllm_generate(
    model_id: str,
    prompts: list[str],
    gen_length: int,
    hf_overrides: dict,
    active: int,
    tp: int,
    engine: dict | None = None,
    sampling: dict | None = None,
) -> list[dict]:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    args = {
        "enforce_eager": True,
        "attention_backend": "TRITON_ATTN",
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "max_model_len": 2048,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 8,
    }
    if active:
        os.environ["DENOISERS_ACTIVE"] = str(active)
        args["scheduler_cls"] = "denoisers.scheduler.resident_active.DenoisersAsyncScheduler"
        args["async_scheduling"] = True
    args.update(engine or {})
    llm = LLM(model=model_id, hf_overrides=hf_overrides, tensor_parallel_size=tp, **args)
    tok = llm.get_tokenizer()
    ids = [chat_ids(tok, prompt) for prompt in prompts]
    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=i) for i in ids],
        SamplingParams(max_tokens=gen_length, **(sampling or {})),
    )
    results = []
    for prompt_ids, o in zip(ids, outs):
        tokens = first_eos(list(o.outputs[0].token_ids))
        logprobs = o.outputs[0].logprobs
        results.append(
            {
                "prompt_ids": prompt_ids,
                "tokens": tokens,
                "text": tok.decode(tokens),
                "logprobs": None if logprobs is None else [len(lp) for lp in logprobs],
            }
        )
    return results


def run_both(
    tag: str,
    model_id: str,
    policy: str,
    prompts: list[str],
    gen_length: int,
    cache_dir: str,
    n_ref: int | None = None,
    active: int = 0,
    canvas: int = 0,
    tp: int = 1,
) -> tuple[list[dict], list[dict]]:
    cache = pathlib.Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    hf_overrides = {"decode_policy": policy} if policy else {}
    if canvas:
        hf_overrides["canvas_length"] = canvas
    sides = {
        "hf": (hf_generate, (model_id, prompts[:n_ref], gen_length)),
        "vllm": (vllm_generate, (model_id, prompts, gen_length, hf_overrides, active, tp)),
    }
    stem = f"{tag}_{model_id.split('/')[-1]}_{policy or 'default'}"
    if canvas:
        stem += f"_c{canvas}"
    if active:
        stem += f"_active{active}"
    pending = {}
    results = {}
    for name, (fn, args) in sides.items():
        path = cache / f"{stem}_{name}_{len(args[1])}x{gen_length}.json"
        if not args[1]:
            results[name] = []
        elif path.exists():
            results[name] = json.loads(path.read_text())
        else:
            pending[name] = (fn.spawn(*args), path)
    for name, (call, path) in pending.items():
        results[name] = call.get()
        path.write_text(json.dumps(results[name]))
    ref, got = results["hf"], results["vllm"]
    assert all(r["prompt_ids"] == g["prompt_ids"] for r, g in zip(ref, got))
    return ref, got


def show(ref: list[dict], got: list[dict]) -> int:
    exact = 0
    for i, g in enumerate(got):
        have = g["tokens"]
        print(
            f"prompt_len={len(g['prompt_ids'])} shift={len(g['prompt_ids']) % 32} got={len(have)}",
            end="",
        )
        if i < len(ref):
            want = ref[i]["tokens"]
            prefix = next(
                (j for j, (a, b) in enumerate(zip(want, have)) if a != b),
                min(len(want), len(have)),
            )
            exact += want == have
            print(f" ref={len(want)} match_prefix={prefix} exact={want == have}")
            print(f"  ref: {ref[i]['text']!r}")
        else:
            print()
        print(f"  got: {g['text']!r}")
    return exact


def final_answer(text: str) -> float | None:
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    source = (boxed[-1] if boxed else text).replace(",", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", source)
    return float(numbers[-1]) if numbers else None


@app.local_entrypoint()
def parity(
    model: str = DEFAULT_MODEL,
    policy: str = "",
    n_ref: int | None = None,
    active: int = 0,
    canvas: int = 0,
    tp: int = 1,
    cache_dir: str = "/tmp/denoisers-eval",
):
    ref, got = run_both("parity", model, policy, PROMPTS, 64, cache_dir, n_ref, active, canvas, tp)
    exact = show(ref, got)
    print(f"exact {exact}/{len(ref)}")


@app.local_entrypoint()
def gsm8k(
    model: str = DEFAULT_MODEL,
    policy: str = "",
    n: int = 100,
    n_ref: int = 30,
    gen_length: int = 256,
    cache_dir: str = "/tmp/denoisers-eval",
):
    with urllib.request.urlopen(GSM8K_URL) as resp:
        problems = [json.loads(line) for line in resp.read().decode().splitlines()[:n]]
    prompts = [p["question"] + GSM8K_INSTRUCTION for p in problems]
    gold = [float(p["answer"].split("####")[-1].strip().replace(",", "")) for p in problems]
    ref, got = run_both("gsm8k", model, policy, prompts, gen_length, cache_dir, n_ref)
    hf_ok = [final_answer(r["text"]) == g for r, g in zip(ref, gold)]
    vllm_ok = [final_answer(r["text"]) == g for r, g in zip(got, gold)]
    m = len(ref)
    agree = sum(a == b for a, b in zip(hf_ok, vllm_ok))
    print(f"gsm8k vllm n={n} acc={sum(vllm_ok) / n:.3f}")
    print(
        f"gsm8k paired n={m} hf={sum(hf_ok) / m:.3f} "
        f"vllm={sum(vllm_ok[:m]) / m:.3f} agree={agree}/{m}"
    )


@app.local_entrypoint()
def features(
    model: str = DEFAULT_MODEL,
    policy: str = "",
    canvas: int = 0,
    backend: str = "TRITON_ATTN",
    only: str = "",
    cache_dir: str = "/tmp/denoisers-eval",
):
    hf_overrides = {"decode_policy": policy} if policy else {}
    if canvas:
        hf_overrides["canvas_length"] = canvas
    graphs = {"enforce_eager": False, "attention_backend": backend}
    chunked = {
        **graphs,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 48,
        "max_num_seqs": 1,
        "scheduler_cls": "denoisers.scheduler.resident_active.DenoisersAsyncScheduler",
        "async_scheduling": True,
    }
    sampled = {"top_p": 0.9, "top_k": 50, "logprobs": 2}
    runs = {
        "graphs": (PROMPTS, 64, hf_overrides, graphs, {}),
        "sampled": (PROMPTS[:4], 64, {**hf_overrides, "temperature": 0.8}, graphs, sampled),
        "chunked": (PROMPTS, 64, hf_overrides, chunked, {}),
        "partial": (PROMPTS, 17, hf_overrides, {**graphs, "max_model_len": 90}, {}),
    }
    calls = {
        name: vllm_generate.spawn(model, prompts, length, overrides, 0, 1, engine, params)
        for name, (prompts, length, overrides, engine, params) in runs.items()
        if not only or name in only.split(",")
    }
    stem = f"parity_{model.split('/')[-1]}_{policy or 'default'}"
    if canvas:
        stem += f"_c{canvas}"
    cached = pathlib.Path(cache_dir) / f"{stem}_vllm_8x64.json"
    reference = json.loads(cached.read_text()) if cached.exists() else None
    for name, call in calls.items():
        try:
            got = call.get()
        except (RuntimeError, modal.exception.RemoteError) as exc:
            print(f"{name}: FAILED {type(exc).__name__}: {str(exc)[:300]}")
            continue
        print(f"{name}: ok")
        if name == "graphs" and reference:
            same = sum(r["tokens"] == g["tokens"] for r, g in zip(reference, got))
            print(f"  same as eager TRITON_ATTN: {same}/{len(got)}")
        for g in got:
            print(
                f"  len={len(g['tokens'])} logprobs={g['logprobs'] and g['logprobs'][:3]} {g['text'][:90]!r}"
            )
