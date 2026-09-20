import sys

from llama_cpp import Llama

from config import (
    MODEL_PATH,
    CONTEXT_SIZE,
    GPU_LAYERS,
    TEMPERATURE,
    MAX_TOKENS,
)


class ModelLoadError(Exception):
    """Raised when the local model fails to load."""


class LocalModel:

    def __init__(self):
        if not MODEL_PATH.exists():
            raise ModelLoadError(
                f"Model file not found at: {MODEL_PATH}\n"
                f"Put a GGUF model there, or update MODEL_PATH in config.py."
            )

        print("Loading local model...")
        print(f"Model: {MODEL_PATH}")

        try:
            # verbose=True on first load lets us confirm GPU offload
            # actually happened. llama-cpp-python will silently fall
            # back to CPU if the CUDA build isn't installed correctly,
            # so we surface that instead of hiding it.
            self.llm = Llama(
                model_path=str(MODEL_PATH),
                n_ctx=CONTEXT_SIZE,
                n_gpu_layers=GPU_LAYERS,
                verbose=True,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model: {exc}\n"
                f"Common causes: wrong/corrupt GGUF file, insufficient VRAM "
                f"for the requested n_gpu_layers, or a CPU-only build of "
                f"llama-cpp-python (no CUDA support)."
            ) from exc

        self._check_gpu_offload()

        print("Model loaded.\n")

    def _check_gpu_offload(self):
        """
        Best-effort check that layers actually landed on the GPU.
        llama.cpp prints offload info to stderr during load; we can't
        reliably capture that here, so instead we surface the metadata
        we do have access to and warn if it looks like a CPU-only build.
        """
        try:
            n_gpu = getattr(self.llm.model_params, "n_gpu_layers", None)
        except Exception:
            n_gpu = None

        if GPU_LAYERS != 0 and n_gpu == 0:
            print(
                "\nWARNING: n_gpu_layers is 0 despite GPU_LAYERS being set. "
                "Your llama-cpp-python build may not have CUDA support. "
                "Check the load log above for 'CUDA' / 'cuBLAS' mentions.\n",
                file=sys.stderr,
            )

    def generate(self, messages):
        try:
            stream = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=True,
            )

            for chunk in stream:
                choices = chunk.get("choices", [])

                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content")

                if content:
                    yield content

        except Exception as exc:
            yield f"\n[Error during generation: {exc}]"