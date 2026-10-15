"""
Model-agnostic tool-calling backend, so the chat is not tied to one vendor.

Two wire protocols cover essentially everything worth running:

  "openai"     any OpenAI-compatible /chat/completions endpoint with tools.
               That is Groq, Together, Fireworks, DeepInfra, OpenRouter, vLLM,
               llama.cpp's server, and Ollama (http://localhost:11434/v1) — so
               one adapter serves both hosted open-weights inference and a
               model running on your own machine.
  "anthropic"  the Claude Messages API.

Select with LLM_BACKEND; everything else is LLM_* env vars. No vendor SDK is
required for the openai path — it is plain `requests` against the documented
JSON, which is also why it works against a local llama.cpp with no extra
dependency.

    LLM_BACKEND=openai
    LLM_BASE_URL=https://api.groq.com/openai/v1
    LLM_API_KEY=...
    LLM_MODEL=llama-3.3-70b-versatile        # whatever that provider lists

    LLM_BACKEND=openai                       # fully local, no key needed
    LLM_BASE_URL=http://localhost:11434/v1
    LLM_MODEL=qwen3:14b

Model names move around; check your provider's current list rather than
trusting the examples above. What matters is that the model supports *tools* —
without that the whole design collapses, because the assistant's only job is
to choose tool calls.

Open-weights models are measurably weaker at following a long list of "never do
X" rules than the frontier models this prompt was written against. Expect more
leakage (artist names inside descriptions, negation in prose). Two defences
live here: `force_tool` can require a call rather than hope for one, and
malformed tool arguments are reported back to the model as an error it can
retry instead of raising. The third lives in crate_chat: if the model will not
call a tool at all on the opening turn, the chat falls back to a deterministic
description search so the user still gets tracks.
"""

import json
import os

BACKEND = os.environ.get("LLM_BACKEND", "anthropic").strip().lower()
MODEL = os.environ.get("LLM_MODEL", "").strip()
BASE_URL = os.environ.get("LLM_BASE_URL", "").strip().rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "").strip()
TIMEOUT = int(os.environ.get("LLM_TIMEOUT", 120))
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", 1200))
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", 0.4))

# Anthropic prompt caching. The static prefix here is large — ~2,500 tokens of
# system prompt plus ~1,200 of tool schemas — and it is resent on EVERY API
# call. Note that is per API call, not per user turn: the tool loop makes two to
# four calls for a single thing the user says, so the static prefix is paid for
# two to four times per message without caching.
#
# A cache_control breakpoint caches everything from the start of the prompt up
# to and including that block, and the prefix is ordered tools -> system ->
# messages. So one breakpoint on the last system block covers both the tools and
# the system prompt.
#
# CACHE_MESSAGES adds a second breakpoint at the end of the conversation so far,
# which makes each call in a tool loop reuse the previous call's prefix too.
# Writes cost more than ordinary input and reads cost much less, so this is a
# win whenever a prefix is read at least once — which inside a tool loop is
# immediate. Turn it off to compare.
CACHE = os.environ.get("LLM_CACHE", "1") not in ("0", "false", "no")
CACHE_MESSAGES = os.environ.get("LLM_CACHE_MESSAGES", "1") not in ("0", "false", "no")

DEFAULT_MODEL = {"anthropic": "claude-sonnet-5",
                 "openai": "llama-3.3-70b-versatile"}


class ToolCall(object):
    """One requested tool call, normalised across protocols."""

    def __init__(self, id, name, args, bad_args=None):
        self.id = id
        self.name = name
        self.args = args or {}
        # Set when the model emitted arguments that would not parse. The loop
        # feeds this back as a tool error so it can correct itself, which open
        # models occasionally need and frontier ones rarely do.
        self.bad_args = bad_args

    def __repr__(self):
        return "ToolCall(%s, %r)" % (self.name, self.args)


class Reply(object):
    """A model turn: free text, and/or tool calls, plus token accounting."""

    def __init__(self, text="", tool_calls=None, raw=None, usage=None):
        self.text = text or ""
        self.tool_calls = tool_calls or []
        self.raw = raw
        # {input, output, cache_write, cache_read}. cache_read > 0 is the only
        # proof that caching is actually working — assume nothing.
        self.usage = usage or {}


# ── neutral message helpers ──────────────────────────────────────────────────
# History is kept in one provider-agnostic shape and translated on the way out:
#   {"role": "user",      "content": "..."}
#   {"role": "assistant", "text": "...", "tool_calls": [ToolCall, ...]}
#   {"role": "tool",      "tool_call_id": "...", "name": "...", "content": "..."}

def user(text):
    return {"role": "user", "content": text}


def assistant(reply):
    return {"role": "assistant", "text": reply.text, "tool_calls": reply.tool_calls}


def tool_result(call, content):
    return {"role": "tool", "tool_call_id": call.id, "name": call.name,
            "content": content if isinstance(content, str) else json.dumps(content)}


# ── backends ─────────────────────────────────────────────────────────────────

class AnthropicBackend(object):
    name = "anthropic"

    def __init__(self, model=None):
        self.model = model or MODEL or DEFAULT_MODEL["anthropic"]
        self._client = None

    def _c(self):
        if self._client is None:
            import anthropic
            kw = {}
            if API_KEY:
                kw["api_key"] = API_KEY
            self._client = anthropic.Anthropic(**kw)
        return self._client

    def _tools(self, tools):
        return [{"name": t["name"], "description": t["description"],
                 "input_schema": t["input_schema"]} for t in tools]

    def _system(self, system):
        """System prompt as blocks, with the cache breakpoint on the last one.

        Because the cached prefix runs tools -> system -> messages, this single
        breakpoint also captures the tool schemas sitting in front of it.
        """
        block = {"type": "text", "text": system}
        if CACHE:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    @staticmethod
    def _mark_last(messages):
        """Second breakpoint: the end of the conversation so far.

        Lets call N of a tool loop reuse call N-1's prefix instead of only the
        static header. Content is normalised to a block list so the marker has
        somewhere to live.
        """
        if not messages:
            return messages
        last = messages[-1]
        content = last.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
            last["content"] = content
        if isinstance(content, list) and content:
            content[-1] = dict(content[-1])
            content[-1]["cache_control"] = {"type": "ephemeral"}
        return messages

    def _messages(self, history):
        out = []
        for m in history:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = []
                if m.get("text"):
                    blocks.append({"type": "text", "text": m["text"]})
                for c in m.get("tool_calls") or []:
                    blocks.append({"type": "tool_use", "id": c.id,
                                   "name": c.name, "input": c.args})
                out.append({"role": "assistant", "content": blocks or [
                    {"type": "text", "text": "(no content)"}]})
            elif m["role"] == "tool":
                # Anthropic carries tool results as user-role blocks, and
                # consecutive results must be merged into one message.
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"],
                         "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def complete(self, system, history, tools, force_tool=False):
        messages = self._messages(history)
        if CACHE and CACHE_MESSAGES:
            messages = self._mark_last(messages)
        kw = dict(model=self.model, max_tokens=MAX_TOKENS,
                  system=self._system(system),
                  tools=self._tools(tools), messages=messages)
        if force_tool:
            kw["tool_choice"] = {"type": "any"}
        r = self._c().messages.create(**kw)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        calls = [ToolCall(b.id, b.name, b.input) for b in r.content
                 if getattr(b, "type", "") == "tool_use"]
        u = getattr(r, "usage", None)
        usage = dict(
            input=getattr(u, "input_tokens", 0) or 0,
            output=getattr(u, "output_tokens", 0) or 0,
            cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
            cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
        ) if u is not None else {}
        return Reply(text, calls, raw=r, usage=usage)


class OpenAIBackend(object):
    """Any OpenAI-compatible /chat/completions endpoint that supports tools."""

    name = "openai"

    def __init__(self, model=None, base_url=None, api_key=None):
        self.model = model or MODEL or DEFAULT_MODEL["openai"]
        self.base_url = (base_url or BASE_URL or "http://localhost:11434/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else API_KEY

    def _tools(self, tools):
        return [{"type": "function",
                 "function": {"name": t["name"], "description": t["description"],
                              "parameters": t["input_schema"]}} for t in tools]

    def _messages(self, system, history):
        out = [{"role": "system", "content": system}]
        for m in history:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("text") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {"id": c.id, "type": "function",
                         "function": {"name": c.name,
                                      "arguments": json.dumps(c.args)}}
                        for c in m["tool_calls"]]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                            "name": m["name"], "content": m["content"]})
        return out

    def complete(self, system, history, tools, force_tool=False):
        import requests
        body = {"model": self.model,
                "messages": self._messages(system, history),
                "tools": self._tools(tools),
                "tool_choice": "required" if force_tool else "auto",
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer %s" % self.api_key
        r = requests.post(self.base_url + "/chat/completions", json=body,
                          headers=headers, timeout=TIMEOUT)
        if r.status_code >= 400:
            raise RuntimeError("%s %s: %s" % (self.base_url, r.status_code,
                                              r.text[:400]))
        data = r.json()
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        uu = data.get("usage") or {}
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw = fn.get("arguments")
            args, bad = {}, None
            if isinstance(raw, dict):
                args = raw                      # some servers pre-parse it
            else:
                try:
                    args = json.loads(raw or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                except Exception as e:
                    bad = "could not parse arguments (%s): %s" % (e, str(raw)[:200])
            calls.append(ToolCall(tc.get("id") or ("call_%d" % len(calls)),
                                  fn.get("name") or "", args, bad_args=bad))
        return Reply(msg.get("content") or "", calls, raw=data,
                     usage=dict(input=uu.get("prompt_tokens", 0) or 0,
                                output=uu.get("completion_tokens", 0) or 0,
                                cache_write=0, cache_read=0))


def get_backend(backend=None, model=None):
    b = (backend or BACKEND).strip().lower()
    if b in ("anthropic", "claude"):
        return AnthropicBackend(model=model)
    if b in ("openai", "openai-compatible", "ollama", "groq", "together",
             "fireworks", "openrouter", "deepinfra", "vllm", "llamacpp", "local"):
        return OpenAIBackend(model=model)
    raise ValueError("unknown LLM_BACKEND %r (use 'openai' or 'anthropic')" % b)


def describe():
    """One line for /api/chat/config, so the UI can say what is answering."""
    b = (BACKEND or "").lower()
    if b in ("anthropic", "claude"):
        return {"backend": "anthropic", "model": MODEL or DEFAULT_MODEL["anthropic"],
                "configured": bool(API_KEY or os.environ.get("ANTHROPIC_API_KEY"))}
    return {"backend": "openai", "model": MODEL or DEFAULT_MODEL["openai"],
            "base_url": BASE_URL or "http://localhost:11434/v1",
            # A local runtime needs no key, so a blank key is not a misconfiguration.
            "configured": bool(BASE_URL or API_KEY)}
