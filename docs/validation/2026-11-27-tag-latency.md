# Tagger latency correction

The task's `gpt-6-luna` default reasoning was consuming the three-second
request window. The tagger now resolves its native Hermes route and uses the
native request builder with `reasoning_config={"enabled": false}`. Hermes
selects the provider-compatible wire parameters and known minimum reasoning
requirements. The single request, profile credentials, timeout cap and speech
validation remain in place. Plugin version: **0.1.4**.

## Local measurements

The baseline comparison on 2026-11-24 used the same model, prompts, credentials
and three-second request timeout, alternating default/off request order.
Inputs were synthetic Brazilian Portuguese scripts.

| Case | Characters | Default reasoning | Reasoning disabled |
| --- | ---: | --- | --- |
| Short | 61 | 3021 ms, 42 reasoning tokens | 1033 ms, 0 reasoning tokens |
| Medium | 209 | APITimeoutError, 3010 ms | 1140 ms, 0 reasoning tokens |
| Long | 560 | APITimeoutError, 3067 ms | 1802 ms, 0 reasoning tokens |

On 2026-11-27, the candidate's actual `_call_configured_tag_model` completed
all six requests in one finite run:

| Case | Characters | Elapsed | Accepted tags | Reasoning tokens |
| --- | ---: | ---: | ---: | ---: |
| Short | 61 | 2823 ms | 1 | 0 |
| Medium | 209 | 1146 ms | 1 | 0 |
| Long | 560 | 1189 ms | 1 | 0 |
| Factual | 124 | 787 ms | 0 | 0 |
| Playful | 68 | 841 ms | 1 | 0 |
| Instruction-shaped speech | 163 | 1344 ms | 1 | 0 |

All six outputs preserved the original spoken words and punctuation. A factual
script returning no tags is valid. These small samples demonstrate the cause
and the corrected request; they do not guarantee provider latency under load.
Timing includes local routing/client setup. The SDK timeout remains capped at
three seconds; it is not an absolute wall-clock deadline across all HTTP phases.

## Regression checks

- Native Hermes runtime source: `2bd378ed1b8d6bf2192cb26e0a55ff3091e6bd19`.
- Python 3.14.7, OpenAI SDK 2.54.0, pytest 8.4.2.
- `python -m pytest -q` with that runtime on `PYTHONPATH`: **30 passed**.
- Without an importable Hermes runtime: unit tests run independently; native
  routing tests are skipped explicitly.
- Native HTTP transport tests inspect the outgoing request for `openai` and a
  custom endpoint with `key_env`: correct host, profile credential, model,
  `reasoning_effort=none`, no legacy token/sampling arguments, and one request.
- HTTP 503 and SDK read timeout preserve the full original script without
  another model attempt. Lower task timeouts and invalid timeout configuration
  retain their previous behavior.
- `hermes plugins doctor . --ci`: runtime discovery, manifest, import and
  registration passed. Ruff and `git diff --check` passed.
- Changed function complexity: 9. Its executable statements are fully covered.
- UB tools: no duplicate candidate >= 0.8; one changed-line mutant killed, none
  surviving or uncovered. The tools misclassify the `/Users/test` parent as a
  test directory, so metrics ran against an identical temporary clone under
  `/tmp`, with the real native runtime and scoped tests.

## Reproduction and remaining scope

Run the native test with `PYTHONPATH=<Hermes runtime>` and an isolated
`HERMES_HOME`: `python -m pytest -q tests/test_task_routing_hermes.py`.
For live timing, configure the QA profile's explicit auxiliary route to
`openai` / `gpt-6-luna`, enter its native profile/secret scope, and time
`rewrite_audio_script(script, max_tags=2)` on short, medium, long, factual,
playful and instruction-shaped synthetic scripts. Record only duration,
exception class, token counts, accepted tag count and speech preservation.

**Staging E2E and WhatsApp delivery: NOT RUN for this correction.** No CRM
configuration, installed production plugin, agent state or WhatsApp message was
changed by these local measurements. WhatsApp delivery is the next validation
phase, as requested.
