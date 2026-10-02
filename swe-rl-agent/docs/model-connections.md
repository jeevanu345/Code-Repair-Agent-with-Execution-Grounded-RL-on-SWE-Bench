# Local and cloud inference

The repair agent has one inference interface, `LLMClient`. It selects a local
vLLM server or a cloud API through a shared connection profile. Provider choice
changes inference, not Docker sandboxing, test evaluation, or training.

## Providers

The GUI includes local vLLM, OpenAI, Anthropic/Claude, NVIDIA NIM, Google/Gemini,
Alibaba Cloud/Qwen, DeepSeek, Mistral, xAI/Grok, Groq, Together AI, Fireworks AI,
OpenRouter, and a custom OpenAI-compatible endpoint.

The registry links each provider's official documentation. Endpoints were reviewed
on 2026-10-02. Enter a model ID from your account or use Fetch models; access and
model availability vary by account. A model list is not proof of generation.
Test connection verifies a text completion. Both discovery and testing contact
the provider; testing may incur charges.

Qwen is a model family hosted by Alibaba Cloud and other providers, not an
independent key issuer. Select the Alibaba region matching the key. Custom
supports other OpenAI-compatible services; Azure deployment-specific API versions
and AWS Bedrock signatures are not implemented.

## GUI

Open Model connections, choose a company, enter a model ID and your key, and
Save connection. Saving makes no provider request. Blank keys retain the existing
key only for the same provider and endpoint. Switching providers replaces the
single active profile. Remove saved connection deletes it and restores environment
defaults. This is a local configuration interface, not a multi-user hosted service.

The repair assistant sends a user-supplied issue/code snippet through the active
connection. It returns suggestions without executing them. Full execution-grounded
rollouts remain available through `python -m swe_rl.cli rollout one INSTANCE_ID`.
CLI rollout/evaluation jobs started from the same working directory read the saved
profile. Existing running clients keep their original connection until restarted.

## Configuration and credentials

Precedence: explicit client configuration, GUI profile, environment defaults.
The profile defaults to `.llm-profile.json` in the working directory. Git ignores
it; writes are atomic with owner-only permissions (0600). Secrets are plaintext
on local disk, not encrypted. API responses omit keys, errors avoid provider response
bodies, and frontend keys are not put in localStorage. Use the app only on loopback.
Trusted Host validation and a same-origin settings token protect mutations.
This is not user authentication for a remotely deployed service.

Environment-only configuration uses `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`
and `LLM_API_KEY`. No credential is supplied by the repository. With no profile,
local vLLM retains `MODEL_NAME`, `VLLM_HOST`, `VLLM_PORT`, and `VLLM_API_KEY`.
Remote custom endpoints require HTTPS; loopback endpoints may use HTTP.

## Architecture

GUI or CLI → shared profile → provider adapter → normalized response
→ ReAct tools → Docker evaluation → trajectory and execution reward.

OpenAI-compatible services use chat completions. Claude uses Messages, with
system prompts separated and token usage normalized. OpenAI uses
`max_completion_tokens`. Cloud requests omit vLLM sampling/seed parameters to
avoid incompatibility with reasoning models. Custom and local vLLM retain them.

Ray receives a connection snapshot with its server-side key when actors start;
only use trusted Ray nodes. Training still loads locally accessible model weights
and requires the explicit heavy-training opt-in. A cloud API key cannot train a
remote provider's model through local TRL. Optional PRM judging retains its separate
judge configuration. Paid API costs require configured token prices; GUI requests
are short, explicit calls and are not governed by rollout cost meters.

## Verification

Mock transports test provider routing and headers without spending credits.
Tests cover Claude protocol, usage normalization, secure file permissions, default
client selection, missing credentials, redacted errors, cross-origin rejection,
and GUI assistant routing. No real key or model weights were added.
Live provider generation requires a user's key and available model.
