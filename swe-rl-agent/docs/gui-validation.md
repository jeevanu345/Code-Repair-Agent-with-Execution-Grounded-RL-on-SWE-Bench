# Console validation

The enterprise UI uses the supplied design and the operational control-center API.

## Terminal interaction checks

```bash
npm install --prefix /tmp/swe-rl-ui jsdom@26
NODE_PATH=/tmp/swe-rl-ui/node_modules node tests/control-center-ui.cjs
```

Coverage includes navigation, measured metrics/empty states, run/evaluation/training
payloads, SSE events and lifecycle controls, split/unified patches, test logs,
conversation, keyboard navigation, provider security/discovery, registry/prompts,
dataset filters, artifact escaping and API failures. Legacy dashboard/connections
entry points run the same regression suite.

Python unit tests mock external services and trainer calls to check local queue
persistence, configuration, controls, credential references, prompt immutability,
artifacts, parallel evaluation and execution reward wiring. These are not live
Docker, paid-provider or GPU end-to-end certifications.

No computer-use automation was used. Target-host workload and visual/device
validation remain distinct from terminal DOM assertions. See
[control center](control-center.md) for runtime prerequisites and scoped limitations.
