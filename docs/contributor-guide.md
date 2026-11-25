# Developer guide

## Development

- [Runtime architecture](architecture.md) · Trace a request from the vLLM worker through the sampler to token emission.
- [Extend the plugin](extending.md) · Implement a policy or [register a model family](extending.md#register-a-model-family).

## Validation

- [Testing](testing.md) · Run local checks, compare tokens and answers with a reference, and verify vLLM upgrades.
- [Benchmarking](benchmarking.md) · Compare serving configurations on the same workload and measure latency, throughput, and goodput.

## Release

1. Configure a [PyPI trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) for `vllm-denoisers`. Use repository `guan404ming/vllm-denoisers`, workflow `release.yml`, and environment `pypi`.
2. Set the release version with `uv version <version>` and merge `pyproject.toml` and `uv.lock` into `main`. Keep MAJOR.MINOR equal to the supported vLLM version. Increase PATCH for each plugin release, for example `0.34.1`. Add a suffix such as `rc1` for a release candidate.
3. Tag the merged commit with `v<version>` and push that tag. The workflow tests the wheel and publishes the wheel and source distribution to PyPI.
