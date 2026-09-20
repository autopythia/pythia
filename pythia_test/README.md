# Tests

Run the complete suite through the TAP frontend:

```bash
python3 -m pythia_test
```

The default worker count is the sum of the queried CPU performance levels. Each
`test_*.py` module runs serially in its own interpreter; independent modules run
in parallel. On a detected VM, guest-visible, affinity-allowed vCPUs are used
instead of treating the hypervisor's core topology as physical host topology;
available virtualization metadata is emitted as TAP comments under source-like
`lscpu` and DMI field names.

Use one worker when diagnosing ordering or load-sensitive failures:

```bash
python3 -m pythia_test --jobs 1
```

Specific modules and unittest IDs may be selected explicitly:

```bash
python3 -m pythia_test pythia_test.test_auto
python3 -m pythia_test \
  pythia_test.test_transport_retry.TransportRetryPolicyTests.test_retry_after_http_date_uses_supplied_clock
```

The test definitions remain standard-library `unittest` tests, so the legacy
targeted form remains available:

```bash
python3 -m unittest pythia_test.test_auto
```

The frontend emits TAP version 13 and returns a nonzero status for test or
worker failures. On systems with `prove`, output can be validated with:

```bash
python3 -m pythia_test > /tmp/pythia.tap
prove /tmp/pythia.tap
```
