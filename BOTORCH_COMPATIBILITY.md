# BoTorch Compatibility Notes

## Version Compatibility

- **Tested with**: BoTorch 0.14.0
- **Minimum required**: BoTorch >= 0.10 (as specified in `pyproject.toml`)

## Known Issues and Workarounds

### 1. `optimize_acqf` with Custom Objectives

**Issue**: When using `optimize_acqf` with custom `MCMultiOutputObjective` implementations, there can be shape validation errors during initial condition generation. This occurs when `raw_samples > 1` and the optimizer tries to evaluate the acquisition function on multiple initial conditions simultaneously.

**Error Message**:
```
RuntimeError: The q-batch shape of the objective values does not agree with the q-batch shape of X. Got N and 1.
```

**Root Cause**: BoTorch's shape validation in `MCMultiOutputObjective.__call__` checks that the q-batch dimension of the objective output matches X's q dimension. During initial condition generation, X has shape `(raw_samples, q, d)`, and the validation can fail depending on how the objective handles batch dimensions.

**Workarounds**:

1. **Use `sequential=True`**: This processes candidates one at a time, avoiding batch shape issues:
   ```python
   candidate, acq_value = optimize_acqf(
       acq_function=acqf,
       bounds=bounds,
       q=1,
       num_restarts=5,
       raw_samples=20,
       sequential=True,  # Process one at a time
   )
   ```

2. **Use direct model sampling**: The objectives work correctly with direct `model.posterior().sample()` calls:
   ```python
   posterior = model.posterior(X)
   samples = posterior.sample(torch.Size([10]))
   obj_vals = objective.forward(samples)
   ```

3. **Reduce `raw_samples`**: Using fewer initial samples can sometimes avoid the issue:
   ```python
   candidate, acq_value = optimize_acqf(
       acq_function=acqf,
       bounds=bounds,
       q=1,
       num_restarts=5,
       raw_samples=4,  # Reduced from default
   )
   ```

### 2. `SmoothChebyshevSetObjective` with Acquisition Functions

**Issue**: `SmoothChebyshevSetObjective` removes the q dimension from the output (returns `(sample_shape x batch_shape)` instead of `(sample_shape x batch_shape x q)`). This can cause BoTorch's shape validation to fail when used with standard acquisition functions.

**Solution**: `SmoothChebyshevSetObjective` is designed for advanced use cases. As documented in its docstring, it should be used with `qSimpleRegret` or `FixedFeatureAcquisition`, or with direct model sampling rather than standard acquisition functions.

## Verified Working Configurations

The following configurations have been tested and work correctly:

1. `SmoothChebyshevObjective` with `qSimpleRegret` (single candidate)
2. `SmoothChebyshevObjective` with direct `model.posterior().sample()` calls
3. `SmoothChebyshevSetObjective` with direct `model.posterior().sample()` calls
4. Both objectives with gradient computation through acquisition functions

## Recommendations

1. **For standard use**: Use `SmoothChebyshevObjective` with `qSimpleRegret` or other standard acquisition functions. Use `sequential=True` when calling `optimize_acqf` if you encounter shape issues.

2. **For batch optimization**: Use `SmoothChebyshevSetObjective` with direct model sampling or custom optimization loops rather than `optimize_acqf`.

3. **For debugging**: Test your objective with direct model sampling first to verify it works correctly before using with acquisition functions.

## Testing

All integration tests in `tests/test_integration.py` verify that:
- Objectives work correctly with BoTorch models
- Objectives handle various input shapes correctly
- Gradients flow through acquisition functions
- Direct model sampling works for both objective types

Run tests with:
```bash
pytest tests/test_integration.py -v
```

