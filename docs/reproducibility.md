# GPUBlind v2 Reproducibility

1. Migrate handwritten kernels:
   `python3 scripts/migrate.py --source kernels --profiles profiles`
2. Verify migrated corpus:
   `python3 corpus/verify.py --write`
3. Render prompts for inspection:
   `python3 eval/conditions.py --kernel corpus/kernels/hw_B`
4. Run mock eval:
   `python3 -m eval.run_eval --model mock --conditions 0,1,2 --mock --judge`
5. Generate report:
   `python3 -m analysis.report`
6. Validate:
   `python3 scripts/validate.py`
