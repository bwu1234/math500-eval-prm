PYTHON ?= python

.PHONY: help test smoke audit regrade report clean

help:
	@echo "test    Run the unit suite (no GPU, no API key, ~1s)"
	@echo "smoke   Exercise the full pipeline with the mock backend"
	@echo "audit   Sweep MATH-500 for answer-grading collisions"
	@echo "regrade Re-grade a finished run against the current grader (dry run)"
	@echo "report  Rebuild report.html from existing results"
	@echo "clean   Remove run outputs and caches"

test:
	$(PYTHON) -m pytest tests/ -q

smoke:
	$(PYTHON) math500_eval.py -n 20 -m mock --no-prm --quiet

audit:
	$(PYTHON) scripts/audit_normaliser.py

regrade:
	$(PYTHON) scripts/regrade.py

report:
	$(PYTHON) math500_eval.py --report-only

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .ruff_cache
	rm -f eval_results*.json eval_comparison.json eval_debug*.log \
	      .eval_checkpoint.jsonl report.html report_*.html *_partial.html
