COMPOSE ?= docker compose
API_URL ?= http://localhost:8100
READYZ_TIMEOUT ?= 60
STATEWAVE_COLD_TIMING_FILE ?= /tmp/statewave_cold_ready_seconds

.PHONY: test-cold
test-cold:
	@test -f .env || cp .env.example .env
	$(COMPOSE) down -v
	$(COMPOSE) up -d
	@chmod +x scripts/wait_readyz.sh
	@STATEWAVE_COLD_TIMING_FILE=$(STATEWAVE_COLD_TIMING_FILE) \
		scripts/wait_readyz.sh "$(API_URL)" $(READYZ_TIMEOUT)
	@python3 -m pytest tests/smoke/ -m smoke -v
	@echo "Time to first ready: $$(cat $(STATEWAVE_COLD_TIMING_FILE))s"
