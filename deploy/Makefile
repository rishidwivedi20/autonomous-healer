COMPOSE = docker compose -f docker-compose.yml

up:
	$(COMPOSE) up -d --wait
	@echo "vote:   http://127.0.0.1:8080"
	@echo "result: http://127.0.0.1:8081"

down:
	$(COMPOSE) down -v

status:
	$(COMPOSE) ps

.PHONY: up down status
