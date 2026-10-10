COMPOSE = docker compose -f docker-compose.yml

up:
	$(COMPOSE) up -d --wait
	@echo "vote:   http://localhost:8080"
	@echo "result: http://localhost:8081"

down:
	$(COMPOSE) down -v

status:
	$(COMPOSE) ps --format "table {{.Service}}\t{{.State}}\t{{.Health}}\t{{.Ports}}"

.PHONY: up down status
