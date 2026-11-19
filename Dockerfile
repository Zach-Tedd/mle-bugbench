FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY tasks ./tasks
RUN uv sync --frozen --no-dev
CMD ["uv", "run", "--no-dev", "python", "-m", "bugbench.runner", "validate"]
