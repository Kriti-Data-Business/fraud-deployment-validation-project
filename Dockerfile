FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /srv
# LightGBM needs the OpenMP runtime on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY run_triage.py run_live.py ./
RUN useradd --system --uid 10001 --create-home appuser && mkdir -p reports && chown -R appuser /srv
USER appuser
# Default: rebuild the Part A model and run the Part C test bench (about a minute).
# Live models:  docker run -e OPENROUTER_API_KEY=... twohats python run_live.py --models <slug> ...
CMD ["python", "run_triage.py"]
