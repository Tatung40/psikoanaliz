# Container-based production deployment for the R-backed MFRM module.
# Streamlit Community Cloud does not build this Dockerfile.
FROM python:3.11-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base r-base-dev build-essential gfortran libblas-dev liblapack-dev \
    libcurl4-openssl-dev libssl-dev libxml2-dev libicu-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
# Exact validated TAM version; fail the image build if unavailable.
RUN Rscript -e 'options(repos=c(CRAN="https://cloud.r-project.org")); install.packages("remotes"); remotes::install_version("TAM",version="4.3-25",upgrade="never",dependencies=NA); stopifnot(as.character(packageVersion("TAM"))=="4.3-25")'
COPY . .
RUN python -c 'from core.mfrm import engine_available; ok,why=engine_available(); assert ok,why; print("MFRM R/TAM READY")'
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=5)" || exit 1
CMD ["streamlit","run","app.py","--server.address=0.0.0.0","--server.port=8501","--server.headless=true"]
