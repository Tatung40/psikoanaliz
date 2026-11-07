# MFRM production deployment

The existing deployment described in README is **Streamlit Community Cloud**. It does **not** execute Dockerfiles. This repository therefore cannot claim the R-backed MFRM works on that existing deployment merely because CI passes.

For a container-capable host, build the included Dockerfile, which installs R and the exact validated TAM 4.3-25 version and fails the build if the engine is unavailable:

```sh
docker build -t psikoanaliz-mfrm .
docker run --rm -p 8501:8501 psikoanaliz-mfrm
```

After starting, open the MFRM page and run a shipped 3-facet fixture, verifying analysis and DOCX download. A real deployment smoke test remains **NOT VERIFIED** until a host runs the image and tests this end-to-end.

This is a deployment *alternative*, not an automatic migration of the existing Streamlit Community Cloud app. The image build may be resource intensive because TAM dependencies compile from source. Before switching hosting, confirm resource limits, security, startup time and ongoing cost. Do not claim readiness or merge based on this file alone.
