# MFRM production deployment

The existing deployment described in README is **Streamlit Community Cloud**. It does **not** execute Dockerfiles. This repository therefore cannot claim the R-backed MFRM works on that existing deployment merely because CI passes.

For a container-capable host, build the included Dockerfile, which installs R and the exact validated TAM 4.3-25 version and fails the build if the engine is unavailable:

```sh
docker build -t psikoanaliz-mfrm .
docker run --rm -p 8501:8501 psikoanaliz-mfrm
```

After starting, open the MFRM page and run a shipped 3-facet fixture, verifying analysis and DOCX download. A real deployment smoke test remains **NOT VERIFIED** until a host runs the image and tests this end-to-end.

This is a deployment *alternative*, not an automatic migration of the existing Streamlit Community Cloud app. The image build may be resource intensive because TAM dependencies compile from source. Before switching hosting, confirm resource limits, security, startup time and ongoing cost. Do not claim readiness or merge based on this file alone.

## Streamlit Community Cloud feasibility (checked 2026-11-09)

- Community Cloud installs system software only through `packages.txt`, which holds apt package names.
  [Streamlit docs: App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
  describes no build hook or custom install command. Users report a Debian trixie base, and an apt outage in
  September 2026 that blocked every `packages.txt` app until a fix shipped
  ([forum](https://discuss.streamlit.io/t/apt-get-fails-on-community-cloud-bullseye-security-release-file-expired-all-apps-with-packages-txt/122420)).
- Debian has no `r-cran-tam` package; the only near match found was `r-cran-tm`. apt cannot provide the validated
  TAM 4.3-25. Installing `r-base` without TAM would still fail `core.mfrm.engine_available()`, so no `packages.txt` was added.
- Compiling TAM and CDM from source while the app starts would be unpinned, slow and impossible to verify from CI, so it
  was not done.
- **Result:** on Community Cloud the MFRM page loads and validates data. The run button stays visible but disabled, with a
  Turkish explanation. No estimate is produced and no alternative estimator is used.
- **Working path:** use the `Dockerfile` image, which builds TAM 4.3-25 and fails if `engine_available()` is false, or any
  host with R and TAM 4.3-25 installed. Live hosting status: **NOT VERIFIED.**
