================================================================
RETAIL INSIGHTS COPILOT  --  DEPLOYMENT RUNBOOKS
================================================================

Ye folder step-by-step "kya karna hai" wale notes hain. Theory/rationale
(kyun Fargate, kyun DuckDB, kya skip kiya) design/deployment.md me hai.
Ye files sirf ACTIONABLE steps hain.

----------------------------------------------------------------
3 moving parts
----------------------------------------------------------------
  Backend   : FastAPI + uvicorn (Docker image). LangGraph supervisor ->
              sql / web / forecast / chart agents -> synthesizer (SSE stream).
  Data      : hm.duckdb (~1.1 GB), read-only single file.
  Frontend  : React + Vite static SPA (build -> dist/, koi container nahi).

----------------------------------------------------------------
Files
----------------------------------------------------------------
  docker-local.txt  : backend ko Docker me build + run karna (local). VERIFIED.
  aws-deploy.txt    : AWS pe host karna (backend Fargate + frontend S3/CloudFront).

----------------------------------------------------------------
Golden rules (dono jagah)
----------------------------------------------------------------
  - .env (keys) KABHI image me / git me nahi. Runtime pe inject hoti.
      git:    .gitignore me hai
      docker: .dockerignore me hai
      local:  docker run --env-file .env
      aws:    AWS Secrets Manager -> ECS task env var
  - DB (1.1 GB) image me bake nahi (default). Local: volume-mount.
      AWS: ya image me bake (demo, simplest) ya S3 se startup-download (prod).
  - Deploy karne ke liye koi deployment.py nahi. Sab AWS CLI / console.
    boto3 sirf app ke andar (agar DB S3 se download karein), deploy ke liye nahi.
