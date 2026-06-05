================================================================
RETAIL INSIGHTS COPILOT  --  DEPLOYMENT RUNBOOKS
================================================================
Step-by-step "what to do" notes. The rationale (why Fargate, why DuckDB,
what we skip) lives in design/deployment.md. These files are just the steps.


Three moving parts
  Backend   : FastAPI + uvicorn (Docker image). Supervisor -> sql / web /
              forecast / chart agents -> synthesizer (SSE stream).
  Data      : hm.duckdb (~1.1 GB), a read-only single file.
  Frontend  : React + Vite static SPA (build -> dist/, no container).


Files
  docker-local.txt  : build + run the backend on Docker locally. VERIFIED.
  aws-deploy.txt    : host on AWS (backend Fargate + frontend S3/CloudFront).


Golden rules (both places)
  - .env (keys) NEVER in the image / git. Injected at runtime.
      git    : in .gitignore
      docker : in .dockerignore
      local  : docker run --env-file .env
      aws    : AWS Secrets Manager -> ECS task env var
  - DB (1.1 GB) is not baked into the image by default.
      local : volume-mount it
      aws   : either bake it in (demo, simplest) or download from S3 on startup (prod)
  - No deployment.py is needed to deploy. It is all AWS CLI / console.
    boto3 is only used inside the app (if the DB is downloaded from S3), not to deploy.
